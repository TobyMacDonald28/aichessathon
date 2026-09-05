"""Trains a HalfKP-style NNUE evaluation net on the Lichess evaluations database.

Two phases, run separately so a slow download never has to repeat while you iterate on the model:

    python train.py fetch --positions 1000000      # streams a sample into mlp/data_v2/
    python train.py fit --epochs 20                # trains on mlp/data_v2/, writes weights.pt

`--data-dir` overrides the dataset location on both (default mlp/data_v2/) — the original
mlp/data/ predates castling features (different MAX_ACTIVE row width, so it's not just
missing phase.f32, it's binary-incompatible with the fetch/fit code below) and is kept around
untouched rather than overwritten, not as something to point `--data-dir` back at.

Data source: https://database.lichess.org/#evals (CC0) — about 395M Stockfish-annotated
positions in a single ~20 GB zstd-compressed JSONL file, one line per position, each carrying one
or more engine lines with a depth, a centipawn or mate score, and the PV. `fetch` pipes
`curl | zstd -dc` and stops as soon as it has collected the requested number of positions, so the
compressed file is never written to disk and a partial run costs only the bytes it actually read.

Training is on the annotated engine score alone (regression), not on game outcomes. weights.pt is
just a state_dict for the NNUE module below — not wired into an agent yet, and not quantized:
this is the float32 architecture stage, quantization and the incrementally-updated accumulator
that make it fast enough to search with are separate follow-on work.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import chess
import numpy as np
import torch
from torch import nn

DATA_DIR = Path(__file__).parent / "data_v2"
WEIGHTS_PATH = Path(__file__).parent / "weights.pt"

EVAL_URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"
TARGET_CLIP_CP = 1000.0
MIN_DEPTH = 16
WDL_SCALE_CP = 400.0  # centipawns that map to a 1-sigma shift in sigmoid win-probability space
SCORE_CP_LIMIT = 800.0  # drop positions this sharp/near-decisive, raw score, before any clipping
DEPTH_GAP_CP_LIMIT = 50.0  # max disagreement between a shallow(>=MIN_DEPTH) and the deepest pass

# ---------------------------------------------------------------------------
# HalfKP feature encoding: the thing that makes this NNUE rather than a plain
# MLP. Each position is seen twice, once from each side's own king. A feature
# is "there is a <piece type, friend-or-foe> on <square>, given my king is on
# <king square>" — so the same knight-on-f3 fact is a different feature
# depending on where the viewer's own king is. Kings themselves aren't
# encoded as pieces (that's the "half" in HalfKP): the king square is the
# anchor the 40,960 piece features are relative to, not a feature itself.
# 4 more features (see castling_indices below) round FEATURE_DIM out to 40,964.
# ---------------------------------------------------------------------------

PIECE_INDEX_NO_KING = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
}
PIECE_FEATURE_DIM = 64 * 64 * 10  # king square x piece square x (5 piece types x friend/foe)

# Castling rights as 4 global facts — (mine/theirs) x (kingside/queenside) — appended after the
# piece-feature block. Unlike every other HalfKP feature these are deliberately NOT multiplied by
# king_square: "I can still castle kingside" is one fact, true or false, not 64 different facts
# depending on exactly which square my king happens to occupy. Each is a single fixed embedding
# row, present or absent in the EmbeddingBag sum like any other active feature.
CASTLING_FEATURE_DIM = 4
FEATURE_DIM = PIECE_FEATURE_DIM + CASTLING_FEATURE_DIM
PAD_INDEX = FEATURE_DIM  # one extra embedding row, held at zero, for padding short bags
MAX_ACTIVE = 34  # <=30 non-king pieces + <=4 simultaneously-active castling-right facts


def _orient(square: int, perspective: bool) -> int:
    """Square as seen by `perspective`: mirrored vertically when that side is Black, so the
    board always "looks like" it's being viewed from the bottom, same as the MLP's mirroring."""
    return square if perspective == chess.WHITE else chess.square_mirror(square)


def castling_indices(board: chess.Board, perspective: bool) -> list[int]:
    indices = []
    for relative_color, color in ((0, perspective), (1, not perspective)):
        if board.has_kingside_castling_rights(color):
            indices.append(PIECE_FEATURE_DIM + relative_color * 2 + 0)
        if board.has_queenside_castling_rights(color):
            indices.append(PIECE_FEATURE_DIM + relative_color * 2 + 1)
    return indices


def halfkp_indices(board: chess.Board, perspective: bool) -> list[int]:
    king_square = _orient(board.king(perspective), perspective)
    indices = []
    for square, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue
        piece_square = _orient(square, perspective)
        relative_color = 0 if piece.color == perspective else 1
        combined_type = relative_color * 5 + PIECE_INDEX_NO_KING[piece.piece_type]
        indices.append(king_square * 640 + piece_square * 10 + combined_type)
    indices.extend(castling_indices(board, perspective))
    return indices


def _padded(indices: list[int]) -> np.ndarray:
    arr = np.full(MAX_ACTIVE, PAD_INDEX, dtype=np.int32)
    arr[: len(indices)] = indices
    return arr


# ---------------------------------------------------------------------------
# The model. A shared "feature transformer" (the accumulator) turns each
# side's sparse HalfKP indices into a 256-wide vector by summing the active
# rows of one embedding table — literally what nn.EmbeddingBag(mode="sum")
# computes, and the same table is reused for both perspectives. The mover's
# accumulator always goes first, so the small head after it never needs to
# know which color is actually moving.
# ---------------------------------------------------------------------------

ACCUMULATOR_DIM = 256


class NNUE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_transformer = nn.EmbeddingBag(
            FEATURE_DIM + 1, ACCUMULATOR_DIM, mode="sum", padding_idx=PAD_INDEX
        )
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(ACCUMULATOR_DIM * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, stm_indices: torch.Tensor, nstm_indices: torch.Tensor) -> torch.Tensor:
        stm_acc = self.feature_transformer(stm_indices)
        nstm_acc = self.feature_transformer(nstm_indices)
        return self.head(torch.cat([stm_acc, nstm_acc], dim=1))


# ---------------------------------------------------------------------------
# fetch: stream the Lichess eval dump into a training set on disk.
# ---------------------------------------------------------------------------


def _pad_fen(fen: str) -> str:
    """The evals dump omits halfmove/fullmove counters; python-chess needs all 6 fields."""
    fields = fen.split(" ")
    while len(fields) < 6:
        fields.append("0" if len(fields) == 4 else "1")
    return " ".join(fields[:6])


def _best_eval(evals: list[dict]) -> dict | None:
    if not evals:
        return None
    return max(evals, key=lambda e: e.get("depth", 0))


def _target_cp(entry: dict, stm_is_white: bool) -> float | None:
    pvs = entry.get("pvs")
    if not pvs:
        return None
    pv = pvs[0]
    if "mate" in pv:
        cp = TARGET_CLIP_CP if pv["mate"] > 0 else -TARGET_CLIP_CP
    elif "cp" in pv:
        cp = float(pv["cp"])
    else:
        return None
    cp = max(-TARGET_CLIP_CP, min(TARGET_CLIP_CP, cp))
    return cp if stm_is_white else -cp


def _is_quiet(board: chess.Board, pv: dict) -> bool:
    """A static evaluator can't see the tactics a search would find, so positions where the side to
    move is already in check, or the engine's own top line is a capture/promotion/check, carry a
    target the net has no way to predict from the position alone — drop them rather than train on
    that noise. Deliberately not stricter than this (e.g. requiring no capture anywhere on the
    board): that would strip out most complex middlegames and bias the dataset toward bland,
    quiet-by-construction positions."""
    if board.is_check():
        return False
    line = pv.get("line")
    if not line:
        return True
    # pv["line"] is UCI_Chess960 notation (castling as king-takes-own-rook, e.g. "e1h1") even
    # though these are standard-chess positions — without chess960 mode a castling move fails to
    # parse as any legal move at all, silently falling through to "assume quiet" below. That
    # happens to be the right answer for castling specifically (it's never a capture), but by
    # accident of the exception handler rather than because the move was understood.
    board.chess960 = True
    try:
        move = board.parse_uci(line.split()[0])
    except ValueError:
        return True
    return not (board.is_capture(move) or move.promotion or board.gives_check(move))


def _raw_cp(pv: dict) -> float | None:
    """The PV's raw score, unclipped — None for a mate line (unbounded magnitude, the single most
    decisive category there is, so it fails any cp-magnitude filter by construction)."""
    if "mate" in pv:
        return None
    cp = pv.get("cp")
    return float(cp) if cp is not None else None


def _depth_disagreement(evals: list[dict], deepest: dict, min_depth: int) -> float | None:
    """|cp gap| between `deepest` (the entry _best_eval already picked) and the shallowest *other*
    pass that still clears min_depth, reusing whatever multi-pass data the dump already has for
    this position rather than running a fresh search. None when there's no second qualifying pass
    to compare against — a position isn't unstable for the sole reason nobody happened to
    re-analyze it a second time, so it's kept rather than penalized for that. A shallower pass
    claiming mate counts as maximal disagreement: a mate mirage that more search resolves away is
    exactly the instability this filter exists to catch."""
    qualifying = [
        e for e in evals if e is not deepest and e.get("depth", 0) >= min_depth and e.get("pvs")
    ]
    if not qualifying:
        return None
    shallow_pv = min(qualifying, key=lambda e: e.get("depth", 0))["pvs"][0]
    if "mate" in shallow_pv:
        return float("inf")
    shallow_cp = shallow_pv.get("cp")
    deep_cp = deepest["pvs"][0].get("cp")
    if shallow_cp is None or deep_cp is None:
        return None
    return abs(shallow_cp - deep_cp)


# Both sides' non-pawn material at a full board: 2 knights + 2 bishops + 2 rooks + 1 queen, per
# side. Same piece-value scale as agent.py's search-time PIECE_VALUE, just for a different purpose
# here (a phase signal, not move ordering).
PHASE_PIECE_VALUE = {
    chess.KNIGHT: 320.0,
    chess.BISHOP: 330.0,
    chess.ROOK: 500.0,
    chess.QUEEN: 900.0,
}
STARTING_NON_PAWN_MATERIAL = 2 * (2 * 320.0 + 2 * 330.0 + 2 * 500.0 + 900.0)


def phase_tag(board: chess.Board) -> float:
    """Non-pawn material still on the board (both sides), normalized against the starting total:
    1.0 at the game's start, trending toward 0.0 as pieces (not pawns) come off. A cheap proxy for
    how "middlegame" vs "endgame" a position is — captured now for later phase-stratified sampling
    or a phase-interpolated output head, neither built yet, just the signal."""
    material = sum(
        len(board.pieces(piece_type, color)) * value
        for piece_type, value in PHASE_PIECE_VALUE.items()
        for color in (chess.WHITE, chess.BLACK)
    )
    return material / STARTING_NON_PAWN_MATERIAL


def fetch(
    num_positions: int,
    sample_every: int,
    data_dir: Path = DATA_DIR,
    score_cp_limit: float = SCORE_CP_LIMIT,
) -> None:
    """Streams the eval dump, keeping one position every `sample_every` lines seen, until
    `num_positions` positions have been kept. sample_every=1 takes a prefix of the file, which is
    fast but whatever ordering Lichess wrote the dump in; a larger value spreads the sample over
    more of the file at the cost of reading (and discarding) more of the stream.

    `score_cp_limit` gates the sharp/near-decisive-position drop (module default SCORE_CP_LIMIT);
    pass float("inf") to keep every position regardless of raw score magnitude."""
    data_dir.mkdir(exist_ok=True)
    stm_path = data_dir / "stm_indices.i32"
    nstm_path = data_dir / "nstm_indices.i32"
    targets_path = data_dir / "targets.f32"
    phase_path = data_dir / "phase.f32"
    count_path = data_dir / "count.npy"

    curl = subprocess.Popen(["curl", "-s", EVAL_URL], stdout=subprocess.PIPE)
    zstd = subprocess.Popen(["zstd", "-dc"], stdin=curl.stdout, stdout=subprocess.PIPE)
    assert curl.stdout is not None
    curl.stdout.close()

    kept = 0
    seen = 0
    # Cheapest filters first (pure dict access, no Board needed) so an expensive board construction
    # + move parse only happens for candidates that already cleared the free checks.
    dropped = {
        "depth": 0,
        "score_magnitude": 0,
        "depth_disagreement": 0,
        "not_quiet": 0,
        "no_target": 0,
        "too_many_pieces": 0,
    }
    try:
        with (
            open(stm_path, "wb") as stm_out,
            open(nstm_path, "wb") as nstm_out,
            open(targets_path, "wb") as target_out,
            open(phase_path, "wb") as phase_out,
        ):
            assert zstd.stdout is not None
            for raw_line in zstd.stdout:
                seen += 1
                if seen % sample_every != 0:
                    continue
                try:
                    row = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue

                evals = row.get("evals", [])
                entry = _best_eval(evals)
                if entry is None or entry.get("depth", 0) < MIN_DEPTH:
                    dropped["depth"] += 1
                    continue

                pvs = entry.get("pvs")
                if not pvs:
                    dropped["depth"] += 1
                    continue
                pv = pvs[0]

                raw_cp = _raw_cp(pv)
                if raw_cp is None or abs(raw_cp) > score_cp_limit:
                    dropped["score_magnitude"] += 1
                    continue

                gap = _depth_disagreement(evals, entry, MIN_DEPTH)
                if gap is not None and gap > DEPTH_GAP_CP_LIMIT:
                    dropped["depth_disagreement"] += 1
                    continue

                board = chess.Board(_pad_fen(row["fen"]))
                if not _is_quiet(board, pv):
                    dropped["not_quiet"] += 1
                    continue

                target = _target_cp(entry, board.turn == chess.WHITE)
                if target is None:
                    dropped["no_target"] += 1
                    continue

                # The eval dump includes board-editor setups, not just game positions, so piece
                # counts can exceed what's reachable in a legal game (e.g. six queens). Skip those.
                stm = halfkp_indices(board, board.turn)
                nstm = halfkp_indices(board, not board.turn)
                if len(stm) > MAX_ACTIVE or len(nstm) > MAX_ACTIVE:
                    dropped["too_many_pieces"] += 1
                    continue

                stm_out.write(_padded(stm).tobytes())
                nstm_out.write(_padded(nstm).tobytes())
                target_out.write(np.float32(target).tobytes())
                phase_out.write(np.float32(phase_tag(board)).tobytes())

                kept += 1
                if kept % 50000 == 0:
                    print(f"kept {kept} / seen {seen}", file=sys.stderr)
                if kept >= num_positions:
                    break
    finally:
        zstd.kill()
        curl.kill()

    np.save(count_path, np.array([kept], dtype=np.int64))
    print(f"done: kept {kept} positions from {seen} lines seen", file=sys.stderr)
    print(f"dropped by filter: {dropped}", file=sys.stderr)


# ---------------------------------------------------------------------------
# fit: train NNUE on the dataset fetch() produced.
# ---------------------------------------------------------------------------


def _progress_bar(current: int, total: int, prefix: str) -> None:
    width = 30
    filled = int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    end = "\n" if current == total else ""
    print(f"\r{prefix} [{bar}] {current}/{total}", end=end, file=sys.stderr, flush=True)


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _wdl_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross-entropy between predicted and target win probability (sigmoid(cp / WDL_SCALE_CP)),
    the standard NNUE training loss — a 900-vs-1000cp miss (both already "winning") counts for far
    less than a 0-vs-100cp miss (drawish vs winning), and BCE's gradient stays strong even when a
    prediction is confidently wrong, unlike MSE's which vanishes as the sigmoid saturates."""
    return nn.functional.binary_cross_entropy_with_logits(
        pred / WDL_SCALE_CP, torch.sigmoid(target / WDL_SCALE_CP)
    )


def fit(
    epochs: int,
    batch_size: int,
    lr: float,
    val_fraction: float,
    device: str,
    output: str | None,
    data_dir: Path = DATA_DIR,
) -> None:
    count = int(np.load(data_dir / "count.npy")[0])
    stm_indices = np.memmap(
        data_dir / "stm_indices.i32", dtype=np.int32, mode="r", shape=(count, MAX_ACTIVE)
    )
    nstm_indices = np.memmap(
        data_dir / "nstm_indices.i32", dtype=np.int32, mode="r", shape=(count, MAX_ACTIVE)
    )
    targets = np.memmap(data_dir / "targets.f32", dtype=np.float32, mode="r", shape=(count,))
    # phase.f32 (data_dir / "phase.f32") is captured by fetch() but not consumed here yet -- for
    # later phase-stratified sampling or a phase-interpolated head, neither built yet.

    rng = np.random.default_rng(0)
    order = rng.permutation(count)
    split = int(count * (1 - val_fraction))
    # Sorted back into file order: a fully random train_idx would make every epoch's shuffle
    # scatter reads across the whole (multi-GB, larger than free RAM on modest machines) memmap,
    # thrashing the page cache. `batches()` below re-randomizes at block granularity instead, so
    # sorting here costs nothing — it's what makes each block a contiguous, cache-friendly region.
    train_idx, val_idx = np.sort(order[:split]), np.sort(order[split:])

    resolved_device = _resolve_device(device)
    print(f"training on {resolved_device}", file=sys.stderr)

    torch.manual_seed(0)
    model = NNUE().to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    train_batches = -(-len(train_idx) // batch_size)  # ceil division

    # idx is sorted into file order, so a block is a contiguous (~520MB at block_rows=2M) memmap
    # region — small enough to stay resident in page cache even on a memory-tight machine. Only
    # the block order (and the row order within each block) gets reshuffled per epoch; two rows
    # from opposite ends of the file are never read back-to-back the way a fully global shuffle
    # would, which is what was thrashing the cache.
    block_rows = 2_000_000

    def batches(idx: np.ndarray, shuffle: bool):
        if shuffle:
            block_starts = list(range(0, len(idx), block_rows))
            rng.shuffle(block_starts)
            idx = np.concatenate(
                [rng.permutation(idx[start : start + block_rows]) for start in block_starts]
            )
        for start in range(0, len(idx), batch_size):
            batch = np.sort(idx[start : start + batch_size])  # ascending access is memmap-friendly
            stm = torch.from_numpy(stm_indices[batch].astype(np.int64)).to(resolved_device)
            nstm = torch.from_numpy(nstm_indices[batch].astype(np.int64)).to(resolved_device)
            y = torch.from_numpy(targets[batch].astype(np.float32)).unsqueeze(1).to(resolved_device)
            yield stm, nstm, y

    for epoch in range(epochs):
        model.train()
        train_loss, train_cp_err, n = 0.0, 0.0, 0
        for step, (stm, nstm, y) in enumerate(batches(train_idx, shuffle=True), start=1):
            optimizer.zero_grad()
            pred = model(stm, nstm)
            loss = _wdl_loss(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)
            train_cp_err += (pred - y).abs().sum().item()
            n += len(y)
            if step % 20 == 0 or step == train_batches:
                prefix = (
                    f"epoch {epoch + 1}/{epochs} "
                    f"loss={train_loss / n:.4f} cp_mae={train_cp_err / n:.1f}"
                )
                _progress_bar(step, train_batches, prefix)
        scheduler.step()

        model.eval()
        val_loss, val_cp_err, vn = 0.0, 0.0, 0
        with torch.no_grad():
            for stm, nstm, y in batches(val_idx, shuffle=False):
                pred = model(stm, nstm)
                val_loss += _wdl_loss(pred, y).item() * len(y)
                val_cp_err += (pred - y).abs().sum().item()
                vn += len(y)

        print(
            f"epoch {epoch + 1}/{epochs} "
            f"train_loss={train_loss / n:.4f} train_cp_mae={train_cp_err / n:.1f} "
            f"val_loss={val_loss / vn:.4f} val_cp_mae={val_cp_err / vn:.1f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}",
            file=sys.stderr,
        )

    if output is None:
        output = f"weights_{epochs}ep_{batch_size}bs_{time.strftime('%Y%m%d-%H%M%S')}.pt"
    output_path = WEIGHTS_PATH.parent / output
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, output_path)
    print(f"saved {output_path}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser("fetch")
    fetch_parser.add_argument("--positions", type=int, default=1_000_000)
    fetch_parser.add_argument("--sample-every", type=int, default=1)
    fetch_parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="where to write the dataset (default mlp/data_v2/) -- never overwrites mlp/data/",
    )
    fetch_parser.add_argument(
        "--score-cp-limit",
        type=float,
        default=SCORE_CP_LIMIT,
        help="drop positions with |raw cp| above this, before clipping (default 800); "
        "pass inf to keep sharp/near-decisive positions too",
    )

    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument("--epochs", type=int, default=20)
    fit_parser.add_argument("--batch-size", type=int, default=8192)
    fit_parser.add_argument("--lr", type=float, default=1e-3)
    fit_parser.add_argument("--val-fraction", type=float, default=0.02)
    fit_parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    fit_parser.add_argument(
        "--output",
        default=None,
        help="weights filename under mlp/ (default: auto-named from epochs/batch-size/timestamp "
        "so runs don't clobber each other; pass 'weights.pt' explicitly to promote a run)",
    )
    fit_parser.add_argument(
        "--data-dir", type=Path, default=DATA_DIR, help="dataset to train on (default mlp/data_v2/)"
    )

    arguments = parser.parse_args()
    if arguments.command == "fetch":
        fetch(
            arguments.positions,
            arguments.sample_every,
            arguments.data_dir,
            arguments.score_cp_limit,
        )
    else:
        fit(
            arguments.epochs,
            arguments.batch_size,
            arguments.lr,
            arguments.val_fraction,
            arguments.device,
            arguments.output,
            arguments.data_dir,
        )


if __name__ == "__main__":
    main()
