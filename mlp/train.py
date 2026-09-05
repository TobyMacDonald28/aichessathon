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

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import chess
import numpy as np

# torch takes ~1.7s to import -- dead weight fetch() doesn't need, and multiprocessing.Pool below
# re-imports this whole module in every worker process, so a module-level `import torch` would pay
# that cost N times just to spawn the pool. Deferred into fit()/_resolve_device()/_wdl_loss(), the
# only places that actually touch it; `from __future__ import annotations` (above) keeps the
# torch.Tensor type hints in their signatures valid without torch imported at module scope.
if TYPE_CHECKING:
    import torch

DATA_DIR = Path(__file__).parent / "data_v2"
WEIGHTS_PATH = Path(__file__).parent / "weights.pt"

EVAL_URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"
# Finite (non-mate) scores clip here -- high enough that a real massive-material-advantage position
# survives intact for the WDL sigmoid below to compress, rather than getting flattened to the same
# value as a merely-comfortable one. Mate scores don't use this: see MATE_TARGET_CP.
TARGET_CLIP_CP = 1500.0
MATE_TARGET_CP = 3000.0  # static target for a mate-in-N line -- more decisive than any finite score
MIN_DEPTH = 16
WDL_SCALE_CP = 400.0  # centipawns that map to a 1-sigma shift in sigmoid win-probability space
DEPTH_GAP_CP_LIMIT = 50.0  # max disagreement between a shallow(>=MIN_DEPTH) and the deepest pass
# No opening-ply truncation: the dump's FEN strings never carry a real fullmove number (_pad_fen
# below hardcodes one on every row that's missing it, which is effectively all of them), so there's
# no ply-position signal in this dataset to filter an "opening truncation" on.

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

# NNUE is defined inside fit() (below), not here, alongside its own `import torch` — see the note
# on the deferred torch import up top for why.


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
        cp = MATE_TARGET_CP if pv["mate"] > 0 else -MATE_TARGET_CP
    elif "cp" in pv:
        cp = max(-TARGET_CLIP_CP, min(TARGET_CLIP_CP, float(pv["cp"])))
    else:
        return None
    return cp if stm_is_white else -cp


def _is_quiet(board: chess.Board, pv: dict) -> bool:
    """A static evaluator can't see a trade in progress the way a search would, so a position whose
    engine's own top line is a capture or promotion carries a target the net has no way to predict
    from the position alone — drop it rather than train on that noise. A position where the side to
    move is already in check is kept rather than dropped: that's the only source of "king under
    attack" signal in the whole dataset (see king_safety_score in agent.py, which has nothing else
    to learn from). Deliberately not stricter than the capture/promotion check (e.g. requiring no
    capture anywhere on the board, or dropping a move that itself gives check): that would strip out
    most complex middlegames and most of the check-in-check-out sequences this filter exists to
    keep, biasing the dataset toward bland, quiet-by-construction positions."""
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
    return not (board.is_capture(move) or move.promotion)


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


_Row = tuple[np.ndarray, np.ndarray, np.float32, np.float32]  # stm, nstm, target, phase
_ProcessResult = tuple[str, None] | tuple[None, _Row]


def _process_line(raw_line: bytes) -> _ProcessResult:
    """The CPU-bound part of turning one dump line into a training row, run in a worker process
    (see fetch()'s multiprocessing.Pool below). Measured directly: chess.Board construction and
    the two halfkp_indices calls are ~80% of this pipeline's per-line cost, and curl+zstd alone can
    supply lines about 4x faster than one core can process them -- spreading this across cores is
    most of the win. Returns (drop_reason, None) for a rejected line, or (None, (stm, nstm, target,
    phase)) -- both padded/typed exactly as fetch() used to write them, so the caller just appends
    bytes without knowing anything happened in another process."""
    try:
        row = json.loads(raw_line)
    except json.JSONDecodeError:
        return "json_error", None

    evals = row.get("evals", [])
    entry = _best_eval(evals)
    if entry is None or entry.get("depth", 0) < MIN_DEPTH:
        return "depth", None

    pvs = entry.get("pvs")
    if not pvs:
        return "depth", None
    pv = pvs[0]

    gap = _depth_disagreement(evals, entry, MIN_DEPTH)
    if gap is not None and gap > DEPTH_GAP_CP_LIMIT:
        return "depth_disagreement", None

    board = chess.Board(_pad_fen(row["fen"]))
    if not _is_quiet(board, pv):
        return "not_quiet", None

    target = _target_cp(entry, board.turn == chess.WHITE)
    if target is None:
        return "no_target", None

    # The eval dump includes board-editor setups, not just game positions, so piece counts can
    # exceed what's reachable in a legal game (e.g. six queens). Skip those.
    stm = halfkp_indices(board, board.turn)
    nstm = halfkp_indices(board, not board.turn)
    if len(stm) > MAX_ACTIVE or len(nstm) > MAX_ACTIVE:
        return "too_many_pieces", None

    return None, (_padded(stm), _padded(nstm), np.float32(target), np.float32(phase_tag(board)))


def fetch(
    num_positions: int,
    sample_every: int,
    data_dir: Path = DATA_DIR,
) -> None:
    """Streams the eval dump, keeping one position every `sample_every` lines seen, until
    `num_positions` positions have been kept. sample_every=1 takes a prefix of the file, which is
    fast but whatever ordering Lichess wrote the dump in; a larger value spreads the sample over
    more of the file at the cost of reading (and discarding) more of the stream.

    The per-line filtering/encoding (_process_line) runs in a multiprocessing.Pool -- one process
    alone leaves most of a modern machine idle here, since network + decompression can feed lines
    several times faster than a single core can turn them into training rows."""
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
    zstd_stdout = zstd.stdout
    assert zstd_stdout is not None

    kept = 0
    seen = [0]  # mutable cell: sampled_lines() below runs in this process, not a worker
    dropped = {
        "json_error": 0,
        "depth": 0,
        "depth_disagreement": 0,
        "not_quiet": 0,
        "no_target": 0,
        "too_many_pieces": 0,
    }

    def sampled_lines() -> Iterator[bytes]:
        for raw_line in zstd_stdout:
            seen[0] += 1
            if seen[0] % sample_every == 0:
                yield raw_line

    workers = max(1, (os.cpu_count() or 2) - 1)  # leave a core for curl/zstd/this process
    try:
        with (
            open(stm_path, "wb") as stm_out,
            open(nstm_path, "wb") as nstm_out,
            open(targets_path, "wb") as target_out,
            open(phase_path, "wb") as phase_out,
            multiprocessing.Pool(processes=workers) as pool,
        ):
            for drop_reason, result in pool.imap_unordered(
                _process_line, sampled_lines(), chunksize=64
            ):
                if result is None:
                    assert drop_reason is not None
                    dropped[drop_reason] += 1
                    continue

                stm_arr, nstm_arr, target, phase = result
                stm_out.write(stm_arr.tobytes())
                nstm_out.write(nstm_arr.tobytes())
                target_out.write(target.tobytes())
                phase_out.write(phase.tobytes())

                kept += 1
                if kept % 50000 == 0:
                    print(f"kept {kept} / seen {seen[0]}", file=sys.stderr)
                if kept >= num_positions:
                    break
    finally:
        zstd.kill()
        curl.kill()

    np.save(count_path, np.array([kept], dtype=np.int64))
    print(f"done: kept {kept} positions from {seen[0]} lines seen", file=sys.stderr)
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
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _wdl_loss(
    pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None
) -> torch.Tensor:
    """Cross-entropy between predicted and target win probability (sigmoid(cp / WDL_SCALE_CP)),
    the standard NNUE training loss — a 900-vs-1000cp miss (both already "winning") counts for far
    less than a 0-vs-100cp miss (drawish vs winning), and BCE's gradient stays strong even when a
    prediction is confidently wrong, unlike MSE's which vanishes as the sigmoid saturates.

    `weight` (see _target_density_weights) is for a --reweight fine-tuning pass; leave it None for
    a from-scratch run and always for validation, where the point is reading the model's true,
    unweighted calibration back out."""
    import torch
    from torch import nn

    return nn.functional.binary_cross_entropy_with_logits(
        pred / WDL_SCALE_CP, torch.sigmoid(target / WDL_SCALE_CP), weight=weight
    )


def _target_density_weights(targets: np.ndarray, num_bins: int = 60) -> np.ndarray:
    """Per-example training weight, inverse to how densely populated its |target| bin is.

    Meant for a `fit --reweight` fine-tuning pass on top of an already-trained model, not a
    from-scratch run: real game positions are heavily clustered near dead-equal (data_v3, e.g., was
    ~28% within 25cp of equal and thinned out fast from there, only ~12.6% in the 100-200cp band),
    and a single shared regression function fit against that skew learns to hedge every prediction
    toward zero to minimize loss on the dominant near-equal mass -- measured directly on data_v3, a
    real ~100-150cp edge came out of the trained model as only ~65-100cp. Weighting examples back
    toward a uniform density over |target| counteracts that.

    sqrt of inverse density, not the full inverse: a bin at 1% of the average density gets ~10x the
    weight, not 100x, so the rarest bins (mate scores, huge material swings) don't end up
    dominating gradient steps just for being rare -- this corrects the near-zero pileup, it doesn't
    demand every band contribute equally regardless of how little signal it carries."""
    magnitude = np.abs(targets).astype(np.float32)
    bin_edges = np.linspace(0.0, float(magnitude.max()) + 1.0, num_bins + 1, dtype=np.float32)
    bin_index = np.clip(np.digitize(magnitude, bin_edges) - 1, 0, num_bins - 1)
    counts = np.bincount(bin_index, minlength=num_bins).astype(np.float32)
    density = counts[bin_index] / len(magnitude)
    weight = 1.0 / np.sqrt(density * num_bins)
    return (weight / weight.mean()).astype(np.float32)  # mean 1 -- comparable loss/lr scale


def fit(
    epochs: int,
    batch_size: int,
    lr: float,
    val_fraction: float,
    device: str,
    output: str | None,
    data_dir: Path = DATA_DIR,
    init_weights: Path | None = None,
    reweight: bool = False,
) -> None:
    import torch
    from torch import nn

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
    if init_weights is not None:
        model.load_state_dict(torch.load(init_weights, map_location=resolved_device))
        print(f"fine-tuning from {init_weights}", file=sys.stderr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    train_batches = -(-len(train_idx) // batch_size)  # ceil division

    sample_weight = _target_density_weights(targets) if reweight else None

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
            w = None
            if sample_weight is not None:
                w = torch.from_numpy(sample_weight[batch]).unsqueeze(1).to(resolved_device)
            yield stm, nstm, y, w

    for epoch in range(epochs):
        model.train()
        train_loss, train_cp_err, n = 0.0, 0.0, 0
        for step, (stm, nstm, y, w) in enumerate(batches(train_idx, shuffle=True), start=1):
            optimizer.zero_grad()
            pred = model(stm, nstm)
            loss = _wdl_loss(pred, y, weight=w)
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
            for stm, nstm, y, _ in batches(val_idx, shuffle=False):
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
    fit_parser.add_argument(
        "--init-weights",
        type=Path,
        default=None,
        help="load this checkpoint instead of random-initializing -- for fine-tuning an existing "
        "model rather than training from scratch",
    )
    fit_parser.add_argument(
        "--reweight",
        action="store_true",
        help="weight training examples inversely to |target| density (see "
        "_target_density_weights) -- for a fine-tuning pass correcting calibration in an "
        "underrepresented cp range; not intended for a from-scratch run",
    )

    arguments = parser.parse_args()
    if arguments.command == "fetch":
        fetch(arguments.positions, arguments.sample_every, arguments.data_dir)
    else:
        fit(
            arguments.epochs,
            arguments.batch_size,
            arguments.lr,
            arguments.val_fraction,
            arguments.device,
            arguments.output,
            arguments.data_dir,
            arguments.init_weights,
            arguments.reweight,
        )


if __name__ == "__main__":
    main()
