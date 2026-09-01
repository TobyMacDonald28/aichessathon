"""Trains a HalfKP-style NNUE evaluation net on the Lichess evaluations database.

Two phases, run separately so a slow download never has to repeat while you iterate on the model:

    python train.py fetch --positions 1000000      # streams a sample into mlp/data/
    python train.py fit --epochs 20                # trains on mlp/data/, writes weights.pt

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

DATA_DIR = Path(__file__).parent / "data"
STM_INDICES_PATH = DATA_DIR / "stm_indices.i32"
NSTM_INDICES_PATH = DATA_DIR / "nstm_indices.i32"
TARGETS_PATH = DATA_DIR / "targets.f32"
COUNT_PATH = DATA_DIR / "count.npy"
WEIGHTS_PATH = Path(__file__).parent / "weights.pt"

EVAL_URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"
TARGET_CLIP_CP = 1000.0
MIN_DEPTH = 16
WDL_SCALE_CP = 400.0  # centipawns that map to a 1-sigma shift in sigmoid win-probability space

# ---------------------------------------------------------------------------
# HalfKP feature encoding: the thing that makes this NNUE rather than a plain
# MLP. Each position is seen twice, once from each side's own king. A feature
# is "there is a <piece type, friend-or-foe> on <square>, given my king is on
# <king square>" — so the same knight-on-f3 fact is a different feature
# depending on where the viewer's own king is. Kings themselves aren't
# encoded as pieces (that's the "half" in HalfKP): the king square is the
# anchor the other 40,960 features are relative to, not a feature itself.
# ---------------------------------------------------------------------------

PIECE_INDEX_NO_KING = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
}
FEATURE_DIM = 64 * 64 * 10  # king square x piece square x (5 piece types x friend/foe)
PAD_INDEX = FEATURE_DIM  # one extra embedding row, held at zero, for padding short bags
MAX_ACTIVE = 32  # a legal position has at most 30 non-king pieces


def _orient(square: int, perspective: bool) -> int:
    """Square as seen by `perspective`: mirrored vertically when that side is Black, so the
    board always "looks like" it's being viewed from the bottom, same as the MLP's mirroring."""
    return square if perspective == chess.WHITE else chess.square_mirror(square)


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
    try:
        move = board.parse_uci(line.split()[0])
    except ValueError:
        return True
    return not (board.is_capture(move) or move.promotion or board.gives_check(move))


def fetch(num_positions: int, sample_every: int) -> None:
    """Streams the eval dump, keeping one position every `sample_every` lines seen, until
    `num_positions` positions have been kept. sample_every=1 takes a prefix of the file, which is
    fast but whatever ordering Lichess wrote the dump in; a larger value spreads the sample over
    more of the file at the cost of reading (and discarding) more of the stream."""
    DATA_DIR.mkdir(exist_ok=True)

    curl = subprocess.Popen(["curl", "-s", EVAL_URL], stdout=subprocess.PIPE)
    zstd = subprocess.Popen(["zstd", "-dc"], stdin=curl.stdout, stdout=subprocess.PIPE)
    assert curl.stdout is not None
    curl.stdout.close()

    kept = 0
    seen = 0
    try:
        with (
            open(STM_INDICES_PATH, "wb") as stm_out,
            open(NSTM_INDICES_PATH, "wb") as nstm_out,
            open(TARGETS_PATH, "wb") as target_out,
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

                entry = _best_eval(row.get("evals", []))
                if entry is None or entry.get("depth", 0) < MIN_DEPTH:
                    continue

                board = chess.Board(_pad_fen(row["fen"]))
                if not _is_quiet(board, (entry.get("pvs") or [{}])[0]):
                    continue

                target = _target_cp(entry, board.turn == chess.WHITE)
                if target is None:
                    continue

                # The eval dump includes board-editor setups, not just game positions, so piece
                # counts can exceed what's reachable in a legal game (e.g. six queens). Skip those.
                stm = halfkp_indices(board, board.turn)
                nstm = halfkp_indices(board, not board.turn)
                if len(stm) > MAX_ACTIVE or len(nstm) > MAX_ACTIVE:
                    continue

                stm_out.write(_padded(stm).tobytes())
                nstm_out.write(_padded(nstm).tobytes())
                target_out.write(np.float32(target).tobytes())

                kept += 1
                if kept % 50000 == 0:
                    print(f"kept {kept} / seen {seen}", file=sys.stderr)
                if kept >= num_positions:
                    break
    finally:
        zstd.kill()
        curl.kill()

    np.save(COUNT_PATH, np.array([kept], dtype=np.int64))
    print(f"done: kept {kept} positions from {seen} lines seen", file=sys.stderr)


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
    epochs: int, batch_size: int, lr: float, val_fraction: float, device: str, output: str | None
) -> None:
    count = int(np.load(COUNT_PATH)[0])
    stm_indices = np.memmap(STM_INDICES_PATH, dtype=np.int32, mode="r", shape=(count, MAX_ACTIVE))
    nstm_indices = np.memmap(
        NSTM_INDICES_PATH, dtype=np.int32, mode="r", shape=(count, MAX_ACTIVE)
    )
    targets = np.memmap(TARGETS_PATH, dtype=np.float32, mode="r", shape=(count,))

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

    arguments = parser.parse_args()
    if arguments.command == "fetch":
        fetch(arguments.positions, arguments.sample_every)
    else:
        fit(
            arguments.epochs,
            arguments.batch_size,
            arguments.lr,
            arguments.val_fraction,
            arguments.device,
            arguments.output,
        )


if __name__ == "__main__":
    main()
