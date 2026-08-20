#!/usr/bin/env python3
"""Convert a run-log of [TILE_TARGET] lines into a per-tile allocation file.

Spec (per user):
    # N=<N> nb=<nb> nt=<nt>
    # eps=<eps>
    # mode=ladder_mx_staircase MX_FTZ=1
    r c FORMAT          (one line per lower-triangular tile, row-major)

FORMAT vocabulary: FP64, FP32, FP16, MXFP16, MXFP8, MXFP4, FP8_plain.
"""
import argparse
import re
import sys
from pathlib import Path


VOCAB = {
    "fp64":     "FP64",
    "fp32":     "FP32",
    "fp16":     "FP16",
    "mx_fp16":  "MXFP16",
    "mx_e4m3":  "MXFP8",
    "mx_e5m2":  "MXFP8",
    "e2m1":     "MXFP4",
    "fp8_e4m3": "FP8_plain",
    "fp8_e5m2": "FP8_plain",
    # mx_fp32 isn't expected from the Full ladder; map to FP32 if it appears.
    "mx_fp32":  "FP32",
    "bf16":     "FP16",
}

TILE_RE = re.compile(r"\[TILE_TARGET\]\s*\((\d+),\s*(\d+)\)\s+(\S+)")


def extract(log_path: Path, n: int, nb: int, eps: str, out_path: Path) -> dict:
    assert n % nb == 0, f"n={n} not divisible by nb={nb}"
    nt = n // nb
    expected = nt * (nt + 1) // 2

    tiles = {}
    for line in log_path.read_text(errors="replace").splitlines():
        m = TILE_RE.search(line)
        if not m:
            continue
        r, c = int(m.group(1)), int(m.group(2))
        tok = m.group(3).strip().lower()
        if tok not in VOCAB:
            raise ValueError(f"unknown TILE_TARGET token {tok!r} in {log_path}")
        if r < c:
            # Upper-tri shouldn't appear; bail if it does.
            raise ValueError(f"unexpected upper-tri tile ({r},{c}) in {log_path}")
        # last-wins per (r, c) is fine — the per-tile loop only emits once.
        tiles[(r, c)] = VOCAB[tok]

    if len(tiles) != expected:
        raise ValueError(
            f"{log_path}: got {len(tiles)} TILE_TARGET lines, expected "
            f"nt*(nt+1)/2 = {expected} (nt={nt})"
        )

    # Diagonal must be FP64 (POTRF requires it).
    bad_diag = [k for k in range(nt) if tiles.get((k, k)) != "FP64"]
    if bad_diag:
        raise ValueError(
            f"{log_path}: non-FP64 diagonal tiles at rows {bad_diag}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {}
    with out_path.open("w") as f:
        f.write(f"# N={n} nb={nb} nt={nt}\n")
        f.write(f"# eps={eps}\n")
        f.write("# mode=ladder_mx_staircase MX_FTZ=1\n")
        for r in range(nt):
            for c in range(r + 1):
                fmt = tiles[(r, c)]
                counts[fmt] = counts.get(fmt, 0) + 1
                f.write(f"{r} {c} {fmt}\n")

    return {"n": n, "nb": nb, "nt": nt, "eps": eps, "tiles": expected, **counts}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, type=Path)
    ap.add_argument("--n", required=True, type=int)
    ap.add_argument("--nb", required=True, type=int)
    ap.add_argument("--eps", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    info = extract(args.log, args.n, args.nb, args.eps, args.out)
    parts = [f"{args.out}"]
    for fmt in ("FP64", "FP32", "MXFP16", "MXFP8", "MXFP4", "FP16", "FP8_plain"):
        if info.get(fmt):
            parts.append(f"{fmt}={info[fmt]}")
    print("  ".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
