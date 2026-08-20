# ooc-cholesky-mx

Out-of-core, GPU-based Cholesky factorization with MX (OCP microscaling)
low-precision tile emulation. This is the MX-emulation fork of the
PLASMA-derived `ooc-cholesky` codebase: the core PLASMA tile-factorization
engine plus the MX/FP8 quantization + bound-selection ("ladder") pipeline
and the two run scripts that drive it.

## Requirements

- CMake >= 3.17 (needs `FindCUDAToolkit`)
- A C/C++ compiler with C++20 support (built and tested with GCC 11)
- CUDA toolkit + `nvcc` (tested with CUDA 12.5) and an NVIDIA GPU
- BLAS + LAPACK + LAPACKE (e.g. OpenBLAS), or Intel MKL
- Python 3 (only needed for `tools/extract_tilemap.py`)

## Build

Eigen is a git submodule (`third_party/eigen`), fetch it first:

```bash
git submodule update --init
```

Then configure and build. `PLASMA_WITH_MKL` is ON by default and falls
back to BLAS/LAPACK/LAPACKE automatically if MKL isn't found; pass
`-DBLA_VENDOR=...` explicitly if CMake picks up the wrong BLAS, and set
`CMAKE_CUDA_ARCHITECTURES` to match your GPU (80 = A100, 90 = H100, ...):

```bash
mkdir build && cd build
cmake .. \
    -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DBLA_VENDOR=OpenBLAS
make -j
```

This builds two executables: `example_dpotrf` (plain double-precision
Cholesky, sanity check) and `example_dpotrf_gpu` (the MX-emulation
pipeline) under `build/`.

## Input data

`example_dpotrf_gpu` factors a symmetric positive-definite matrix read
from a raw binary file (`--bin`) — row-major `N x N` `float64`, no
header, `N` inferred from the file size. Nothing ships with this repo;
see [`data/README.md`](data/README.md) for the exact format and naming
convention, and drop your own matrix file(s) under `data/`.

## Running it directly

```bash
./build/example_dpotrf_gpu --bin data/my_cov_weak_32k.bin --nb 2048 --cores 32
```

Relevant env vars read at runtime (all optional, see
`compute/ladder_selection.h` / `compute/mx_apply.h` for the full set):

| Var | Meaning |
|---|---|
| `MX_SELECTION_CRITERIA=bound` | use the bound-eligibility ladder selector instead of the epsilon-ratio heuristic |
| `MX_BOUND_LADDER` | `ladder_ieee` \| `ladder_mx` \| `ladder_mx_mxfp16` — which ladder rungs are eligible (matches `LADDER=` one-for-one) |
| `MX_UNDERFLOW_MODE` | `gu` (gradual underflow, default) \| `fz` (flush-to-zero) — block-level bound-eligibility threshold |
| `MX_SOURCE_EPSILON` | target relative error for tier/rung selection |
| `MX_BOUND_DEBUG=1` | emit `[TILE_TARGET] (row, col) fmt` lines per tile (consumed by `extract_tilemap.py`) |

## Running the ladder sweep

`run_ladder.sh` drives `example_dpotrf_gpu` across bins x epsilons x
underflow modes for one ladder at a time:

```bash
./run_ladder.sh                              # Ladder MX+MXFP16 (default), gu, data/*.bin
LADDER=baseline ./run_ladder.sh              # Baseline heuristic, plain FP8_E4M3 low tier
LADDER=ladder_ieee ./run_ladder.sh           # Ladder IEEE (no MX formats)
LADDER=ladder_mx ./run_ladder.sh             # Ladder MX (MXFP4 + MXFP8_E4M3 only)
BIN_LIST="/path/to/my.bin" ./run_ladder.sh   # point at data living outside this repo
```

| `LADDER` | Format chain |
|---|---|
| `baseline` | fp32 -> fp16 -> fp8_e4m3 -> fp64 (epsilon-ratio heuristic, not the bound selector; no MX shared scale) |
| `ladder_ieee` | fp8_e4m3 -> fp16 -> fp32 -> fp64 (bound selector, no MX formats) |
| `ladder_mx` | e2m1(MXFP4) -> mx_e4m3(MXFP8_E4M3) -> fp16 -> fp32 -> fp64 |
| `ladder_mx_mxfp16` | e2m1 -> mx_e4m3 -> mx_fp16 -> fp32 -> fp64 (all three low/mid rungs MX-scaled) |

Other knobs: `UNDERFLOW_LIST` (space-separated `gu`/`fz`), `EPS_LIST`
(space-separated epsilons), `NB` (force one tile size), `CORES`,
`OUT_DIR` (default `ladder_sweep/`, gitignored).

Output lands in `$OUT_DIR/results.csv` (columns: `sweep, bin, n, nb,
mx_mode, underflow, source_epsilon, rel_factor_error, abs_factor_error,
relative_residual, tile_breakdown`), with one raw run log per config
under `$OUT_DIR/run_logs/` and a running `$OUT_DIR/sweep_master.log`.

## Tile-allocation extraction

`run_ladder_ftz_tilemap_nb4096.sh` recovers just the per-tile format
decisions (not a full factorization) by killing the run once every
lower-triangular tile has logged a `[TILE_TARGET]` line, then converts
the log to a `.tilemap` file via `tools/extract_tilemap.py`. Same
`data/`-relative bin defaults as `run_ladder.sh` (override via
`DATA_DIR`).

## Changes vs. the baseline framework

Unchanged from the baseline: stock PLASMA's out-of-core tile Cholesky
(`example_dpotrf`), and the original HPCASIA'24 epsilon-ratio
mixed-precision heuristic, still reachable as `LADDER=baseline`.

Added on top:

- **Bound-eligibility ladder selector** — picks the cheapest per-tile
  format whose error bound still holds, instead of a fixed
  epsilon-ratio check (`compute/ladder_selection.h`, `mx_apply.h`,
  `mx_emulation.{h,cpp}`). Adds the `ladder_ieee`, `ladder_mx`, and
  `ladder_mx_mxfp16` modes.
- **MX shared-scale tiers** — `e2m1` / `mx_e4m3` / `mx_fp16`, 32-element
  row-vector scale groups, plus 6 MX metadata fields on
  `MixedPrecisionTile`.
- **`deduceMemcpyKind`** — the `pdpotrf_gpu_*.cpp` paths inspect the
  actual pointer instead of hardcoding the copy direction, so device and
  managed memory copy correctly.
- **Run scripts** — `run_ladder.sh`,
  `run_ladder_ftz_tilemap_nb4096.sh`, `tools/extract_tilemap.py`.
- **Build/portability** — LAPACKE detection, `data/`-relative input
  paths, and `CMAKE_CUDA_ARCHITECTURES = "80-real;90-real;90-virtual"`
  (native SASS for A100 and Hopper, plus PTX so newer architectures
  still run via driver JIT).
