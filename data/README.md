# data/

Input matrices go here. Nothing is checked into git except this file —
`.bin` files are gitignored (they're gigabytes each) and are never fetched
automatically. Drop your own here, or point `--bin` / `BIN_LIST` /
`DATA_DIR` elsewhere.

## Format

Raw binary, no header: `N*N` `float64` values, row-major, representing a
symmetric positive-definite `N x N` matrix. File size must be exactly
`N*N*8` bytes. `N` is inferred from the file size unless `--n` is passed
explicitly to `example_dpotrf_gpu`.

## Naming convention

The run scripts (`run_ladder.sh`, `run_ladder_ftz_tilemap_nb4096.sh`)
expect `my_cov_weak_<size>.bin`, e.g.:

```
data/my_cov_weak_32k.bin    # N = 32768
data/my_cov_weak_40k.bin    # N = 40960
data/my_cov_weak_65k.bin    # N = 65536
```

A missing file is skipped with a `[WARN]`/`[SKIP]`, not fatal — sweeps run
over whichever of the configured bins are actually present.

## How these matrices were generated

The matrices are synthetic Matérn covariance matrices `C`, generated with
[ExaGeoStatCPP](https://github.com/ecrc/ExaGeoStatCPP) and dumped to disk
as raw `float64`:

- kernel `UnivariateMaternStationary`, exact (non-approximated) computation
- `theta = variance:range:smoothness = 1:0.02627:0.5` — weak correlation,
  used for every matrix in this benchmark, hence the `my_cov_weak_*` naming
- problem size `N` as in the filename, fixed seed (1234) for reproducibility

Any generator producing a dense SPD matrix in the format above works just
as well; nothing here depends on ExaGeoStat specifically. Note that dumping
the covariance matrix to a file is not a stock ExaGeoStatCPP feature — it
needs a small patch to its `DataGeneration` example to write `C` out after
assembly.

## Usage

```bash
# direct run against one file
./examples/example_dpotrf_gpu --bin data/my_cov_weak_32k.bin --nb 2048 --cores 32

# sweep script, using whatever's in data/ by default
./run_ladder.sh

# point at a dataset that lives elsewhere instead
BIN_LIST="/path/to/my_cov_weak_32k.bin" ./run_ladder.sh
```
