#!/usr/bin/env bash
# Generate per-tile allocation files for the Full ladder + FTZ mode at nb=4096
# across N in {20480, 32768, 40960, 65536, 81920, 98304, 122880}.
#
# Trick: we don't need the factorization — only the per-tile decisions. The
# binary emits "[TILE_TARGET] (r, c) fmt" lines for every lower-triangular
# tile *before* any kernel runs. This script tails the per-run log, kills the
# process once it has counted nt*(nt+1)/2 TILE_TARGET lines, then converts the
# log into a canonical .tilemap file.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXE="$SCRIPT_DIR/examples/example_dpotrf_gpu"
EXTRACTOR="$SCRIPT_DIR/tools/extract_tilemap.py"

OUT_DIR=${OUT_DIR:-"$SCRIPT_DIR/tile_maps_ladder_ftz_nb4096"}
RUN_LOG_DIR="$OUT_DIR/run_logs"
INDEX="$OUT_DIR/index.tsv"
mkdir -p "$RUN_LOG_DIR"

NB=4096
CORES=${CORES:-32}
EPS_LIST=${EPS_LIST:-"1e-5 1e-6 1e-7 1e-8"}

# Bin paths (n, path). Naming labels are loose ("100k" -> N=98304 etc.).
# Defaults to this repo's own data/ dir (see data/README.md) -- missing
# files are [SKIP]ped below, not fatal. Override by editing this array or
# by pointing DATA_DIR at a directory that already has them.
DATA_DIR=${DATA_DIR:-"$SCRIPT_DIR/data"}
BIN_LIST=(
  "20480   $DATA_DIR/my_cov_weak_20k.bin"
  "32768   $DATA_DIR/my_cov_weak_32k.bin"
  "40960   $DATA_DIR/my_cov_weak_40k.bin"
  "65536   $DATA_DIR/my_cov_weak_65k.bin"
  "81920   $DATA_DIR/my_cov_weak_80k.bin"
  "98304   $DATA_DIR/my_cov_weak_100k.bin"
  "122880  $DATA_DIR/my_cov_weak_120k.bin"
)

# Write index header only if the file doesn't exist (resume-safe).
if [[ ! -f "$INDEX" ]]; then
  echo -e "n\tnb\tnt\teps\ttiles\tFP64\tFP32\tMXFP16\tMXFP8\tMXFP4\tFP16\tFP8_plain\ttilemap_file" > "$INDEX"
fi

run_one() {
  local N="$1" BIN="$2" eps="$3"
  local nt=$(( N / NB ))
  local need=$(( nt * (nt + 1) / 2 ))
  local stamp
  stamp=$(date +"%Y%m%d_%H%M%S")
  local run_log="$RUN_LOG_DIR/run_N${N}_eps${eps}_${stamp}.log"
  local tilemap="$OUT_DIR/ladder_ftz_N${N}_nb${NB}_eps${eps}.tilemap"

  if [[ ! -f "$BIN" ]]; then
    echo "[SKIP] missing bin for N=$N: $BIN"
    return
  fi

  # Resume: if the tilemap already exists, skip this config.
  if [[ -f "$tilemap" ]]; then
    echo "[SKIP] N=$N eps=$eps tilemap exists: $(basename "$tilemap")"
    return
  fi

  echo "[RUN] N=$N nb=$NB nt=$nt need=$need eps=$eps  bin=$(basename "$BIN")"

  # Launch the binary with the Full-ladder + FTZ env. Stdout -> run_log.
  # `exec` replaces the subshell with the binary so $pid IS the binary's PID
  # — without this, kill -KILL on the subshell orphans the binary to init.
  (
    cd "$SCRIPT_DIR/examples"
    exec env \
      MX_SELECTION_CRITERIA=bound \
      MX_BOUND_LADDER=ladder_mx_mxfp16 \
      MX_BOUND_DEBUG=1 \
      MX_UNDERFLOW_MODE=fz \
      MX_SOURCE_EPSILON="$eps" \
      "$EXE" --nb "$NB" --cores "$CORES" --bin "$BIN"
  ) >"$run_log" 2>&1 &
  local pid=$!

  # Watcher: poll the log for TILE_TARGET line count; kill when we hit need.
  local count=0 elapsed=0
  while kill -0 "$pid" 2>/dev/null; do
    count=$(grep "^\[TILE_TARGET\]" "$run_log" 2>/dev/null | wc -l)
    if (( count >= need )); then
      # Drain a small grace period to make sure the last line is fully written.
      sleep 0.5
      count=$(grep "^\[TILE_TARGET\]" "$run_log" 2>/dev/null | wc -l)
      echo "[KILL] N=$N eps=$eps got $count/$need TILE_TARGET lines; SIGTERM"
      kill -TERM "$pid" 2>/dev/null || true
      sleep 1
      kill -KILL "$pid" 2>/dev/null || true
      break
    fi
    sleep 2
    elapsed=$(( elapsed + 2 ))
    # 4-hour safety: if we still don't have all lines, give up on this run.
    if (( elapsed >= 14400 )); then
      echo "[TIMEOUT] N=$N eps=$eps after ${elapsed}s ($count/$need); killing"
      kill -KILL "$pid" 2>/dev/null || true
      # Also catch any orphan with the same bin still running.
      pkill -KILL -f "example_dpotrf_gpu --nb $NB --cores $CORES --bin $BIN" 2>/dev/null || true
      break
    fi
  done
  wait "$pid" 2>/dev/null || true

  # Final count check + extract.
  count=$(grep "^\[TILE_TARGET\]" "$run_log" 2>/dev/null | wc -l)
  if [[ "$count" -ne "$need" ]]; then
    echo "[FAIL] N=$N eps=$eps got $count TILE_TARGET lines (need $need); skipping tilemap"
    return
  fi

  # Extract and append index row.
  local extractor_out
  if ! extractor_out=$(python3 "$EXTRACTOR" --log "$run_log" --n "$N" --nb "$NB" --eps "$eps" --out "$tilemap" 2>&1); then
    echo "[FAIL] extractor for N=$N eps=$eps: $extractor_out"
    return
  fi

  echo "[OK]   $extractor_out"

  # Append index row: parse counts from extractor stdout.
  local row="$N\t$NB\t$nt\t$eps\t$need"
  for fmt in FP64 FP32 MXFP16 MXFP8 MXFP4 FP16 FP8_plain; do
    v=$(echo "$extractor_out" | tr '  ' '\n' | grep -E "^${fmt}=" | sed 's/.*=//' | head -1)
    row+="\t${v:-0}"
  done
  row+="\t$(basename "$tilemap")"
  echo -e "$row" >> "$INDEX"
}

for entry in "${BIN_LIST[@]}"; do
  read -r N BIN <<< "$entry"
  for eps in $EPS_LIST; do
    run_one "$N" "$BIN" "$eps"
  done
done

echo
echo "Done. Tilemaps in: $OUT_DIR"
echo "Index:             $INDEX"
