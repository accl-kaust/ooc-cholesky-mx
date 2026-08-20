#!/usr/bin/env bash
# Unified MX ladder sweep driver — replaces the family of near-duplicate
# run_ladder_*.sh / run_bound_ladder_*.sh / run_requant_ladder_*.sh scripts.
#
# This build supports exactly one MX shared-scale mode: mx (32-element
# row-vector groups, the canonical OCP MX block size) — there is no
# tile/block granularity to select any more, so this script has no such
# axis. What still varies from run to run is: ladder mode, underflow mode,
# which matrix bin(s), tile size, and epsilon sweep. All are env-var knobs
# with sensible defaults, following this repo's existing convention.
#
# Rounding/underflow: element-level FP8/MX grid rounding is *always* gradual
# underflow (subnormals) -- that used to be a knob (MX_FP8_SUBNORMAL), but
# it's now hardcoded in quantizeFp() since FTZ was never actually wanted.
# The remaining underflow axis is MX_UNDERFLOW_MODE (block-level bound
# eligibility threshold, via UNDERFLOW_LIST, default gu). Note this only
# affects the three bound-selector modes -- baseline's heuristic path never
# reads MX_UNDERFLOW_MODE at all, so UNDERFLOW_LIST is a no-op there.
# Override UNDERFLOW_LIST=fz for an FTZ-threshold comparison run.
#
# Usage examples:
#   ./run_ladder.sh                                                 # Ladder MX+MXFP16 (default), gu, 32k/40k/65k bins
#   LADDER=ladder_ieee ./run_ladder.sh                                # Ladder IEEE, gu
#   LADDER=ladder_mx ./run_ladder.sh                                 # Ladder MX (MXFP4 + MXFP8_E4M3 only)
#   LADDER=baseline ./run_ladder.sh                                  # Baseline: no bound selection, plain FP8_E4M3 low tier
#   UNDERFLOW_LIST=fz BIN_LIST="/path/32k.bin" ./run_ladder.sh        # FTZ comparison run
#   BIN_LIST="/path/40k.bin" NB=4096 ./run_ladder.sh                  # 40k @ nb=4096 fill-in
#
# Modes (LADDER) — names match the paper:
#   baseline           Baseline: fp32 -> fp16 -> fp8_e4m3 -> fp64 (epsilon-ratio
#                       heuristic, NOT the bound selector; low tier is plain FP8,
#                       no MX shared scale at all)
#   ladder_ieee         Ladder IEEE: fp8_e4m3 -> fp16 -> fp32 -> fp64 (bound
#                       selector, no MX formats)
#   ladder_mx           Ladder MX: e2m1(MXFP4) -> mx_e4m3(MXFP8_E4M3) -> fp16 ->
#                       fp32 -> fp64 (bound selector; only the two lowest rungs
#                       are MX-scaled — fp16 rung above them is plain IEEE)
#   ladder_mx_mxfp16    Ladder MX+MXFP16: e2m1 -> mx_e4m3 -> mx_fp16 -> fp32 ->
#                       fp64 (bound selector; all three low/mid rungs MX-scaled)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXE="$SCRIPT_DIR/examples/example_dpotrf_gpu"

LADDER=${LADDER:-ladder_mx_mxfp16}            # baseline | ladder_ieee | ladder_mx | ladder_mx_mxfp16
UNDERFLOW_LIST=${UNDERFLOW_LIST:-gu}          # space-separated: fz gu -- gu (gradual underflow /
                                               # subnormals) is now the default everywhere subnormal
                                               # support is available; override to fz for comparison
EPS_LIST=${EPS_LIST:-"1e-5 1e-6 1e-7 1e-8"}
CORES=${CORES:-32}
BIN_LIST=${BIN_LIST:-"\
$SCRIPT_DIR/data/my_cov_weak_32k.bin \
$SCRIPT_DIR/data/my_cov_weak_40k.bin \
$SCRIPT_DIR/data/my_cov_weak_65k.bin"}
# ^ defaults to this repo's own data/ dir (see data/README.md for the
# expected format) -- missing files are skipped with a [WARN], not fatal.
# Override BIN_LIST to point anywhere else, e.g. a shared dataset location.
NB_OVERRIDE=${NB:-}                           # set to force one nb for every bin

nb_for_bin() {
  case "$(basename "$1")" in
    *65k*) echo 4096 ;;
    *40k*) echo 2048 ;;
    *32k*) echo 2048 ;;
    *)     echo "${NB_DEFAULT:-2048}" ;;
  esac
}

n_for_bin() {
  case "$(basename "$1")" in
    *32k*) echo 32768 ;;
    *40k*) echo 40960 ;;
    *65k*) echo 65536 ;;
    *)     echo 0 ;;
  esac
}

case "$LADDER" in
  baseline)
    # No MX_SELECTION_CRITERIA=bound: falls through to the pre-ladder
    # epsilon-ratio heuristic (fp32/fp16 tiers decided by the heuristic,
    # low tier fixed to plain FP8_E4M3 -- apply_plain_fp_quant, no MX block
    # scale -- Baseline isn't an MX mode, so this isn't configurable).
    LADDER_ENV=(MX_BUCKET_FP32=fp32 MX_BUCKET_FP16=fp16)
    LADDER_NAME="Baseline"
    LADDER_DESC="fp32 -> fp16 -> fp8_e4m3 -> fp64 (epsilon-ratio heuristic, no MX)"
    SWEEP_TAG="baseline"
    MX_MODE_LABEL="plain"
    ;;
  ladder_ieee)
    LADDER_ENV=(MX_SELECTION_CRITERIA=bound MX_BOUND_LADDER=ladder_ieee)
    LADDER_NAME="Ladder IEEE"
    LADDER_DESC="fp8_e4m3 -> fp16 -> fp32 -> fp64"
    SWEEP_TAG="ladder_ieee"
    MX_MODE_LABEL="mx"
    ;;
  ladder_mx)
    LADDER_ENV=(MX_SELECTION_CRITERIA=bound MX_BOUND_LADDER=ladder_mx)
    LADDER_NAME="Ladder MX"
    LADDER_DESC="e2m1(MXFP4) -> mx_e4m3(MXFP8_E4M3) -> fp16 -> fp32 -> fp64"
    SWEEP_TAG="ladder_mx"
    MX_MODE_LABEL="mx"
    ;;
  ladder_mx_mxfp16)
    LADDER_ENV=(MX_SELECTION_CRITERIA=bound MX_BOUND_LADDER=ladder_mx_mxfp16)
    LADDER_NAME="Ladder MX+MXFP16"
    LADDER_DESC="e2m1 -> mx_e4m3 -> mx_fp16 -> fp32 -> fp64"
    SWEEP_TAG="ladder_mx_mxfp16"
    MX_MODE_LABEL="mx"
    ;;
  *)
    echo "[FATAL] unknown LADDER='$LADDER' (want baseline|ladder_ieee|ladder_mx|ladder_mx_mxfp16)" >&2
    exit 1
    ;;
esac

OUT_DIR=${OUT_DIR:-"$SCRIPT_DIR/ladder_sweep"}
RUN_LOG_DIR="$OUT_DIR/run_logs"
CSV="$OUT_DIR/results.csv"
MASTER_LOG="$OUT_DIR/sweep_master.log"
mkdir -p "$RUN_LOG_DIR"

if [[ ! -f "$CSV" ]]; then
  echo "sweep,bin,n,nb,mx_mode,underflow,source_epsilon,rel_factor_error,abs_factor_error,relative_residual,tile_breakdown" > "$CSV"
fi

{
  echo "[START] $(date -Is)"
  echo "[CFG] ladder=$LADDER_NAME ($LADDER_DESC)"
  echo "[CFG] mx_mode=$MX_MODE_LABEL (subtile=32 when MX-scaled; this build has no other MX mode)"
  echo "[CFG] underflow_list=$UNDERFLOW_LIST"
  echo "[CFG] bins=$BIN_LIST"
  echo "[CFG] eps=$EPS_LIST"
  echo "[CFG] nb_override=${NB_OVERRIDE:-<auto via nb_for_bin>}"
  echo "[CFG] csv=$CSV"
} | tee -a "$MASTER_LOG"

run_one() {
  local bin="$1" underflow="$2" eps="$3"
  local base nb n sweep stamp run_log
  base=$(basename "$bin" .bin)
  nb=${NB_OVERRIDE:-$(nb_for_bin "$bin")}
  n=$(n_for_bin "$bin")
  sweep="${SWEEP_TAG}_${underflow}"
  stamp=$(date +"%Y%m%d_%H%M%S_%N")
  run_log="$RUN_LOG_DIR/${sweep}_${base}_nb${nb}_eps${eps}_${stamp}.log"

  echo "[RUN] ladder=$LADDER_NAME bin=$base underflow=$underflow eps=$eps nb=$nb log=$(basename "$run_log")" | tee -a "$MASTER_LOG"

  (
    cd "$SCRIPT_DIR/examples"
    exec env \
      MX_BOUND_DEBUG=1 \
      "${LADDER_ENV[@]}" \
      MX_UNDERFLOW_MODE="$underflow" \
      MX_SOURCE_EPSILON="$eps" \
      "$EXE" --nb "$nb" --cores "$CORES" --bin "$bin"
  ) >"$run_log" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[FAIL] rc=$rc see $run_log" | tee -a "$MASTER_LOG"
    return
  fi

  local rel abs res tb
  rel=$(grep -m1 "^relative_error:" "$run_log" | awk '{print $NF}')
  abs=$(grep -m1 "^error:" "$run_log" | awk '{print $NF}')
  res=$(grep -m1 "^relative_residual:" "$run_log" | awk '{print $NF}')
  tb=$(grep -oE "^\[TILE_TARGET\] \([0-9]+, [0-9]+\) \S+" "$run_log" \
       | awk '{print $NF}' | sort | uniq -c | sort -rn \
       | awk '{printf "%s=%s;",$2,$1}')

  echo "[OK] bin=$base underflow=$underflow eps=$eps rel=$rel abs=$abs res=$res tiles=$tb" | tee -a "$MASTER_LOG"
  echo "$sweep,$bin,$n,$nb,$MX_MODE_LABEL,$underflow,$eps,${rel:-NA},${abs:-NA},${res:-NA},\"$tb\"" >> "$CSV"
}

for bin in $BIN_LIST; do
  if [[ ! -f "$bin" ]]; then
    echo "[WARN] missing bin, skipping: $bin" | tee -a "$MASTER_LOG"
    continue
  fi
  for underflow in $UNDERFLOW_LIST; do
    for eps in $EPS_LIST; do
      run_one "$bin" "$underflow" "$eps"
    done
  done
done

echo "[END] $(date -Is)" | tee -a "$MASTER_LOG"
echo "CSV: $CSV"
echo "Master log: $MASTER_LOG"
