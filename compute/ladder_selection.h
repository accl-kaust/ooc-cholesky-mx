/**
 * Ladder tile-format selection: the three bound-selector modes (Ladder
 * IEEE, Ladder MX, Ladder MX+MXFP16 in the paper's naming -- IeeeOnly,
 * MxStaircase, Full internally). Baseline is NOT here; it uses a separate
 * epsilon-ratio heuristic that stays inline in dpotrf_mixed_precision.cpp,
 * since it's a genuinely different selection mechanism (no per-format
 * bound evaluation at all).
 *
 * Header-only for the same reason as mx_apply.h: evaluate_format_tile_bound
 * and run() read the source tile through a live Eigen expression template,
 * so they're templated on its concrete type.
 */
#ifndef LADDER_SELECTION_H
#define LADDER_SELECTION_H

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

#include "mixed_precision.h"
#include "mx_emulation.h"
#include "mx_apply.h"

namespace mx_ladder {

// MX_SELECTION_CRITERIA=bound|explicit_bound|nf enables this whole module;
// otherwise the caller should fall through to the baseline heuristic.
inline bool selection_criteria_is_bound() {
  static int init = 0;
  static bool enabled = false;
  if (!init) {
    if (const char *env = getenv("MX_SELECTION_CRITERIA")) {
      std::string v = env;
      std::transform(v.begin(), v.end(), v.begin(),
                     [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
      enabled = (v == "bound" || v == "explicit_bound" || v == "nf");
    }
    init = 1;
  }
  return enabled;
}

inline bool bound_debug_enabled() {
  static const int v = []() {
    const char *env = getenv("MX_BOUND_DEBUG");
    if (!env) return 0;
    return (env[0] == '1') ? 1 : 0;
  }();
  return v != 0;
}

// Ladder IEEE / Ladder MX / Ladder MX+MXFP16 in the paper's naming.
enum class BoundLadderMode { Full, IeeeOnly, MxStaircase };

inline BoundLadderMode current_mode() {
  static const BoundLadderMode m = []() {
    const char *env = getenv("MX_BOUND_LADDER");
    if (!env) {
      // default: enable full low->high ladder in bound mode
      return BoundLadderMode::Full;
    }
    std::string v = env;
    std::transform(v.begin(), v.end(), v.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    // Values match LADDER= in run_ladder.sh one-for-one -- one naming
    // scheme for the whole codebase, no separate aliases.
    if (v == "ladder_mx_mxfp16") {
      return BoundLadderMode::Full;
    }
    if (v == "ladder_ieee") {
      return BoundLadderMode::IeeeOnly;
    }
    if (v == "ladder_mx") {
      return BoundLadderMode::MxStaircase;
    }
    // Unrecognized value: same as no env at all.
    return BoundLadderMode::Full;
  }();
  return m;
}

struct BoundEvalResult {
  bool fits = false;
  bool overflow = false;
  bool uses_scale = false;
  long double u = 0.0L;
  long double fmin = 0.0L;
  long double fmax = 0.0L;
  long double gmin = 0.0L;
  long double tol = 0.0L;
  long double nrmE = 0.0L;
  long double nrmF = 0.0L;
  long double min_scale = 1.0L;
  long double max_scale = 1.0L;
  long double avg_scale = 1.0L;
  long long scale_groups = 0;
  long long underflows = 0;
};

// Would `fmt_name` (fp32/fp16/mx_e4m3/e2m1/...) fit this tile within the
// per-tile error tolerance? Mirrors the actual quant path from mx_apply.h:
// no-scale formats scan the whole tile at scale 1; shared-scale formats
// mirror the vec1d quantizer's kMxVecSize-column groups, one scale each.
template <typename Derived>
BoundEvalResult evaluate_format_tile_bound(
    const std::string &fmt_name, const MixedPrecisionTile &tile,
    const Eigen::MatrixBase<Derived> &mappedBlock, long double sourceEpsilon,
    long double normA, long double normTile, long nt) {
  BoundEvalResult out;
  const long double u = format_unit_roundoff(fmt_name);
  const long double fmin = format_fmin(fmt_name);
  const long double fmax = format_fmax(fmt_name);
  const long double gmin = format_gmin(fmt_name);
  out.u = u;
  out.fmin = fmin;
  out.fmax = fmax;
  out.gmin = gmin;
  if (!(u > 0.0L)) return out;

  const long double tol_tile = sourceEpsilon * normA / static_cast<long double>(nt);
  const long double nrmE = u * normTile;
  long double nrmF2 = 0.0L;
  bool overflow = false;
  out.tol = tol_tile;
  out.nrmE = nrmE;

  const bool uses_scale = format_uses_shared_scale(fmt_name);
  const size_t tile_m = static_cast<size_t>(tile.m);
  const size_t tile_n = static_cast<size_t>(tile.n);
  out.uses_scale = uses_scale;

  long double sum_scale = 0.0L;
  long double min_scale = (std::numeric_limits<long double>::max)();
  long double max_scale = 0.0L;
  long long scale_groups = 0;
  long long underflows = 0;

  auto note_scale = [&](long double pre_scale) {
    sum_scale += pre_scale;
    if (pre_scale < min_scale) min_scale = pre_scale;
    if (pre_scale > max_scale) max_scale = pre_scale;
    ++scale_groups;
  };

  // pre_scale is the multiplier used before quantization (x_qin = pre_scale * x).
  // This matches the implementation path where x_qin = x * inv_scale.
  auto consume_val = [&](long double ax, long double pre_scale) {
    const long double scaled = pre_scale * ax;
    if (fmax > 0.0L && scaled > fmax) {
      overflow = true;
      return;
    }
    if (fmin > 0.0L && gmin > 0.0L && scaled < fmin) {
      long double eta = (pre_scale > 0.0L) ? (gmin / pre_scale) : gmin;
      if (eta > ax) eta = ax;
      nrmF2 += eta * eta;
      ++underflows;
    }
  };

  if (!uses_scale) {
    for (size_t r = 0; r < tile_m && !overflow; ++r) {
      for (size_t c = 0; c < tile_n; ++c) {
        const long double ax = std::fabs(static_cast<long double>(
            mappedBlock(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c))));
        consume_val(ax, 1.0L);
        if (overflow) break;
      }
    }
  } else {
    // Mirror the actual vec1d quant path: each row split into groups
    // of kMxVecSize columns, one shared scale per group.
    const size_t vec_sz = static_cast<size_t>(kMxVecSize);
    for (size_t r = 0; r < tile_m && !overflow; ++r) {
      for (size_t c0 = 0; c0 < tile_n && !overflow; c0 += vec_sz) {
        const size_t c_max = (vec_sz < (tile_n - c0)) ? vec_sz : (tile_n - c0);
        float max_val = 0.0f;
        for (size_t c = 0; c < c_max; ++c) {
          const float v = static_cast<float>(std::fabs(
              mappedBlock(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c0 + c))));
          max_val = std::fmax(max_val, v);
        }
        const int grp_scale = static_cast<int>(computeScaleFromMax(max_val));
        const long double pre_scale = std::ldexp(1.0L, -grp_scale);
        note_scale(pre_scale);
        for (size_t c = 0; c < c_max; ++c) {
          const long double ax = std::fabs(static_cast<long double>(
              mappedBlock(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c0 + c))));
          consume_val(ax, pre_scale);
          if (overflow) break;
        }
      }
    }
  }

  out.overflow = overflow;
  const long double nrmF = std::sqrt(nrmF2);
  out.nrmF = nrmF;
  out.underflows = underflows;
  out.scale_groups = scale_groups;
  if (scale_groups > 0) {
    out.min_scale = min_scale;
    out.max_scale = max_scale;
    out.avg_scale = sum_scale / static_cast<long double>(scale_groups);
  }
  out.fits = (!overflow) && ((nrmE + nrmF) <= tol_tile);
  return out;
}

namespace detail {

template <typename Derived>
void write_fp64_fallback(MixedPrecisionTile &tile, int row, int col,
                          const Eigen::MatrixBase<Derived> &mappedBlock) {
  mx_apply::log_tile_target(row, col, "fp64");
  tile.dtype = CUDA_R_64F;
  cudaMallocHost(&tile.data, sizeof(uint64_t) * tile.m * tile.n);
  Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>{
      static_cast<double *>(tile.data), static_cast<Eigen::Index>(tile.m),
      static_cast<Eigen::Index>(tile.n)} = mappedBlock.template cast<double>();
}

}  // namespace detail

// Run the bound-selector ladder for one tile: try each rung of the
// current mode's ladder in ascending cost order, apply the first whose
// bound-eligibility check fits (via mx_apply.h), or fall back to FP64 if
// none do. Handles its own BOUND_SELECT/BOUND_EVAL/BOUND_DECISION debug
// output when MX_BOUND_DEBUG=1.
template <typename Derived>
void run(MixedPrecisionTile &tile, int row, int col,
         const Eigen::MatrixBase<Derived> &mappedBlock, long double sourceEpsilon,
         long double normA, long double normTile, long nt,
         const std::string &fp32_bucket, const std::string &fp16_bucket) {
  const bool bound_debug = bound_debug_enabled();
  const BoundLadderMode mode = current_mode();

  const std::string fp32_choice = mx_apply::canonical_format_name(fp32_bucket, "fp32");
  const std::string fp16_choice = mx_apply::canonical_format_name(fp16_bucket, "fp16");

  if (bound_debug) {
    std::cout << "[BOUND_SELECT] tile(" << row << "," << col << ")"
              << " fp32_bucket=" << fp32_choice << " fp16_bucket=" << fp16_choice
              << " ladder="
              << (mode == BoundLadderMode::Full
                      ? "ladder_mx_mxfp16"
                      : (mode == BoundLadderMode::IeeeOnly ? "ladder_ieee" : "ladder_mx"))
              << " mode=vec1d" << " subtile=" << kMxVecSize << std::endl;
  }

  auto eval_one = [&](const std::string &fmt) {
    return evaluate_format_tile_bound(fmt, tile, mappedBlock, sourceEpsilon, normA, normTile, nt);
  };
  auto print_eval = [&](const std::string &name, const BoundEvalResult &ev) {
    if (!bound_debug) return;
    std::cout << "[BOUND_EVAL] tile(" << row << "," << col << ")"
              << " fmt=" << name << " fits=" << (ev.fits ? 1 : 0)
              << " overflow=" << (ev.overflow ? 1 : 0) << " uses_scale=" << (ev.uses_scale ? 1 : 0)
              << " u=" << ev.u << " fmin=" << ev.fmin << " fmax=" << ev.fmax
              << " gmin=" << ev.gmin << " tol=" << ev.tol << " nrmE=" << ev.nrmE
              << " nrmF=" << ev.nrmF << " nrmE+nrmF=" << (ev.nrmE + ev.nrmF)
              << " underflows=" << ev.underflows << " scale_groups=" << ev.scale_groups
              << " s_min=" << ev.min_scale << " s_avg=" << ev.avg_scale
              << " s_max=" << ev.max_scale << std::endl;
  };

  std::string decision_fmt;
  std::string decision_reason;
  int decision_step = -1;

  const char *const *ladder_raw = nullptr;
  int ladder_len = 0;
  const char *full_ladder[] = {"e2m1", "mx_e4m3", "mx_fp16", "fp32"};
  const char *staircase_ladder[] = {"e2m1", "mx_e4m3", "fp16", "fp32"};
  const char *ieee_ladder[] = {"fp8_e4m3", "fp16", "fp32"};
  const char *accept_reason = nullptr;
  const char *reject_reason = nullptr;

  if (mode == BoundLadderMode::Full) {
    // FP4/FP6/FP8 -> FP16/MXFP16 -> FP32/MXFP32 -> FP64 fallback.
    ladder_raw = full_ladder;
    ladder_len = 4;
    accept_reason = "ladder_mx_mxfp16_accept";
    reject_reason = "ladder_mx_mxfp16_reject_all";
  } else if (mode == BoundLadderMode::MxStaircase) {
    // Same as Full but the FP16 rung is plain IEEE FP16 (no shared scale)
    // instead of MXFP16. MXFP4 -> MXFP8 (E4M3) -> FP16 -> FP32 -> FP64.
    ladder_raw = staircase_ladder;
    ladder_len = 4;
    accept_reason = "ladder_mx_accept";
    reject_reason = "ladder_mx_reject_all";
  } else {
    // IeeeOnly: no shared-scale MX formats. FP8_E4M3 -> FP16 -> FP32 -> FP64.
    ladder_raw = ieee_ladder;
    ladder_len = 3;
    accept_reason = "ladder_ieee_accept";
    reject_reason = "ladder_ieee_reject_all";
  }

  bool placed = false;
  for (int idx = 0; idx < ladder_len; ++idx) {
    const char *cand_raw = ladder_raw[idx];
    std::string cand = mx_apply::canonical_format_name(cand_raw, cand_raw);
    const auto ev = eval_one(cand);
    print_eval(cand, ev);
    if (ev.fits) {
      mx_apply::apply_bucket(cand, cand.c_str(), tile, row, col, mappedBlock);
      decision_fmt = cand;
      decision_reason = accept_reason;
      decision_step = idx;
      placed = true;
      break;
    }
  }
  if (!placed) {
    detail::write_fp64_fallback(tile, row, col, mappedBlock);
    decision_fmt = "fp64";
    decision_reason = reject_reason;
  }

  if (bound_debug) {
    std::cout << "[BOUND_DECISION] tile(" << row << "," << col << ")"
              << " selected=" << decision_fmt << " reason=" << decision_reason
              << " step=" << decision_step << std::endl;
  }
}

}  // namespace mx_ladder

#endif  // LADDER_SELECTION_H
