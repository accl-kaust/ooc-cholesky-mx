/**
 * MX/FP8 numeric emulation layer: per-format properties, shared-exponent
 * scale computation, and the element-level quantization grid used by both
 * the plain-FP8 baseline path and the MX block-scaled ladder modes.
 *
 * Pure functions only (no Eigen/CUDA/tile-state dependency) -- everything
 * here operates on plain floats/strings and env-var knobs.
 */
#ifndef MX_EMULATION_H
#define MX_EMULATION_H

#include <cstddef>
#include <cstdint>
#include <string>

// MX block size: 32 elements (the canonical OCP MX shared-scale group),
// applied as a 1D run of columns within each row ("vec1d"). This build
// supports exactly this one MX mode.
inline constexpr int kMxVecSize = 32;

// -- Per-format numeric properties (used by the bound-eligibility check) --

// Unit roundoff (machine epsilon) for a format name (fp32/fp16/
// mx_e4m3/mx_e5m2/fp8_e4m3/fp8_e5m2/e3m2/e2m3/e2m1/mx_fp16).
long double format_unit_roundoff(const std::string &fmt_in);

// Whether a format uses an MX shared (block) scale, vs. a plain per-element
// grid with no block scaling.
bool format_uses_shared_scale(const std::string &fmt_in);

// Smallest normal magnitude representable by the format.
long double format_fmin(const std::string &fmt_in);

// Largest finite magnitude representable by the format.
long double format_fmax(const std::string &fmt_in);

// Smallest positive value that survives rounding (gu: first subnormal
// step; fz: half the smallest normal). Depends on MX_UNDERFLOW_MODE.
long double format_gmin(const std::string &fmt_in);

// -- Shared-exponent (block) scale computation --

// floor(log2(max_val)), clamped to an int8 exponent range. 0 if max_val<=0.
int8_t computeScaleFromMax(float max_val);

// Same, but scans a raw array for its max-abs value first.
int8_t computeScale(const float *data, size_t n);

// Variants parameterized by an explicit scale-bit width instead of the
// fixed int8 range (used for higher-precision block scales).
int computeScaleBitsFromMax(float max_val, int scale_bits);
int computeScaleBits(const float *data, size_t n, int scale_bits);

// 2^exp as a float.
float pow2(int exp);

// -- Element-level quantization --

// Round x onto the (ebits, mbits) floating-point grid (E4M3/E5M2/E3M2/
// E2M3/E2M1, or any other ebits/mbits combination), saturating to
// +/-max_norm and using OCP-spec gradual underflow (subnormals) below the
// format's normal range -- the only underflow policy this build supports.
float quantizeFp(float x, int ebits, int mbits, float max_norm);

#endif  // MX_EMULATION_H
