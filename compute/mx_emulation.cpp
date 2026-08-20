#include "mx_emulation.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>

namespace {
std::string to_lower_copy(const std::string &fmt_in) {
  std::string fmt = fmt_in;
  std::transform(fmt.begin(), fmt.end(), fmt.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return fmt;
}

bool underflow_mode_is_gu() {
  static int mode_init = 0;
  static bool use_gu = false;
  if (!mode_init) {
    if (const char *env = getenv("MX_UNDERFLOW_MODE")) {
      const std::string m = to_lower_copy(env);
      use_gu = (m == "gu" || m == "gradual");
    }
    mode_init = 1;
  }
  return use_gu;
}
}  // namespace

long double format_unit_roundoff(const std::string &fmt_in) {
  const std::string fmt = to_lower_copy(fmt_in);
  if (fmt == "fp32") {
    return static_cast<long double>(std::numeric_limits<float>::epsilon());
  }
  if (fmt == "fp16" || fmt == "mx_fp16" || fmt == "mx_f16") {
    // IEEE binary16 machine epsilon = 2^-10 (10 mantissa bits).
    return std::ldexp(1.0L, -10);
  }
  if (fmt == "mx_e4m3" || fmt == "mx_fp8_e4m3" || fmt == "e4m3" ||
      fmt == "fp8_e4m3" || fmt == "fp8e4m3") {
    return 1.0L / 16.0L;
  }
  if (fmt == "mx_e5m2" || fmt == "mx_fp8_e5m2" || fmt == "e5m2" ||
      fmt == "fp8_e5m2" || fmt == "fp8e5m2" || fmt == "e3m2") {
    return 1.0L / 8.0L;
  }
  if (fmt == "e2m3") {
    return 1.0L / 16.0L;
  }
  if (fmt == "e2m1") {
    return 1.0L / 4.0L;
  }
  return 1.0L / 16.0L;
}

bool format_uses_shared_scale(const std::string &fmt_in) {
  const std::string fmt = to_lower_copy(fmt_in);
  return (fmt == "mx_fp16" || fmt == "mx_f16" || fmt == "mx_e4m3" ||
          fmt == "mx_fp8_e4m3" || fmt == "mx_e5m2" || fmt == "mx_fp8_e5m2" ||
          fmt == "e3m2" || fmt == "e2m3" || fmt == "e2m1");
}

long double format_fmin(const std::string &fmt_in) {
  const std::string fmt = to_lower_copy(fmt_in);
  if (fmt == "fp32") {
    return std::ldexp(1.0L, -126);
  }
  if (fmt == "fp16" || fmt == "mx_fp16" || fmt == "mx_f16") {
    return std::ldexp(1.0L, -14);
  }
  if (fmt == "mx_e4m3" || fmt == "mx_fp8_e4m3" || fmt == "e4m3" ||
      fmt == "fp8_e4m3" || fmt == "fp8e4m3") {
    return std::ldexp(1.0L, -6);
  }
  if (fmt == "mx_e5m2" || fmt == "mx_fp8_e5m2" || fmt == "e5m2" ||
      fmt == "fp8_e5m2" || fmt == "fp8e5m2") {
    return std::ldexp(1.0L, -14);
  }
  if (fmt == "e3m2") {
    return std::ldexp(1.0L, -2);
  }
  if (fmt == "e2m3") {
    return std::ldexp(1.0L, 0);
  }
  if (fmt == "e2m1") {
    return std::ldexp(1.0L, 0);
  }
  return 0.0L;
}

long double format_fmax(const std::string &fmt_in) {
  const std::string fmt = to_lower_copy(fmt_in);
  if (fmt == "fp32") {
    return (std::numeric_limits<float>::max)();
  }
  if (fmt == "fp16" || fmt == "mx_fp16" || fmt == "mx_f16") {
    return 65504.0L;
  }
  if (fmt == "mx_e4m3" || fmt == "mx_fp8_e4m3" || fmt == "e4m3" ||
      fmt == "fp8_e4m3" || fmt == "fp8e4m3") {
    return 448.0L;
  }
  if (fmt == "mx_e5m2" || fmt == "mx_fp8_e5m2" || fmt == "e5m2" ||
      fmt == "fp8_e5m2" || fmt == "fp8e5m2") {
    return 57344.0L;
  }
  if (fmt == "e3m2") {
    return 28.0L;
  }
  if (fmt == "e2m3") {
    return 7.5L;
  }
  if (fmt == "e2m1") {
    return 6.0L;
  }
  return 0.0L;
}

long double format_gmin(const std::string &fmt_in) {
  const long double fmin = format_fmin(fmt_in);
  if (!(fmin > 0.0L)) return 0.0L;

  const long double u = format_unit_roundoff(fmt_in);
  return underflow_mode_is_gu() ? (u * fmin) : (0.5L * fmin);
}

int8_t computeScaleFromMax(float max_val) {
  if (max_val <= 0.0f) return 0;
  int scale = static_cast<int>(std::floor(std::log2(max_val)));
  if (scale > 127) scale = 127;
  if (scale < -128) scale = -128;
  return static_cast<int8_t>(scale);
}

int8_t computeScale(const float *data, size_t n) {
  float max_val = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    max_val = std::fmax(max_val, std::fabs(data[i]));
  }
  if (max_val <= 0.0f) return 0;
  int scale = static_cast<int>(std::floor(std::log2(max_val)));
  if (scale > 127) scale = 127;
  if (scale < -128) scale = -128;
  return static_cast<int8_t>(scale);
}

int computeScaleBits(const float *data, size_t n, int scale_bits) {
  if (scale_bits <= 0) {
    return static_cast<int>(computeScale(data, n));
  }
  float max_val = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    max_val = std::fmax(max_val, std::fabs(data[i]));
  }
  if (max_val <= 0.0f) return 0;
  int scale = static_cast<int>(std::floor(std::log2(max_val)));
  const int max_scale = (1 << (scale_bits - 1)) - 1;
  const int min_scale = -(1 << (scale_bits - 1));
  if (scale > max_scale) scale = max_scale;
  if (scale < min_scale) scale = min_scale;
  return scale;
}

int computeScaleBitsFromMax(float max_val, int scale_bits) {
  if (max_val <= 0.0f) return 0;
  int scale = static_cast<int>(std::floor(std::log2(max_val)));
  if (scale_bits <= 0) {
    if (scale > 127) scale = 127;
    if (scale < -128) scale = -128;
    return scale;
  }
  const int max_scale = (1 << (scale_bits - 1)) - 1;
  const int min_scale = -(1 << (scale_bits - 1));
  if (scale > max_scale) scale = max_scale;
  if (scale < min_scale) scale = min_scale;
  return scale;
}

float pow2(int exp) { return std::ldexp(1.0f, exp); }

float quantizeFp(float x, int ebits, int mbits, float max_norm) {
  if (x == 0.0f) return 0.0f;
  const float sign = std::signbit(x) ? -1.0f : 1.0f;
  float ax = std::fabs(x);
  if (ax > max_norm) ax = max_norm;
  int exp = static_cast<int>(std::floor(std::log2(ax)));
  const int bias = (1 << (ebits - 1)) - 1;
  const int exp_enc = exp + bias;
  if (exp_enc <= 0) {
    // OCP-spec gradual underflow (subnormals) -- the only underflow policy
    // this build supports; encode as mant_int * 2^(1 - bias - mbits).
    // Smallest representable subnormal:
    //   E4M3 (bias=7, mbits=3) -> 2^-9 ≈ 1.95e-3
    //   E5M2 (bias=15, mbits=2) -> 2^-16 ≈ 1.53e-5
    // Round-to-nearest-even via std::lrint (with FE_TONEAREST default).
    // mant_int = 0 -> underflow to zero.
    // mant_int = 2^mbits -> "rounded up to smallest normal" — keep the natural
    // value (mi * subn_step = 2^(1-bias) = smallest normal). No cap needed; ax
    // is bounded above by 2^(1-bias) when exp_enc <= 0, so mi <= 2^mbits.
    const float subn_step = std::ldexp(1.0f, 1 - bias - mbits);
    int mi = static_cast<int>(std::lrint(ax / subn_step));
    if (mi <= 0) return 0.0f;
    return sign * static_cast<float>(mi) * subn_step;
  }
  const int exp_max = (1 << ebits) - 1;
  if (exp_enc >= exp_max) {
    return sign * max_norm;
  }
  const float base = pow2(exp);
  float mant = ax / base - 1.0f;
  const float step = 1.0f / static_cast<float>(1 << mbits);
  mant = std::round(mant / step) * step;
  float out = (1.0f + mant) * base;
  if (out > max_norm) out = max_norm;
  return sign * out;
}
