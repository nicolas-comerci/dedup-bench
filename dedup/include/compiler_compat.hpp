//
// Created by NicolasC on 18/7/2026.
//

#ifndef DEDUP_BENCH_COMPILER_COMPAT_H
#define DEDUP_BENCH_COMPILER_COMPAT_H

#include <cstdint>

#if defined(_MSC_VER)
    #include <intrin.h>
#endif
#if defined(__BMI__)
    #include <immintrin.h>
#endif

#if defined(_MSC_VER)
    #define RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
    #define RESTRICT __restrict__
#else
    #define RESTRICT
#endif

// Inlines a hot-loop helper regardless of the compiler's size heuristics,
// which otherwise depend on how much other code ends up in the caller.
#if defined(_MSC_VER)
    #define ALWAYS_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
    #define ALWAYS_INLINE inline __attribute__((always_inline))
#else
    #define ALWAYS_INLINE inline
#endif

// Same for a lambda; goes after its parameter list.
#if defined(__GNUC__) || defined(__clang__)
    #define ALWAYS_INLINE_LAMBDA __attribute__((always_inline))
#else
    #define ALWAYS_INLINE_LAMBDA
#endif

// Number of trailing zero bits of value, which must be nonzero. tzcnt needs
// BMI (-mbmi); without it, bsf-based alternatives give the same result for
// nonzero values.
inline uint32_t trailing_zeros_u32(uint32_t value) {
#if defined(__BMI__)
    return _tzcnt_u32(value);
#elif defined(_MSC_VER)
    unsigned long index;
    _BitScanForward(&index, value);
    return static_cast<uint32_t>(index);
#elif defined(__GNUC__) || defined(__clang__)
    return static_cast<uint32_t>(__builtin_ctz(value));
#else
    uint32_t count = 0;
    while ((value & 1u) == 0) {
        value >>= 1;
        ++count;
    }
    return count;
#endif
}

#endif //DEDUP_BENCH_COMPILER_COMPAT_H
