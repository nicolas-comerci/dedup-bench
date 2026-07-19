//
// Created by NicolasC on 18/7/2026.
//

#ifndef DEDUP_BENCH_COMPILER_COMPAT_H
#define DEDUP_BENCH_COMPILER_COMPAT_H

#if defined(_MSC_VER)
    #define RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
    #define RESTRICT __restrict__
#else
    #define RESTRICT
#endif

#endif //DEDUP_BENCH_COMPILER_COMPAT_H
