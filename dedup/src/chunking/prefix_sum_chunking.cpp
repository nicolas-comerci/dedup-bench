#include "prefix_sum_chunking.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

#include <immintrin.h>

#include "config_error.hpp"
#include "gear_common.hpp"

namespace {

constexpr uint32_t optimization_batch_size = 4;

}  // namespace

Prefix_Sum_Chunking::Prefix_Sum_Chunking(const Config& config) {
    min_block_size = config.get_prefix_sum_min_block_size();
    avg_block_size = config.get_prefix_sum_avg_block_size();
    max_block_size = config.get_prefix_sum_max_block_size();
    use_64bit_prefix_sum = config.get_use_64bit_prefix_sum();
    use_gear_table_lookup = config.get_use_gear_table_lookup();
    simd_mode = config.get_simd_mode();
    use_subminimum_skipping = config.get_use_subminimum_skipping();
    use_context_repair = config.get_use_context_repair();
    use_supercdc_backup = config.get_use_supercdc_backup();
    normalization_level =
        config.get_optional_fastcdc_normalization_level();
    const bool lookahead_default =
        simd_mode != SIMD_Mode::NONE && !use_gear_table_lookup;
    contribution_lookahead =
        config.get_prefix_sum_contribution_lookahead(lookahead_default);
    if (simd_mode == SIMD_Mode::NONE && contribution_lookahead) {
        throw ConfigError(
            "Prefix-sum SIMD optimizations require a SIMD mode");
    }

    if (min_block_size == 0 || avg_block_size == 0 || max_block_size == 0) {
        throw ConfigError("Prefix-sum block sizes must be greater than zero");
    }
    if (min_block_size > avg_block_size || avg_block_size > max_block_size) {
        throw ConfigError(
            "Prefix-sum block sizes must satisfy min <= avg <= max");
    }
    if ((avg_block_size & (avg_block_size - 1)) != 0) {
        throw ConfigError("Prefix-sum average block size must be a power of two");
    }
    if (avg_block_size > (1ull << 32)) {
        throw ConfigError(
            "Prefix-sum average block size must not exceed 2^32 bytes");
    }
    mask = avg_block_size - 1;
    small_mask = mask;
    large_mask = mask;
    if (normalization_level != 0) {
        uint64_t mask_bits = 0;
        for (uint64_t value = avg_block_size; value > 1; value >>= 1) {
            ++mask_bits;
        }
        if (normalization_level >= mask_bits) {
            throw ConfigError(
                "FastCDC normalization level must be smaller than the "
                "Prefix Sum average-size bit count");
        }
        small_mask =
            (uint64_t{1} << (mask_bits + normalization_level)) - 1;
        large_mask =
            (uint64_t{1} << (mask_bits - normalization_level)) - 1;
    }
    backup_mask = large_mask >> 1;

    if (!use_64bit_prefix_sum) {
        const uint64_t widest_mask = normalization_level != 0
            ? small_mask
            : mask;
        uint32_t maximum_mask_shift = 0;
        const char* simd_name = nullptr;
        if (simd_mode == SIMD_Mode::AVX256) {
            maximum_mask_shift = 7;
            simd_name = "AVX2";
        } else if (simd_mode == SIMD_Mode::AVX512) {
            maximum_mask_shift = 15;
            simd_name = "AVX-512";
        }
        if (simd_name != nullptr
            && widest_mask
                > (static_cast<uint64_t>(
                       std::numeric_limits<uint32_t>::max())
                   >> maximum_mask_shift)) {
            throw ConfigError(
                std::string("32-bit Prefix Sum ") + simd_name
                + " masks are too wide for exact per-lane shifting");
        }
    }
    technique_name = "prefix_sum";
}
template <typename FingerprintType, bool UseGearTable>
static inline FingerprintType update_fingerprint(
    FingerprintType fingerprint, unsigned char byte) {
    if constexpr (UseGearTable) {
        return gear_common::roll(fingerprint, byte);
    }
    return static_cast<FingerprintType>(
        (fingerprint << 1) + static_cast<FingerprintType>(byte));
}

template <typename FingerprintType, bool UseGearTable>
static FingerprintType initial_fingerprint(const unsigned char* data,
                                           uint64_t begin, uint64_t end) {
    FingerprintType fingerprint = 0;
    for (uint64_t idx = begin; idx < end; ++idx) {
        fingerprint = update_fingerprint<FingerprintType, UseGearTable>(
            fingerprint, data[idx]);
    }
    return fingerprint;
}

template <typename FingerprintType, bool UseGearTable, bool SkipSubminimum,
          bool RepairContext>
static FingerprintType initialize_fingerprint(const unsigned char* data,
                                              uint64_t min_block_size) {
    if constexpr (!SkipSubminimum) {
        return initial_fingerprint<FingerprintType, UseGearTable>(
            data, 0, min_block_size);
    }
    if constexpr (RepairContext) {
        constexpr uint64_t repair_width =
            std::numeric_limits<FingerprintType>::digits;
        const uint64_t repair_begin = min_block_size > repair_width
            ? min_block_size - repair_width
            : 0;
        return initial_fingerprint<FingerprintType, UseGearTable>(
            data, repair_begin, min_block_size);
    }
    return 0;
}

#if defined(__AVX2__) || defined(__AVX512F__)
static uint32_t first_set_bit(uint32_t bits) {
    uint32_t bit = 0;
    while ((bits & 1u) == 0) {
        bits >>= 1;
        ++bit;
    }
    return bit;
}
#endif

#if defined(__AVX2__)
template <bool UseGearTable>
static inline __m256i load_contributions_32_avx2(
    const unsigned char* data) {
    // Eight packed bytes widen to all eight 32-bit AVX2 lanes.
    const __m128i incoming_bytes = _mm_loadu_si64(data);
    const __m256i values = _mm256_cvtepu8_epi32(incoming_bytes);
    if constexpr (UseGearTable) {
        return _mm256_i32gather_epi32(
            reinterpret_cast<const int*>(gear_common::table_32), values, 4);
    }
    return values;
}

template <bool UseGearTable>
static inline __m256i load_contributions_64_avx2(
    const unsigned char* data) {
    // Four packed bytes widen to all four 64-bit AVX2 lanes.
    const __m128i incoming_bytes = _mm_loadu_si32(data);
    if constexpr (UseGearTable) {
        const __m128i indices = _mm_cvtepu8_epi32(incoming_bytes);
        return _mm256_i32gather_epi64(
            reinterpret_cast<const long long*>(gear_common::table_64),
            indices, 8);
    }
    return _mm256_cvtepu8_epi64(incoming_bytes);
}

template <bool UseGearTable>
static inline void load_contribution_batch_32_avx2(
    const unsigned char* data,
    __m256i (&contributions)[optimization_batch_size]) {
    for (uint32_t i = 0; i < optimization_batch_size; ++i) {
        contributions[i] = load_contributions_32_avx2<UseGearTable>(
            data + i * 8);
    }
}

template <bool UseGearTable>
static inline void load_contribution_batch_64_avx2(
    const unsigned char* data,
    __m256i (&contributions)[optimization_batch_size]) {
    for (uint32_t i = 0; i < optimization_batch_size; ++i) {
        contributions[i] = load_contributions_64_avx2<UseGearTable>(
            data + i * 4);
    }
}
#endif

template <bool UseGearTable, bool SkipSubminimum, bool RepairContext,
          bool Normalize, bool UseBackup>
uint64_t Prefix_Sum_Chunking::find_cutpoint_serial(
    const unsigned char* data, uint64_t size) const {
    if (size <= min_block_size) {
        return size;
    }

    const uint64_t limit = std::min(size, max_block_size);
    uint64_t idx = min_block_size;

    if (use_64bit_prefix_sum) {
        uint64_t sum = initialize_fingerprint<
            uint64_t, UseGearTable, SkipSubminimum, RepairContext>(data, idx);
        const auto scan_until = [&](uint64_t phase_limit,
                                    uint64_t phase_mask) {
            while (idx < phase_limit) {
                sum = update_fingerprint<uint64_t, UseGearTable>(
                    sum, data[idx]);
                if ((sum & phase_mask) == 0) {
                    return idx;
                }
                ++idx;
            }
            return phase_limit;
        };
        if constexpr (Normalize || UseBackup) {
            const uint64_t first_phase = std::min(limit, avg_block_size);
            const uint64_t cut = scan_until(
                first_phase, Normalize ? small_mask : mask);
            if (cut != first_phase) {
                return cut;
            }
        }
        const uint64_t post_average_mask = Normalize ? large_mask : mask;
        if constexpr (UseBackup) {
            while (idx < limit) {
                sum = update_fingerprint<uint64_t, UseGearTable>(
                    sum, data[idx]);
                if ((sum & backup_mask) == 0) {
                    if ((sum & post_average_mask) == 0) {
                        return idx;
                    }
                    const uint64_t backup_idx = idx++;
                    const uint64_t cut = scan_until(
                        limit, post_average_mask);
                    return cut != limit ? cut : backup_idx;
                }
                ++idx;
            }
            return limit;
        }
        return scan_until(limit, post_average_mask);
    } else {
        uint32_t sum = initialize_fingerprint<
            uint32_t, UseGearTable, SkipSubminimum, RepairContext>(data, idx);
        const auto scan_until = [&](uint64_t phase_limit,
                                    uint32_t phase_mask) {
            while (idx < phase_limit) {
                sum = update_fingerprint<uint32_t, UseGearTable>(
                    sum, data[idx]);
                if ((sum & phase_mask) == 0) {
                    return idx;
                }
                ++idx;
            }
            return phase_limit;
        };
        if constexpr (Normalize || UseBackup) {
            const uint64_t first_phase = std::min(limit, avg_block_size);
            const uint64_t cut = scan_until(
                first_phase,
                static_cast<uint32_t>(Normalize ? small_mask : mask));
            if (cut != first_phase) {
                return cut;
            }
        }
        const uint32_t post_average_mask = static_cast<uint32_t>(
            Normalize ? large_mask : mask);
        if constexpr (UseBackup) {
            const uint32_t backup_mask32 =
                static_cast<uint32_t>(backup_mask);
            while (idx < limit) {
                sum = update_fingerprint<uint32_t, UseGearTable>(
                    sum, data[idx]);
                if ((sum & backup_mask32) == 0) {
                    if ((sum & post_average_mask) == 0) {
                        return idx;
                    }
                    const uint64_t backup_idx = idx++;
                    const uint64_t cut = scan_until(
                        limit, post_average_mask);
                    return cut != limit ? cut : backup_idx;
                }
                ++idx;
            }
            return limit;
        }
        return scan_until(limit, post_average_mask);
    }
}

#if defined(__AVX2__)
static inline __m256i scale_contributions_avx2(
    __m256i values, __m256i lane_shifts) {
    return _mm256_sllv_epi32(values, lane_shifts);
}

static inline __m256i scale_mask_avx2(
    __m256i mask, __m256i lane_shifts) {
    return _mm256_sllv_epi32(mask, lane_shifts);
}

static inline __m256i inclusive_prefix_sum_avx2(
    __m256i values, __m256i lane_shifts) {
    // Scaling by descending powers of two turns the rolling recurrence into
    // an ordinary prefix sum. The masks use the same scale below.
    __m256i sums = scale_contributions_avx2(values, lane_shifts);
    sums = _mm256_add_epi32(
        sums, _mm256_slli_si256(sums, 4));
    sums = _mm256_add_epi32(
        sums, _mm256_slli_si256(sums, 8));

    __m256i carry = _mm256_permute2x128_si256(sums, sums, 0x08);
    carry = _mm256_shuffle_epi32(carry, 0xff);
    return _mm256_add_epi32(sums, carry);
}

static inline void schedule_batch_avx2(
    __m256i (&values)[optimization_batch_size]) {
    asm volatile(""
                 : "+v"(values[0]), "+v"(values[1]),
                   "+v"(values[2]), "+v"(values[3]));
}

static inline __m256i inclusive_prefix_sum_64_avx2(__m256i values) {
    __m256i sums = _mm256_add_epi64(
        values, _mm256_slli_epi64(_mm256_slli_si256(values, 8), 1));
    __m256i carry = _mm256_permute2x128_si256(sums, sums, 0x08);
    carry = _mm256_shuffle_epi32(carry, 0xee);
    carry = _mm256_sllv_epi64(
        carry, _mm256_setr_epi64x(0, 0, 1, 2));
    return _mm256_add_epi64(sums, carry);
}

static inline void prefix_sum_batch_32_avx2(
    __m256i (&sums)[optimization_batch_size], __m256i lane_shifts) {
    for (auto& value : sums) {
        value = scale_contributions_avx2(value, lane_shifts);
    }
    schedule_batch_avx2(sums);

    for (auto& value : sums) {
        value = _mm256_add_epi32(value, _mm256_slli_si256(value, 4));
    }
    schedule_batch_avx2(sums);

    for (auto& value : sums) {
        value = _mm256_add_epi32(value, _mm256_slli_si256(value, 8));
    }
    schedule_batch_avx2(sums);

    __m256i carries[optimization_batch_size];
    for (uint32_t i = 0; i < optimization_batch_size; ++i) {
        carries[i] = _mm256_permute2x128_si256(sums[i], sums[i], 0x08);
    }
    schedule_batch_avx2(carries);
    for (auto& carry : carries) {
        carry = _mm256_shuffle_epi32(carry, 0xff);
    }
    schedule_batch_avx2(carries);

    for (uint32_t i = 0; i < optimization_batch_size; ++i) {
        sums[i] = _mm256_add_epi32(sums[i], carries[i]);
    }
    schedule_batch_avx2(sums);
}

static inline void prefix_sum_batch_64_avx2(
    __m256i (&sums)[optimization_batch_size]) {
    for (auto& value : sums) {
        value = inclusive_prefix_sum_64_avx2(value);
    }
}

template <bool UseGearTable, bool SkipSubminimum, bool RepairContext,
          bool Lookahead, bool Normalize, bool UseBackup>
uint64_t Prefix_Sum_Chunking::find_cutpoint_avx2(
    const unsigned char* data, uint64_t size) const {
    if (size <= min_block_size) {
        return size;
    }

    const __m256i zero = _mm256_setzero_si256();

    const uint64_t limit = std::min(size, max_block_size);
    uint64_t idx = min_block_size;
    uint64_t backup_idx = 0;

    if (use_64bit_prefix_sum) {
        uint64_t sum = initialize_fingerprint<
            uint64_t, UseGearTable, SkipSubminimum, RepairContext>(data, idx);
        const __m256i base_shifts = _mm256_setr_epi64x(1, 2, 3, 4);

        for (uint32_t phase = 0;
             phase < (Normalize || UseBackup ? 2u : 1u); ++phase) {
            const uint64_t phase_limit =
                (Normalize || UseBackup) && phase == 0
                ? std::min(limit, avg_block_size)
                : limit;
            const uint64_t phase_mask = Normalize
                ? (phase == 0 ? small_mask : large_mask)
                : mask;
            const __m256i mask_vector = _mm256_set1_epi64x(
                static_cast<long long>(phase_mask));
            if constexpr (Lookahead) {
                while (idx + 4 * optimization_batch_size <= phase_limit) {
                __m256i local_sums[optimization_batch_size];
                load_contribution_batch_64_avx2<UseGearTable>(
                        data + idx, local_sums);
                prefix_sum_batch_64_avx2(local_sums);
                for (uint32_t block = 0; block < optimization_batch_size;
                     ++block) {
                    const __m256i sums = _mm256_add_epi64(
                        local_sums[block],
                        _mm256_sllv_epi64(
                            _mm256_set1_epi64x(static_cast<long long>(sum)),
                            base_shifts));
                    const __m256i matches = _mm256_cmpeq_epi64(
                        _mm256_and_si256(sums, mask_vector), zero);
                    const uint32_t match_mask = static_cast<uint32_t>(
                        _mm256_movemask_pd(_mm256_castsi256_pd(matches)));
                    if (match_mask != 0) {
                        return idx + block * 4 + first_set_bit(match_mask);
                    }
                    if constexpr (UseBackup) {
                        if (phase == 1 && backup_idx == 0) {
                            const __m256i backup_mask_vector =
                                _mm256_set1_epi64x(static_cast<long long>(
                                    backup_mask));
                            const __m256i backup_matches = _mm256_cmpeq_epi64(
                                _mm256_and_si256(
                                    sums, backup_mask_vector), zero);
                            const uint32_t backup_matches_mask =
                                static_cast<uint32_t>(_mm256_movemask_pd(
                                    _mm256_castsi256_pd(backup_matches)));
                            if (backup_matches_mask != 0) {
                                backup_idx = idx + block * 4
                                    + first_set_bit(backup_matches_mask);
                            }
                        }
                    }
                    sum = static_cast<uint64_t>(
                        _mm256_extract_epi64(sums, 3));
                }
                idx += 4 * optimization_batch_size;
            }
            }

            while (idx + 4 <= phase_limit) {
            const __m256i incoming =
                load_contributions_64_avx2<UseGearTable>(data + idx);
            const __m256i sums = _mm256_add_epi64(
                inclusive_prefix_sum_64_avx2(incoming),
                _mm256_sllv_epi64(
                    _mm256_set1_epi64x(static_cast<long long>(sum)),
                    base_shifts));
            const __m256i matches = _mm256_cmpeq_epi64(
                _mm256_and_si256(sums, mask_vector), zero);
            const unsigned int match_mask = static_cast<unsigned int>(
                _mm256_movemask_pd(_mm256_castsi256_pd(matches)));
            if (match_mask != 0) {
                return idx + first_set_bit(match_mask);
            }
            if constexpr (UseBackup) {
                if (phase == 1 && backup_idx == 0) {
                    const __m256i backup_mask_vector =
                        _mm256_set1_epi64x(
                            static_cast<long long>(backup_mask));
                    const __m256i backup_matches = _mm256_cmpeq_epi64(
                        _mm256_and_si256(sums, backup_mask_vector), zero);
                    const uint32_t backup_matches_mask =
                        static_cast<uint32_t>(_mm256_movemask_pd(
                            _mm256_castsi256_pd(backup_matches)));
                    if (backup_matches_mask != 0) {
                        backup_idx = idx
                            + first_set_bit(backup_matches_mask);
                    }
                }
            }

            sum = static_cast<uint64_t>(_mm256_extract_epi64(sums, 3));
            idx += 4;
        }

            while (idx < phase_limit) {
            sum = update_fingerprint<uint64_t, UseGearTable>(sum, data[idx]);
            if ((sum & phase_mask) == 0) {
                return idx;
            }
            if constexpr (UseBackup) {
                if (phase == 1 && backup_idx == 0
                    && (sum & backup_mask) == 0) {
                    backup_idx = idx;
                }
            }
            ++idx;
            }
        }
        return UseBackup && backup_idx != 0 ? backup_idx : limit;
    }

    uint32_t sum = initialize_fingerprint<
        uint32_t, UseGearTable, SkipSubminimum, RepairContext>(data, idx);
    const __m256i lane_shifts = _mm256_setr_epi32(
        7, 6, 5, 4, 3, 2, 1, 0);

    for (uint32_t phase = 0;
         phase < (Normalize || UseBackup ? 2u : 1u); ++phase) {
        const uint64_t phase_limit =
            (Normalize || UseBackup) && phase == 0
            ? std::min(limit, avg_block_size)
            : limit;
        const uint32_t phase_mask = static_cast<uint32_t>(Normalize
            ? (phase == 0 ? small_mask : large_mask)
            : mask);
        const __m256i mask_vector = scale_mask_avx2(
            _mm256_set1_epi32(static_cast<int>(phase_mask)),
            lane_shifts);
        const __m256i backup_mask_vector = scale_mask_avx2(
            _mm256_set1_epi32(static_cast<int>(backup_mask)),
            lane_shifts);
        if constexpr (Lookahead) {
            while (idx + 8 * optimization_batch_size <= phase_limit) {
            __m256i local_sums[optimization_batch_size];
            load_contribution_batch_32_avx2<UseGearTable>(
                    data + idx, local_sums);
            prefix_sum_batch_32_avx2(local_sums, lane_shifts);
            for (uint32_t block = 0; block < optimization_batch_size; ++block) {
                const __m256i sums = _mm256_add_epi32(
                    local_sums[block],
                    _mm256_set1_epi32(static_cast<int>(sum << 8)));
                const __m256i matches = _mm256_cmpeq_epi32(
                    _mm256_and_si256(sums, mask_vector), zero);
                const uint32_t match_mask = static_cast<uint32_t>(
                    _mm256_movemask_ps(_mm256_castsi256_ps(matches)));
                if (match_mask != 0) {
                    return idx + block * 8 + first_set_bit(match_mask);
                }
                if constexpr (UseBackup) {
                    if (phase == 1 && backup_idx == 0) {
                        const __m256i backup_matches = _mm256_cmpeq_epi32(
                            _mm256_and_si256(sums, backup_mask_vector), zero);
                        const uint32_t backup_matches_mask =
                            static_cast<uint32_t>(_mm256_movemask_ps(
                                _mm256_castsi256_ps(backup_matches)));
                        if (backup_matches_mask != 0) {
                            backup_idx = idx + block * 8
                                + first_set_bit(backup_matches_mask);
                        }
                    }
                }
                sum = static_cast<uint32_t>(_mm256_extract_epi32(sums, 7));
            }
            idx += 8 * optimization_batch_size;
        }
        }

        while (idx + 8 <= phase_limit) {
        const __m256i incoming =
            load_contributions_32_avx2<UseGearTable>(data + idx);
        const __m256i sums = _mm256_add_epi32(
            inclusive_prefix_sum_avx2(incoming, lane_shifts),
            _mm256_set1_epi32(static_cast<int>(sum << 8)));
        const __m256i matches = _mm256_cmpeq_epi32(
            _mm256_and_si256(sums, mask_vector), zero);
        const unsigned int match_mask = static_cast<unsigned int>(
            _mm256_movemask_ps(_mm256_castsi256_ps(matches)));
        if (match_mask != 0) {
            return idx + first_set_bit(match_mask);
        }
        if constexpr (UseBackup) {
            if (phase == 1 && backup_idx == 0) {
                const __m256i backup_matches = _mm256_cmpeq_epi32(
                    _mm256_and_si256(sums, backup_mask_vector), zero);
                const uint32_t backup_matches_mask =
                    static_cast<uint32_t>(_mm256_movemask_ps(
                        _mm256_castsi256_ps(backup_matches)));
                if (backup_matches_mask != 0) {
                    backup_idx = idx + first_set_bit(backup_matches_mask);
                }
            }
        }

        sum = static_cast<uint32_t>(
            _mm256_extract_epi32(sums, 7));
        idx += 8;
    }

        while (idx < phase_limit) {
        sum = update_fingerprint<uint32_t, UseGearTable>(sum, data[idx]);
        if ((sum & phase_mask) == 0) {
            return idx;
        }
        if constexpr (UseBackup) {
            if (phase == 1 && backup_idx == 0
                && (sum & static_cast<uint32_t>(backup_mask)) == 0) {
                backup_idx = idx;
            }
        }
        ++idx;
        }
    }
    return UseBackup && backup_idx != 0 ? backup_idx : limit;
}
#endif

#if defined(__AVX512F__)
uint64_t Prefix_Sum_Chunking::find_cutpoint_actually_just_prefixsum_avx512(
    const unsigned char* data, uint64_t size) const {
    if (size <= min_block_size) {
        return size;
    }

    constexpr uint64_t window_size = 32;
    const uint64_t limit = std::min(size, max_block_size);
    const __m512i zero = _mm512_setzero_si512();
    const uint32_t window_mask = static_cast<uint32_t>(
        std::max<uint64_t>(avg_block_size / window_size, 1) - 1);
    const __m512i mask_vector = _mm512_set1_epi32(
        static_cast<int>(window_mask));
    const __m512i lane_shifts = _mm512_setr_epi32(
        15, 14, 13, 12, 11, 10, 9, 8,
        7, 6, 5, 4, 3, 2, 1, 0);

    uint32_t sum = 0;
    uint64_t idx = min_block_size > window_size
        ? min_block_size - window_size
        : 0;

    while (idx < min_block_size) {
        sum += data[idx];
        ++idx;
    }

    while (idx < limit && idx < window_size) {
        sum += data[idx];
        if ((sum & window_mask) == 0) {
            return idx;
        }
        ++idx;
    }

    while (idx + 16 <= limit) {
        // Each lane is the change between consecutive 32-byte windows.
        const __m128i incoming_bytes = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(data + idx));
        __m512i sums = _mm512_cvtepu8_epi32(incoming_bytes);

        // Prefix-summing the changes produces all 16 rolling window sums.
        sums = _mm512_sllv_epi32(sums, lane_shifts);
        sums = _mm512_add_epi32(
            sums, _mm512_alignr_epi32(sums, zero, 15));
        sums = _mm512_add_epi32(
            sums, _mm512_alignr_epi32(sums, zero, 14));
        sums = _mm512_add_epi32(
            sums, _mm512_alignr_epi32(sums, zero, 12));
        sums = _mm512_add_epi32(
            sums, _mm512_alignr_epi32(sums, zero, 8));
        sums = _mm512_add_epi32(
            sums, _mm512_set1_epi32(static_cast<int>(sum)));

        // Check for results
        const __mmask16 matches = _mm512_cmpeq_epi32_mask(
            _mm512_and_si512(sums, mask_vector), zero);
        if (matches != 0) {
            return idx + static_cast<uint64_t>(
                __builtin_ctz(static_cast<uint32_t>(matches)));
        }

        const __m128i final_quarter =
            _mm512_extracti32x4_epi32(sums, 3);
        sum = static_cast<uint32_t>(_mm_extract_epi32(final_quarter, 3));
        idx += 16;
    }

    while (idx < limit) {
        sum -= data[idx - window_size];
        sum += data[idx];
        if ((sum & window_mask) == 0) {
            return idx;
        }
        ++idx;
    }

    return limit;
}

template <int LaneShift>
static inline __m512i shift_lanes_left_avx512(__m512i values) {
    static_assert(LaneShift > 0 && LaneShift < 16,
                  "AVX-512 lane shift must be between 1 and 15");
    return _mm512_alignr_epi32(
        values, _mm512_setzero_si512(), 16 - LaneShift);
}

template <int LaneShift>
static inline __m512i shift_64bit_lanes_left_avx512(__m512i values) {
    static_assert(LaneShift > 0 && LaneShift < 8,
                  "AVX-512 lane shift must be between 1 and 7");
    return _mm512_alignr_epi64(
        values, _mm512_setzero_si512(), 8 - LaneShift);
}

static inline __m512i scale_contributions_avx512(
    __m512i values, __m512i lane_shifts) {
    return _mm512_sllv_epi32(values, lane_shifts);
}

static inline __m512i scale_mask_avx512(
    __m512i mask, __m512i lane_shifts) {
    return _mm512_sllv_epi32(mask, lane_shifts);
}

static inline __m512i inclusive_prefix_sum_avx512(
    __m512i values, __m512i lane_shifts) {
    // Scaling by descending powers of two turns the rolling recurrence into
    // an ordinary prefix sum. The masks use the same scale below.
    __m512i sums = scale_contributions_avx512(values, lane_shifts);
    sums = _mm512_add_epi32(
        sums, shift_lanes_left_avx512<1>(sums));
    sums = _mm512_add_epi32(
        sums, shift_lanes_left_avx512<2>(sums));
    sums = _mm512_add_epi32(
        sums, shift_lanes_left_avx512<4>(sums));
    sums = _mm512_add_epi32(
        sums, shift_lanes_left_avx512<8>(sums));
    return sums;
}

static inline void schedule_batch_avx512(
    __m512i (&values)[optimization_batch_size]) {
    asm volatile(""
                 : "+v"(values[0]), "+v"(values[1]),
                   "+v"(values[2]), "+v"(values[3]));
}

static inline __m512i inclusive_prefix_sum_64_avx512(__m512i values) {
    __m512i sums = values;
    sums = _mm512_add_epi64(
        sums, _mm512_slli_epi64(
            shift_64bit_lanes_left_avx512<1>(sums), 1));
    sums = _mm512_add_epi64(
        sums, _mm512_slli_epi64(
            shift_64bit_lanes_left_avx512<2>(sums), 2));
    sums = _mm512_add_epi64(
        sums, _mm512_slli_epi64(
            shift_64bit_lanes_left_avx512<4>(sums), 4));
    return sums;
}

static inline void prefix_sum_batch_32_avx512(
    __m512i (&sums)[optimization_batch_size], __m512i lane_shifts) {
    for (auto& value : sums) {
        value = scale_contributions_avx512(value, lane_shifts);
    }
    schedule_batch_avx512(sums);

    for (auto& value : sums) {
        value = _mm512_add_epi32(
            value, shift_lanes_left_avx512<1>(value));
    }
    schedule_batch_avx512(sums);
    for (auto& value : sums) {
        value = _mm512_add_epi32(
            value, shift_lanes_left_avx512<2>(value));
    }
    schedule_batch_avx512(sums);
    for (auto& value : sums) {
        value = _mm512_add_epi32(
            value, shift_lanes_left_avx512<4>(value));
    }
    schedule_batch_avx512(sums);
    for (auto& value : sums) {
        value = _mm512_add_epi32(
            value, shift_lanes_left_avx512<8>(value));
    }
    schedule_batch_avx512(sums);
}

static inline void prefix_sum_batch_64_avx512(
    __m512i (&sums)[optimization_batch_size]) {
    for (auto& value : sums) {
        value = inclusive_prefix_sum_64_avx512(value);
    }
}

template <bool UseGearTable>
static inline __m512i load_contributions_32_avx512(
    const unsigned char* data) {
    // Sixteen packed bytes widen to all sixteen 32-bit AVX-512 lanes.
    const __m128i incoming_bytes = _mm_loadu_si128(
        reinterpret_cast<const __m128i*>(data));
    const __m512i values = _mm512_cvtepu8_epi32(incoming_bytes);
    if constexpr (UseGearTable) {
        return _mm512_i32gather_epi32(values, gear_common::table_32, 4);
    }
    return values;
}

template <bool UseGearTable>
static inline __m512i load_contributions_64_avx512(
    const unsigned char* data) {
    // Eight packed bytes widen to all eight 64-bit AVX-512 lanes.
    const __m128i incoming_bytes = _mm_loadu_si64(data);
    if constexpr (UseGearTable) {
        const __m256i indices = _mm256_cvtepu8_epi32(incoming_bytes);
        return _mm512_i32gather_epi64(indices, gear_common::table_64, 8);
    }
    return _mm512_cvtepu8_epi64(incoming_bytes);
}

template <bool UseGearTable>
static inline void load_contribution_batch_32_avx512(
    const unsigned char* data,
    __m512i (&contributions)[optimization_batch_size]) {
    for (uint32_t i = 0; i < optimization_batch_size; ++i) {
        contributions[i] = load_contributions_32_avx512<UseGearTable>(
            data + i * 16);
    }
}

template <bool UseGearTable>
static inline void load_contribution_batch_64_avx512(
    const unsigned char* data,
    __m512i (&contributions)[optimization_batch_size]) {
    for (uint32_t i = 0; i < optimization_batch_size; ++i) {
        contributions[i] = load_contributions_64_avx512<UseGearTable>(
            data + i * 8);
    }
}

template <bool UseGearTable, bool SkipSubminimum, bool RepairContext,
          bool Lookahead, bool Normalize, bool UseBackup>
uint64_t Prefix_Sum_Chunking::find_cutpoint_avx512(
    const unsigned char* data, uint64_t size) const {
    if (size <= min_block_size) {
        return size;
    }

    const uint64_t limit = std::min(size, max_block_size);
    uint64_t idx = min_block_size;
    uint64_t backup_idx = 0;

    if (use_64bit_prefix_sum) {
        uint64_t sum = initialize_fingerprint<
            uint64_t, UseGearTable, SkipSubminimum, RepairContext>(data, idx);
        const __m512i base_shifts = _mm512_setr_epi64(
            1, 2, 3, 4, 5, 6, 7, 8);

        for (uint32_t phase = 0;
             phase < (Normalize || UseBackup ? 2u : 1u); ++phase) {
            const uint64_t phase_limit =
                (Normalize || UseBackup) && phase == 0
                ? std::min(limit, avg_block_size)
                : limit;
            const uint64_t phase_mask = Normalize
                ? (phase == 0 ? small_mask : large_mask)
                : mask;
            const __m512i mask_vector = _mm512_set1_epi64(
                static_cast<long long>(phase_mask));
            if constexpr (Lookahead) {
                while (idx + 8 * optimization_batch_size <= phase_limit) {
                __m512i local_sums[optimization_batch_size];
                load_contribution_batch_64_avx512<UseGearTable>(
                        data + idx, local_sums);
                prefix_sum_batch_64_avx512(local_sums);
                for (uint32_t block = 0; block < optimization_batch_size;
                     ++block) {
                    const __m512i sums = _mm512_add_epi64(
                        local_sums[block],
                        _mm512_sllv_epi64(
                            _mm512_set1_epi64(static_cast<long long>(sum)),
                            base_shifts));
                    const __mmask8 matches = _mm512_cmpeq_epi64_mask(
                        _mm512_and_si512(sums, mask_vector),
                        _mm512_setzero_si512());
                    if (matches != 0) {
                        return idx + block * 8
                            + first_set_bit(static_cast<uint32_t>(matches));
                    }
                    if constexpr (UseBackup) {
                        if (phase == 1 && backup_idx == 0) {
                            const __mmask8 backup_matches =
                                _mm512_cmpeq_epi64_mask(
                                    _mm512_and_si512(
                                        sums, _mm512_set1_epi64(
                                            static_cast<long long>(
                                                backup_mask))),
                                    _mm512_setzero_si512());
                            if (backup_matches != 0) {
                                backup_idx = idx + block * 8
                                    + first_set_bit(static_cast<uint32_t>(
                                        backup_matches));
                            }
                        }
                    }
                    const __m128i final_quarter =
                        _mm512_extracti32x4_epi32(sums, 3);
                    sum = static_cast<uint64_t>(
                        _mm_extract_epi64(final_quarter, 1));
                }
                idx += 8 * optimization_batch_size;
            }
            }

            while (idx + 8 <= phase_limit) {
            const __m512i incoming =
                load_contributions_64_avx512<UseGearTable>(data + idx);
            const __m512i sums = _mm512_add_epi64(
                inclusive_prefix_sum_64_avx512(incoming),
                _mm512_sllv_epi64(
                    _mm512_set1_epi64(static_cast<long long>(sum)),
                    base_shifts));
            const __mmask8 matches = _mm512_cmpeq_epi64_mask(
                _mm512_and_si512(sums, mask_vector),
                _mm512_setzero_si512());
            if (matches != 0) {
                return idx + first_set_bit(static_cast<uint32_t>(matches));
            }
            if constexpr (UseBackup) {
                if (phase == 1 && backup_idx == 0) {
                    const __mmask8 backup_matches = _mm512_cmpeq_epi64_mask(
                        _mm512_and_si512(
                            sums, _mm512_set1_epi64(
                                static_cast<long long>(backup_mask))),
                        _mm512_setzero_si512());
                    if (backup_matches != 0) {
                        backup_idx = idx + first_set_bit(
                            static_cast<uint32_t>(backup_matches));
                    }
                }
            }

            const __m128i final_quarter = _mm512_extracti32x4_epi32(sums, 3);
            sum = static_cast<uint64_t>(_mm_extract_epi64(final_quarter, 1));
            idx += 8;
        }

            while (idx < phase_limit) {
            sum = update_fingerprint<uint64_t, UseGearTable>(sum, data[idx]);
            if ((sum & phase_mask) == 0) {
                return idx;
            }
            if constexpr (UseBackup) {
                if (phase == 1 && backup_idx == 0
                    && (sum & backup_mask) == 0) {
                    backup_idx = idx;
                }
            }
            ++idx;
            }
        }
        return UseBackup && backup_idx != 0 ? backup_idx : limit;
    }

    uint32_t sum = initialize_fingerprint<
        uint32_t, UseGearTable, SkipSubminimum, RepairContext>(data, idx);
    const __m512i lane_shifts = _mm512_setr_epi32(
        15, 14, 13, 12, 11, 10, 9, 8,
        7, 6, 5, 4, 3, 2, 1, 0);
    const __m512i zero = _mm512_setzero_si512();

    for (uint32_t phase = 0;
         phase < (Normalize || UseBackup ? 2u : 1u); ++phase) {
        const uint64_t phase_limit =
            (Normalize || UseBackup) && phase == 0
            ? std::min(limit, avg_block_size)
            : limit;
        const uint32_t phase_mask = static_cast<uint32_t>(Normalize
            ? (phase == 0 ? small_mask : large_mask)
            : mask);
        const __m512i mask_vector = scale_mask_avx512(
            _mm512_set1_epi32(static_cast<int>(phase_mask)),
            lane_shifts);
        const __m512i backup_mask_vector = scale_mask_avx512(
            _mm512_set1_epi32(static_cast<int>(backup_mask)),
            lane_shifts);
        if constexpr (Lookahead) {
            while (idx + 16 * optimization_batch_size <= phase_limit) {
            __m512i local_sums[optimization_batch_size];
            load_contribution_batch_32_avx512<UseGearTable>(
                    data + idx, local_sums);
            prefix_sum_batch_32_avx512(local_sums, lane_shifts);
            for (uint32_t block = 0; block < optimization_batch_size; ++block) {
                const __m512i sums = _mm512_add_epi32(
                    local_sums[block],
                    _mm512_set1_epi32(static_cast<int>(sum << 16)));
                const __mmask16 matches = _mm512_cmpeq_epi32_mask(
                    _mm512_and_si512(sums, mask_vector), zero);
                if (matches != 0) {
                    return idx + block * 16
                        + first_set_bit(static_cast<uint32_t>(matches));
                }
                if constexpr (UseBackup) {
                    if (phase == 1 && backup_idx == 0) {
                        const __mmask16 backup_matches =
                            _mm512_cmpeq_epi32_mask(
                                _mm512_and_si512(
                                    sums, backup_mask_vector),
                                zero);
                        if (backup_matches != 0) {
                            backup_idx = idx + block * 16
                                + first_set_bit(static_cast<uint32_t>(
                                    backup_matches));
                        }
                    }
                }
                const __m128i final_quarter =
                    _mm512_extracti32x4_epi32(sums, 3);
                sum = static_cast<uint32_t>(
                    _mm_extract_epi32(final_quarter, 3));
            }
            idx += 16 * optimization_batch_size;
        }
        }

        while (idx + 16 <= phase_limit) {
        const __m512i incoming =
            load_contributions_32_avx512<UseGearTable>(data + idx);
        const __m512i sums = _mm512_add_epi32(
            inclusive_prefix_sum_avx512(incoming, lane_shifts),
            _mm512_set1_epi32(static_cast<int>(sum << 16)));
        const __mmask16 matches = _mm512_cmpeq_epi32_mask(
            _mm512_and_si512(sums, mask_vector), zero);
        if (matches != 0) {
            return idx + first_set_bit(static_cast<uint32_t>(matches));
        }
        if constexpr (UseBackup) {
            if (phase == 1 && backup_idx == 0) {
                const __mmask16 backup_matches = _mm512_cmpeq_epi32_mask(
                    _mm512_and_si512(
                        sums, backup_mask_vector), zero);
                if (backup_matches != 0) {
                    backup_idx = idx + first_set_bit(
                        static_cast<uint32_t>(backup_matches));
                }
            }
        }

        const __m128i final_quarter = _mm512_extracti32x4_epi32(sums, 3);
        sum = static_cast<uint32_t>(_mm_extract_epi32(final_quarter, 3));
        idx += 16;
    }

        while (idx < phase_limit) {
        sum = update_fingerprint<uint32_t, UseGearTable>(sum, data[idx]);
        if ((sum & phase_mask) == 0) {
            return idx;
        }
        if constexpr (UseBackup) {
            if (phase == 1 && backup_idx == 0
                && (sum & static_cast<uint32_t>(backup_mask)) == 0) {
                backup_idx = idx;
            }
        }
        ++idx;
        }
    }
    return UseBackup && backup_idx != 0 ? backup_idx : limit;
}
#endif

uint64_t Prefix_Sum_Chunking::find_cutpoint(char* data, uint64_t size) {
    const auto* bytes = reinterpret_cast<const unsigned char*>(data);
#define CALL_OPTIMIZED(Method, GearTable, Skip, Repair, Lookahead)      \
    if (use_supercdc_backup) {                                          \
        return normalization_level != 0                                \
            ? Method<GearTable, Skip, Repair, Lookahead, true, true>(   \
                bytes, size)                                            \
            : Method<GearTable, Skip, Repair, Lookahead, false, true>(  \
                bytes, size);                                           \
    }                                                                   \
    return normalization_level != 0                                    \
        ? Method<GearTable, Skip, Repair, Lookahead, true, false>(      \
            bytes, size)                                                \
        : Method<GearTable, Skip, Repair, Lookahead, false, false>(     \
            bytes, size)
#define CALL_SERIAL(GearTable, Skip, Repair)                            \
    if (use_supercdc_backup) {                                          \
        return normalization_level != 0                                \
            ? find_cutpoint_serial<GearTable, Skip, Repair, true, true>(\
                bytes, size)                                            \
            : find_cutpoint_serial<GearTable, Skip, Repair, false, true>(\
                bytes, size);                                           \
    }                                                                   \
    return normalization_level != 0                                    \
        ? find_cutpoint_serial<GearTable, Skip, Repair, true, false>(   \
            bytes, size)                                                \
        : find_cutpoint_serial<GearTable, Skip, Repair, false, false>(  \
            bytes, size)
#define DISPATCH_MODE(Method, GearTable)                                \
    if (use_subminimum_skipping) {                                      \
        if (use_context_repair) {                                       \
            if (contribution_lookahead) {                               \
                CALL_OPTIMIZED(Method, GearTable, true, true, true);    \
            }                                                           \
            CALL_OPTIMIZED(Method, GearTable, true, true, false);       \
        }                                                               \
        if (contribution_lookahead) {                                   \
            CALL_OPTIMIZED(Method, GearTable, true, false, true);       \
        }                                                               \
        CALL_OPTIMIZED(Method, GearTable, true, false, false);          \
    }                                                                   \
    if (contribution_lookahead) {                                       \
        CALL_OPTIMIZED(Method, GearTable, false, false, true);          \
    }                                                                   \
    CALL_OPTIMIZED(Method, GearTable, false, false, false)
#define DISPATCH_OPTIMIZED(Method)                                      \
    if (use_gear_table_lookup) {                                        \
        DISPATCH_MODE(Method, true);                                    \
    }                                                                   \
    DISPATCH_MODE(Method, false)

    switch (simd_mode) {
        case SIMD_Mode::NONE:
            if (use_gear_table_lookup) {
                if (!use_subminimum_skipping) {
                    CALL_SERIAL(true, false, false);
                }
                if (use_context_repair) {
                    CALL_SERIAL(true, true, true);
                }
                CALL_SERIAL(true, true, false);
            }
            if (!use_subminimum_skipping) {
                CALL_SERIAL(false, false, false);
            }
            if (use_context_repair) {
                CALL_SERIAL(false, true, true);
            }
            CALL_SERIAL(false, true, false);
#if defined(__AVX2__)
        case SIMD_Mode::AVX256:
            DISPATCH_OPTIMIZED(find_cutpoint_avx2);
            break;
#endif
#if defined(__AVX512F__)
        case SIMD_Mode::AVX512:
            if (!use_64bit_prefix_sum
                && !use_gear_table_lookup
                && !use_subminimum_skipping
                && !use_context_repair
                && !contribution_lookahead
                && normalization_level == 0
                && !use_supercdc_backup) {
                return find_cutpoint_actually_just_prefixsum_avx512(
                    bytes, size);
            }
            DISPATCH_OPTIMIZED(find_cutpoint_avx512);
            break;
#endif
        default:
            std::cerr << "SIMD mode unsupported for prefix-sum chunking"
                      << std::endl;
            std::exit(EXIT_FAILURE);
    }
#undef DISPATCH_OPTIMIZED
#undef DISPATCH_MODE
#undef CALL_SERIAL
#undef CALL_OPTIMIZED
    std::abort();
}
