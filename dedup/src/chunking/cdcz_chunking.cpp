
/**
 * @file cdcz_chunking.cpp
 * @author NicolasComerci
 * @brief Implementations for cdcz chunking technique
 * @version 0.1
 * @date 2026-7-20
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "cdcz_chunking.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>

#ifdef _WIN32
#include <malloc.h>
#endif

#include <immintrin.h>

#include "compiler_compat.hpp"
#include "config_error.hpp"
#include "gear_common.hpp"

extern bool disable_hashing;

Cdcz_Chunking::Cdcz_Chunking(const Config& config) {
    min_block_size = config.get_gear_min_block_size();
    max_block_size = config.get_gear_max_block_size();
    avg_block_size = config.get_gear_avg_block_size();

    int number_of_ones = log2(avg_block_size);

    // to produce chunks that are 8KB. Using the
    // formula log2(8192) = 13, we can use a mask with the 13 most-significant
    // bits set to 1
    mask = 0x0000000000000000;
    for (int i = 0; i < 64; i++) {
        if (number_of_ones-- >= 0) {
            mask += 1;
        }
        mask = mask << 1;
    }

    simd_mode = config.get_simd_mode();
	use_64bit_gear = config.get_use_64bit_gear();
	if (use_64bit_gear) {
		throw ConfigError("64-bit Gear is unsupported for CDCZ");
	}
	optimization_level = config.get_sscdc_optimization_level();
}


static inline void doGear_serial(uint8_t c, uint32_t* x) {
	*x = (*x << 1) + gear_common::table_32[c];
}

#ifndef _MSC_VER
#define vector_idx(table, idx) \
	((__v16si)table)[idx]
#else
#define vector_idx(table, idx) \
	table.m512i_i32[idx]
#endif
#define HASHLEN 256
#define GEAR_HASHLEN 32

inline void* portable_aligned_alloc(std::size_t alignment, std::size_t size) {
	if (alignment == 0) {
		return nullptr;
	}
	const std::size_t minimum_size = std::max(size, alignment);
	if (minimum_size > std::numeric_limits<std::size_t>::max() - (alignment - 1)) {
		return nullptr;
	}
	const std::size_t aligned_size = ((minimum_size + alignment - 1) / alignment) * alignment;
#ifdef _WIN32
	return _aligned_malloc(aligned_size, alignment);
#elif defined(__APPLE__) || defined(__MACH__)
	// macOS: posix_memalign is safest
	void* ptr = nullptr;
	if (posix_memalign(&ptr, alignment, aligned_size) != 0) {
		return nullptr;
	}
	return ptr;
#else
	// Linux and others with C11 support
	return std::aligned_alloc(alignment, aligned_size);
#endif
}

inline void portable_aligned_free(void* ptr) {
#ifdef _WIN32
	_aligned_free(ptr);
#else
	std::free(ptr);
#endif
}

enum CutPointCandidateType : uint8_t {
	HARD_CUT_MASK,  // Satisfied harder mask before average size (FastCDC normalized chunking)
	EASY_CUT_MASK,  // Satisfied easier mask after average size (FastCDC normalized chunking)
	SUPERCDC_BACKUP_MASK,  // Satisfied SuperCDC backup mask because no other mask worked
	MAX_SIZE,  // Forcibly cut because the data size reached the chunk max allowed size
	EOF_CUT  // Forcibly cut because the data span reached its EOF
  };

struct CutPointCandidate {
	CutPointCandidateType type;
	uint64_t offset;
};

#if defined(__AVX2__) || defined(__AVX512F__) || !defined(NDEBUG)
struct CdczMasks {
	uint32_t hard;
	uint32_t easy;
	uint32_t backup;
};

static CdczMasks make_level_one_masks(uint32_t high_average_mask, uint64_t avg_block_size) {
	const int mask_bits = static_cast<int>(std::round(std::log2(avg_block_size)));
	const uint32_t average_mask = high_average_mask >> (32 - mask_bits);
	const uint32_t hard_mask = (average_mask << 1) | 1u;
	const uint32_t easy_mask = average_mask >> 1;
	return {hard_mask, easy_mask, easy_mask >> 1};
}
#endif

// Precondition: cutpoint_bitmap is allocated with at least (file_size / 8) (+1 if not divisible) bytes
// Serial version of SS-CDC phase one to have a simpler, slower but correct implementation for reference
static void cdcz_chunking_phase_one_serial_gear(uint32_t mask, uint64_t min_block_size, unsigned char* RESTRICT file_data, uint64_t file_size, uint8_t* RESTRICT cutpoint_bitmap) {
	uint64_t curr_pos = 0;
	uint32_t hash = 0;
	for (; curr_pos < GEAR_HASHLEN; curr_pos++) {
		hash = (hash << 1) + gear_common::table_32[file_data[curr_pos]];
	}
	for (; curr_pos < file_size; curr_pos++) {
		hash = (hash << 1) + gear_common::table_32[file_data[curr_pos]];
		if (!(hash & mask)) {
			uint8_t bits = cutpoint_bitmap[curr_pos / 8];  // get bits for block of 8 bytes
			bits |= 1 << (curr_pos % 8);  // set appropriate bit
			cutpoint_bitmap[curr_pos / 8] = bits;  // overwrite byte in bitmap with newly set bit
			//printf("Serial: offset %d hash %x\n", offset, hash);
		}
	}
}

static inline bool is_chunk_invariance_condition_satisfied(
  bool is_prev_candidate_hard, uint64_t dist_with_prev, CutPointCandidateType new_candidate_type,
  uint64_t min_size, uint64_t avg_size, uint64_t max_size
) {
	return is_prev_candidate_hard &&
	  // Given that the previous candidate is of HARD type, either it will be used, or it will be discarded which can only happen
	  // if a cut was used with at most min_size distance before the previous candidate. Knowing the previous cut to be used is
	  // at most at distance_w_prev_cut_candidate + min_size distance we can ensure we don't violate max_size if we use the current candidate.
	  (dist_with_prev + min_size <= max_size) &&
	  (
		// We also need to check that the current candidate is actually eligible, a HARD type cut needs to be at least min_size from the previous
		// cut to be valid, whereas an EASY cut needs to be at least avg_size from it.
		// Note that we check using distance_w_prev_cut_candidate, with the same logic as the check we did for the max_size, if it won't be used
		// then the actual distance to the previous cut will be even larger so these conditions will also validate the eligibility of the current
		// candidate in that case
		(new_candidate_type == CutPointCandidateType::HARD_CUT_MASK && dist_with_prev >= min_size) ||
		(new_candidate_type == CutPointCandidateType::EASY_CUT_MASK && dist_with_prev >= avg_size)
	  );
}

#if defined(__AVX2__) || defined(__AVX512F__)
static uint64_t find_supercdc_backup_cut(
		const unsigned char* RESTRICT file_data, uint64_t prev_cut_pos,
		uint64_t avg_block_size, uint64_t max_block_size, uint32_t backup_mask) {
	const uint64_t search_start = prev_cut_pos + avg_block_size;
	const uint64_t search_end = prev_cut_pos + max_block_size;

	const uint64_t warmup_start = search_start >= GEAR_HASHLEN ? search_start - GEAR_HASHLEN : 0;
	uint32_t pattern = 0;
	for (uint64_t pos = warmup_start; pos < search_start; pos++) {
		pattern = (pattern << 1) + gear_common::table_32[file_data[pos]];
	}
	for (uint64_t pos = search_start; pos < search_end; pos++) {
		pattern = (pattern << 1) + gear_common::table_32[file_data[pos]];
		if (!(pattern & backup_mask)) {
			return pos;
		}
	}
	return search_end;
}
#endif

#if defined(__AVX512F__)
#define doGearAvx512(hash, cbytes) {\
					hash = _mm512_slli_epi32(hash, 1);\
					__m512i idx = _mm512_and_epi32(cbytes, cmask);\
					cbytes = _mm512_srli_epi32(cbytes, 8);\
					__m512i tentry = _mm512_i32gather_epi32(idx, gear_common::table_32, 4);\
					hash = _mm512_add_epi32(hash, tentry);}

static inline void load_and_transpose_16x8_epi32_avx512(
	const unsigned char* RESTRICT file_data,
	const uint32_t (&lane_positions)[16],
	const uint32_t (&lane_end_positions)[16],
	__m512i (&columns)[8]);

// Precondition: cutpoint_bitmap is allocated with at least (file_size / 8) (+1 if not divisible) bytes (ideally aligned to 64bytes=512bits)
template<bool SCHEDULE_GATHERS, bool USE_LOAD_TRANSPOSE>
static void cdcz_chunking_phase_one_avx512_gear_impl(
	uint32_t mask,
	uint64_t avg_block_size,
	uint64_t min_block_size,
	uint64_t max_block_size,
	const unsigned char* RESTRICT file_data,
	uint64_t file_size,
	uint8_t* RESTRICT cutpoint_bitmap,
	std::vector<uint64_t>& cutpoints
) {
	static constexpr uint32_t LANE_COUNT = 16;
	(void)cutpoint_bitmap;
	std::array<std::vector<CutPointCandidate>, LANE_COUNT> lane_results{};
	std::array<bool, LANE_COUNT> lane_achieved_chunk_invariance{};
	lane_achieved_chunk_invariance.fill(false);
	lane_achieved_chunk_invariance[0] = true;

	const CdczMasks masks = make_level_one_masks(mask, avg_block_size);
	const uint32_t hard_mask = masks.hard;
	const uint32_t easy_mask = masks.easy;
	const uint32_t backup_mask = masks.backup;
	const __m512i mm_break_mark_hard = _mm512_set1_epi32(hard_mask);
	const __m512i mm_break_mark_easy = _mm512_set1_epi32(easy_mask);
	const __m512i cmask = _mm512_set1_epi32(0xff);
	const __m512i zero_vec = _mm512_setzero_si512();
	const __m512i high_bit_vec = _mm512_set1_epi32(static_cast<int32_t>(1u << 31));
	uint64_t file_data_offset = 0;
	uint64_t total_bytes_left = file_size;
	uint64_t prev_cut_offset = 0;

	const auto consume_lane_results = [&]() {
		for (auto& lane_result : lane_results) {
			for (const auto& candidate : lane_result) {
				if (candidate.offset <= prev_cut_offset) {
					continue;
				}

				while (candidate.offset - prev_cut_offset >= max_block_size) {
					const uint64_t backup_pos = find_supercdc_backup_cut(
						file_data, prev_cut_offset, avg_block_size, max_block_size, backup_mask);
					if (backup_pos < prev_cut_offset + max_block_size) {
						prev_cut_offset = backup_pos;
						cutpoints.emplace_back(prev_cut_offset);
					}
					else if (candidate.offset != prev_cut_offset + max_block_size) {
						prev_cut_offset += max_block_size;
						cutpoints.emplace_back(prev_cut_offset);
					}
					else break;
				}

				const uint64_t dist_with_prev = candidate.offset - prev_cut_offset;
				const bool candidate_is_eligible =
					(candidate.type == CutPointCandidateType::HARD_CUT_MASK && dist_with_prev >= min_block_size) ||
					(candidate.type == CutPointCandidateType::EASY_CUT_MASK && dist_with_prev >= avg_block_size) ||
					candidate.type == CutPointCandidateType::SUPERCDC_BACKUP_MASK ||
					candidate.type == CutPointCandidateType::MAX_SIZE ||
					candidate.type == CutPointCandidateType::EOF_CUT;
				if (candidate_is_eligible) {
					prev_cut_offset = candidate.offset;
					cutpoints.emplace_back(prev_cut_offset);
				}
			}
			lane_result.clear();
		}
	};

	while (total_bytes_left > std::max<uint64_t>(2ull * LANE_COUNT * GEAR_HASHLEN, min_block_size)) {  // if bytes left are too few just complete the remainder with serial version
		// Ensure the window offsets don't overflow (2GB is the max i32 value, 1GB is good enough)
		uint64_t window_bytes = std::min<uint64_t>(1 << 30, total_bytes_left);
		// SS-CDC would only need GEAR_HASHLEN - 1 of overlap between lane/thread segments, but we just use GEAR_HASHLEN as its easier.
		// Also ensure each lane has a even number of GEAR_HASHLEN bytes sets of data to roll GEAR to.
		// We do this so when we scatter the bitmap results we do it efficiently.
		const uint64_t bytes_per_lane_without_overlap = (((window_bytes - GEAR_HASHLEN) / LANE_COUNT) / GEAR_HASHLEN) * GEAR_HASHLEN;
		// Add GEAR_HASHLEN overlap and we still have an even number of GEAR_HASHLEN bytes=8 Gather instructions
		const uint64_t bytes_per_lane = bytes_per_lane_without_overlap + GEAR_HASHLEN;
		// Readjust window_bytes as it might now be a few bytes smaller because of divisibility
		window_bytes = (bytes_per_lane_without_overlap * LANE_COUNT) + GEAR_HASHLEN;

		const uint64_t expected_candidates_per_lane =
			((bytes_per_lane + avg_block_size - 1) / avg_block_size) + 2;
		for (auto& lane_result : lane_results) {
			lane_result.reserve(expected_candidates_per_lane);
		}

		__m512i vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
		// bytes_per_lane_without_overlap used for each lane's start pos, lanes overlap at the end of their data into next lane's data.
		// This applies even for the last lane, the next iteration will start at the start of the last lane's overlap data.
		vindex = _mm512_mullo_epi32(vindex, _mm512_set1_epi32(bytes_per_lane_without_overlap));
		const __m512i vindex_end = _mm512_add_epi32(
			vindex,
			_mm512_set1_epi32(static_cast<int32_t>(bytes_per_lane))
		);
		alignas(64) uint32_t lane_end_positions[LANE_COUNT];
		_mm512_store_si512(reinterpret_cast<__m512i*>(lane_end_positions), vindex_end);
		std::array<uint64_t, LANE_COUNT> lane_candidate_floor{};

		__m512i hash = zero_vec;
		__m512i candidates_hard_vmask = zero_vec;
		__m512i candidates_easy_vmask = zero_vec;
		__m512i warmup_cbytes[8]{};
		if constexpr (USE_LOAD_TRANSPOSE) {
			alignas(64) uint32_t lane_positions[LANE_COUNT];
			_mm512_store_si512(reinterpret_cast<__m512i*>(lane_positions), vindex);
			load_and_transpose_16x8_epi32_avx512(
				file_data + file_data_offset,
				lane_positions,
				lane_end_positions,
				warmup_cbytes);
		}
		// First do the "warmup" so each lane gets the valid GEAR hash pattern after the overlap with the previous lane's data
		// ( for the first lane on the first segment we assume the minimum chunk size >= GEAR_HASHLEN )
		for (int warmup_iter = 0; warmup_iter < 8; warmup_iter++) {
			__m512i cbytes = USE_LOAD_TRANSPOSE
				? warmup_cbytes[warmup_iter]
				: _mm512_i32gather_epi32(
					vindex, file_data + file_data_offset + (4 * warmup_iter), 1);
			for (uint64_t j = 0; j < sizeof(int32_t); j++) {
				doGearAvx512(hash, cbytes);
			}
		}
		vindex = _mm512_add_epi32(vindex, _mm512_set1_epi32(32));
		__mmask16 lanes_at_end = _mm512_cmpeq_epi32_mask(vindex, vindex_end);
		while (lanes_at_end != static_cast<__mmask16>(0xffff)) {
			const __mmask16 active_lanes_mask = static_cast<__mmask16>(~lanes_at_end);
			const __m512i candidate_high_bit_vec = _mm512_maskz_mov_epi32(active_lanes_mask, high_bit_vec);
			const auto record_candidate_masks = [&](const __m512i current_hash) {
					const __mmask16 lane_easy_mask = static_cast<__mmask16>(
						_mm512_testn_epi32_mask(current_hash, mm_break_mark_easy) & active_lanes_mask);
					candidates_easy_vmask = _mm512_srli_epi32(candidates_easy_vmask, 1);
					candidates_hard_vmask = _mm512_srli_epi32(candidates_hard_vmask, 1);
					if (lane_easy_mask == 0) {
						return;
					}
					candidates_easy_vmask = _mm512_or_epi32(
						candidates_easy_vmask,
						_mm512_maskz_mov_epi32(lane_easy_mask, candidate_high_bit_vec)
					);
					const __mmask16 lane_hard_mask = static_cast<__mmask16>(
						_mm512_testn_epi32_mask(current_hash, mm_break_mark_hard) & active_lanes_mask);
					candidates_hard_vmask = _mm512_or_epi32(
						candidates_hard_vmask,
						_mm512_maskz_mov_epi32(lane_hard_mask, candidate_high_bit_vec)
					);
			};

			if constexpr (SCHEDULE_GATHERS) {
				__m512i cbytes_by_gather[8];
				__m512i tentries_by_gather[8][4];
				if constexpr (USE_LOAD_TRANSPOSE) {
					alignas(64) uint32_t lane_positions[LANE_COUNT];
					_mm512_store_si512(reinterpret_cast<__m512i*>(lane_positions), vindex);
					load_and_transpose_16x8_epi32_avx512(
						file_data + file_data_offset,
						lane_positions,
						lane_end_positions,
						cbytes_by_gather);
				}
				else {
					for (int inner_gather_i = 0; inner_gather_i < 8; inner_gather_i++) {
						cbytes_by_gather[inner_gather_i] = _mm512_mask_i32gather_epi32(
							zero_vec, active_lanes_mask, vindex,
							file_data + file_data_offset + (4 * inner_gather_i), 1);
					}
				}
				for (int inner_gather_i = 0; inner_gather_i < 8; inner_gather_i++) {
					const __m512i cbytes = cbytes_by_gather[inner_gather_i];
					const __m512i idx0 = _mm512_and_epi32(cbytes, cmask);
					const __m512i idx1 = _mm512_and_epi32(_mm512_srli_epi32(cbytes, 8), cmask);
					const __m512i idx2 = _mm512_and_epi32(_mm512_srli_epi32(cbytes, 16), cmask);

					const __m512i idx3 = _mm512_srli_epi32(cbytes, 24);
					tentries_by_gather[inner_gather_i][0] = _mm512_i32gather_epi32(idx0, gear_common::table_32, 4);
					tentries_by_gather[inner_gather_i][1] = _mm512_i32gather_epi32(idx1, gear_common::table_32, 4);
					tentries_by_gather[inner_gather_i][2] = _mm512_i32gather_epi32(idx2, gear_common::table_32, 4);
					tentries_by_gather[inner_gather_i][3] = _mm512_i32gather_epi32(idx3, gear_common::table_32, 4);
				}
				for (int inner_gather_i = 0; inner_gather_i < 8; inner_gather_i++) {
					for (uint32_t byte_i = 0; byte_i < 4; byte_i++) {
						hash = _mm512_add_epi32(
							_mm512_slli_epi32(hash, 1),
							tentries_by_gather[inner_gather_i][byte_i]);
						record_candidate_masks(hash);
					}
				}
			}
			else {
				for (int inner_gather_i = 0; inner_gather_i < 8; inner_gather_i++) {
					__m512i cbytes = _mm512_mask_i32gather_epi32(
						zero_vec, active_lanes_mask, vindex,
						file_data + file_data_offset + (4 * inner_gather_i), 1);
					for (uint64_t j = 0; j < sizeof(int32_t); j++) {
						doGearAvx512(hash, cbytes);
						record_candidate_masks(hash);
					}
				}
			}

			if (_mm512_cmpneq_epi32_mask(candidates_easy_vmask, zero_vec) == 0) {
				candidates_easy_vmask = zero_vec;
				candidates_hard_vmask = zero_vec;
				vindex = _mm512_min_epi32(
					_mm512_add_epi32(vindex, _mm512_set1_epi32(GEAR_HASHLEN)),
					vindex_end
				);
				lanes_at_end = _mm512_cmpeq_epi32_mask(vindex, vindex_end);
				continue;
			}

			alignas(64) uint32_t hard_bitmap_words[LANE_COUNT];
			alignas(64) uint32_t easy_bitmap_words[LANE_COUNT];
			alignas(64) uint32_t current_lane_positions[LANE_COUNT];
			alignas(64) uint32_t next_lane_positions[LANE_COUNT];
			_mm512_store_si512(reinterpret_cast<__m512i*>(hard_bitmap_words), candidates_hard_vmask);
			_mm512_store_si512(reinterpret_cast<__m512i*>(easy_bitmap_words), candidates_easy_vmask);
			_mm512_store_si512(reinterpret_cast<__m512i*>(current_lane_positions), vindex);
			std::copy(
				std::begin(current_lane_positions),
				std::end(current_lane_positions),
				std::begin(next_lane_positions)
			);

			for (uint32_t lane_i = 0; lane_i < LANE_COUNT; lane_i++) {
				if (current_lane_positions[lane_i] == lane_end_positions[lane_i]) {
					continue;
				}

				const uint64_t current_batch_end =
					static_cast<uint64_t>(current_lane_positions[lane_i]) + GEAR_HASHLEN;
				const uint64_t normal_next_pos = std::min<uint64_t>(
					current_batch_end,
					lane_end_positions[lane_i]
				);
				uint64_t desired_next_pos = normal_next_pos;
				bool lane_accepted_candidate = false;

				uint32_t candidates_easy_bits = easy_bitmap_words[lane_i];
				while (candidates_easy_bits != 0) {
					const uint32_t bit = _tzcnt_u32(candidates_easy_bits);
					candidates_easy_bits &= candidates_easy_bits - 1;
					const uint64_t candidate_relative_pos =
						static_cast<uint64_t>(current_lane_positions[lane_i]) + bit;
					if (candidate_relative_pos < lane_candidate_floor[lane_i]) {
						continue;
					}

					const uint64_t candidate_pos = file_data_offset + candidate_relative_pos;
					const auto result_type = ((hard_bitmap_words[lane_i] >> bit) & 1u) != 0
						? CutPointCandidateType::HARD_CUT_MASK
						: CutPointCandidateType::EASY_CUT_MASK;
					if (!lane_achieved_chunk_invariance[lane_i]) {
						if (!lane_results[lane_i].empty()) {
							const auto& prev_cut_candidate = lane_results[lane_i].back();
							const uint64_t dist_with_prev = candidate_pos - prev_cut_candidate.offset;
							const bool is_prev_candidate_hard = prev_cut_candidate.type == CutPointCandidateType::HARD_CUT_MASK;
							if (is_chunk_invariance_condition_satisfied(
								is_prev_candidate_hard, dist_with_prev, result_type,
								min_block_size, avg_block_size, max_block_size)) {
								lane_achieved_chunk_invariance[lane_i] = true;
							}
						}
						lane_results[lane_i].push_back({result_type, candidate_pos});
					}
					else {
						uint64_t prev_cut_pos = lane_results[lane_i].empty()
							? prev_cut_offset : lane_results[lane_i].back().offset;
						uint64_t dist_with_prev = candidate_pos - prev_cut_pos;

						while (dist_with_prev >= max_block_size) {
							const uint64_t backup_pos = find_supercdc_backup_cut(
								file_data, prev_cut_pos, avg_block_size, max_block_size, backup_mask);
							if (backup_pos < prev_cut_pos + max_block_size) {
								prev_cut_pos = backup_pos;
								lane_results[lane_i].push_back({CutPointCandidateType::SUPERCDC_BACKUP_MASK, prev_cut_pos});
							}
							else {
								prev_cut_pos += max_block_size;
								lane_results[lane_i].push_back({CutPointCandidateType::MAX_SIZE, prev_cut_pos});
							}
							dist_with_prev = candidate_pos - prev_cut_pos;
						}

						if ((result_type == CutPointCandidateType::HARD_CUT_MASK && dist_with_prev >= min_block_size) ||
							(result_type == CutPointCandidateType::EASY_CUT_MASK && dist_with_prev >= avg_block_size)) {
							lane_results[lane_i].push_back({result_type, candidate_pos});
							lane_candidate_floor[lane_i] = candidate_relative_pos + min_block_size;
							uint64_t washout_start = lane_candidate_floor[lane_i] >= GEAR_HASHLEN
								? lane_candidate_floor[lane_i] - GEAR_HASHLEN : 0;
							washout_start &= ~(static_cast<uint64_t>(GEAR_HASHLEN) - 1);
							desired_next_pos = std::max<uint64_t>(normal_next_pos, washout_start);
							lane_accepted_candidate = true;
						}
					}
				}

				uint64_t next_lane_pos = normal_next_pos;
				if (lane_accepted_candidate) {
					next_lane_pos = std::min<uint64_t>(desired_next_pos, lane_end_positions[lane_i]);
				}
				next_lane_positions[lane_i] = static_cast<uint32_t>(next_lane_pos);
			}

			vindex = _mm512_load_si512(reinterpret_cast<const __m512i*>(next_lane_positions));
			candidates_easy_vmask = zero_vec;
			candidates_hard_vmask = zero_vec;
			lanes_at_end = _mm512_cmpeq_epi32_mask(vindex, vindex_end);
		}

		consume_lane_results();
		lane_achieved_chunk_invariance.fill(false);
		lane_achieved_chunk_invariance[0] = true;

		file_data_offset += window_bytes - GEAR_HASHLEN;
		total_bytes_left -= window_bytes - GEAR_HASHLEN;
	}

	// Deal with any leftover data serially
	file_data_offset = file_data_offset >= GEAR_HASHLEN ? file_data_offset - GEAR_HASHLEN : 0;
	uint32_t pattern = 0;
	// "Regenerate" the gear hash so we can continue with leftover data as if we didn't separate the processing
	for (int j = 0; j < GEAR_HASHLEN && file_data_offset < file_size; j++) {
		pattern = (pattern << 1) + gear_common::table_32[file_data[file_data_offset]];
		file_data_offset++;
	}
	// Finish the remaining data and add results to the cutpoint_bitmap
	while (file_data_offset < file_size) {
		pattern = (pattern << 1) + gear_common::table_32[file_data[file_data_offset]];
		if (!(pattern & easy_mask)) {
			const auto result_type = !(pattern & hard_mask)
				? CutPointCandidateType::HARD_CUT_MASK
				: CutPointCandidateType::EASY_CUT_MASK;
			lane_results[LANE_COUNT - 1].push_back({result_type, file_data_offset});
		}
		file_data_offset++;
	}

	consume_lane_results();

	while (file_size - prev_cut_offset > max_block_size) {
		const uint64_t backup_pos = find_supercdc_backup_cut(
			file_data, prev_cut_offset, avg_block_size, max_block_size, backup_mask);
		prev_cut_offset = backup_pos < prev_cut_offset + max_block_size
			? backup_pos : prev_cut_offset + max_block_size;
		cutpoints.emplace_back(prev_cut_offset);
	}
	// Add a final cut for the end of the file
	if (prev_cut_offset < file_size) {
		cutpoints.emplace_back(file_size);
	}
}

static void cdcz_chunking_phase_one_avx512_gear(
	uint32_t mask,
	uint64_t avg_block_size,
	uint64_t min_block_size,
	uint64_t max_block_size,
	const unsigned char* RESTRICT file_data,
	uint64_t file_size,
	uint8_t* RESTRICT cutpoint_bitmap,
	std::vector<uint64_t>& cutpoints
) {
	cdcz_chunking_phase_one_avx512_gear_impl<false, false>(
		mask, avg_block_size, min_block_size, max_block_size,
		file_data, file_size, cutpoint_bitmap, cutpoints);
}
#undef doGearAvx512

static inline void gather_gear_entries_avx512(
		__m512i cbytes,
		__m512i byte_mask,
		__m512i& tentry0,

		__m512i& tentry1,
		__m512i& tentry2,
		__m512i& tentry3) {
	const __m512i idx0 = _mm512_and_epi32(cbytes, byte_mask);
	const __m512i idx1 = _mm512_and_epi32(_mm512_srli_epi32(cbytes, 8), byte_mask);
	const __m512i idx2 = _mm512_and_epi32(_mm512_srli_epi32(cbytes, 16), byte_mask);
	const __m512i idx3 = _mm512_srli_epi32(cbytes, 24);

	tentry0 = _mm512_i32gather_epi32(idx0, gear_common::table_32, 4);
	tentry1 = _mm512_i32gather_epi32(idx1, gear_common::table_32, 4);
	tentry2 = _mm512_i32gather_epi32(idx2, gear_common::table_32, 4);
	tentry3 = _mm512_i32gather_epi32(idx3, gear_common::table_32, 4);
}

static inline void roll_gear_avx512(__m512i& hash, __m512i tentry) {
	hash = _mm512_add_epi32(_mm512_slli_epi32(hash, 1), tentry);
}

static inline void record_cutpoints_avx512(
		__m512i hash,
		__m512i break_mark,
		__m512i& cutpoint_bitmap_vmask) {
	const __mmask16 lane_cutpoint_mask = _mm512_testn_epi32_mask(hash, break_mark);
	cutpoint_bitmap_vmask = _mm512_srli_epi32(cutpoint_bitmap_vmask, 1);
	if (lane_cutpoint_mask != 0) {
		const __m512i ret = _mm512_maskz_set1_epi32(lane_cutpoint_mask, static_cast<int32_t>(1u << 31));
		cutpoint_bitmap_vmask = _mm512_or_epi32(cutpoint_bitmap_vmask, ret);
	}
}

// Precondition: cutpoint_bitmap is allocated with at least (file_size / 8) (+1 if not divisible) bytes (ideally aligned to 64bytes=512bits)
static void cdcz_chunking_phase_one_avx512_gear_with_gather_scheduling(
	uint32_t mask,
	uint64_t avg_block_size,
	uint64_t min_block_size,
	uint64_t max_block_size,
	const unsigned char* RESTRICT file_data,
	uint64_t file_size,
	uint8_t* RESTRICT cutpoint_bitmap,
	std::vector<uint64_t>& cutpoints
) {
	cdcz_chunking_phase_one_avx512_gear_impl<true, false>(
		mask, avg_block_size, min_block_size, max_block_size,
		file_data, file_size, cutpoint_bitmap, cutpoints);
}
#endif

#if defined(__AVX2__)
#define doGearAvx2(hash, cbytes) {\
					hash = _mm256_slli_epi32(hash, 1);\
					__m256i idx = _mm256_and_si256(cbytes, cmask);\
					cbytes = _mm256_srli_epi32(cbytes, 8);\
					__m256i tentry = _mm256_i32gather_epi32(reinterpret_cast<const int*>(gear_common::table_32), idx, 4);\
					hash = _mm256_add_epi32(hash, tentry);}

static inline void gather_gear_entries_avx2(
		__m256i cbytes,
		__m256i byte_mask,
		__m256i& tentry0,
		__m256i& tentry1,
		__m256i& tentry2,
		__m256i& tentry3) {
	const __m256i idx0 = _mm256_and_si256(cbytes, byte_mask);
	const __m256i idx1 = _mm256_and_si256(_mm256_srli_epi32(cbytes, 8), byte_mask);
	const __m256i idx2 = _mm256_and_si256(_mm256_srli_epi32(cbytes, 16), byte_mask);
	const __m256i idx3 = _mm256_srli_epi32(cbytes, 24);

	tentry0 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(gear_common::table_32), idx0, 4);
	tentry1 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(gear_common::table_32), idx1, 4);
	tentry2 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(gear_common::table_32), idx2, 4);
	tentry3 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(gear_common::table_32), idx3, 4);
}

static inline void roll_gear_avx2(__m256i& hash, __m256i tentry) {
	hash = _mm256_add_epi32(_mm256_slli_epi32(hash, 1), tentry);
}

static inline void record_cutpoints_avx2(
		__m256i hash,
		__m256i break_mark,
		__m256i zero_vec,
		__m256i high_bit_vec,
		__m256i& cutpoint_bitmap_vmask) {
	const __m256i lane_cutpoint_mask = _mm256_cmpeq_epi32(
		_mm256_and_si256(hash, break_mark),
		zero_vec
	);
	cutpoint_bitmap_vmask = _mm256_srli_epi32(cutpoint_bitmap_vmask, 1);
	cutpoint_bitmap_vmask = _mm256_or_si256(
		cutpoint_bitmap_vmask,
		_mm256_and_si256(lane_cutpoint_mask, high_bit_vec)
	);
}

static inline void load_and_transpose_8x8_epi32_avx2(
		const unsigned char* RESTRICT file_data,
		const uint32_t (&lane_positions)[8],
		const uint32_t (&lane_end_positions)[8],
		__m256i (&columns)[8]) {
	assert((reinterpret_cast<std::uintptr_t>(file_data) % GEAR_HASHLEN) == 0);
	__m256i rows[8];
	for (uint32_t lane_i = 0; lane_i < 8; lane_i++) {
		assert((lane_positions[lane_i] % GEAR_HASHLEN) == 0);
		assert((lane_end_positions[lane_i] % GEAR_HASHLEN) == 0);
		rows[lane_i] = lane_positions[lane_i] == lane_end_positions[lane_i]
			? _mm256_setzero_si256()
			: _mm256_load_si256(reinterpret_cast<const __m256i*>(file_data + lane_positions[lane_i]));
	}

	const __m256i unpacked32_0 = _mm256_unpacklo_epi32(rows[0], rows[1]);
	const __m256i unpacked32_1 = _mm256_unpackhi_epi32(rows[0], rows[1]);
	const __m256i unpacked32_2 = _mm256_unpacklo_epi32(rows[2], rows[3]);
	const __m256i unpacked32_3 = _mm256_unpackhi_epi32(rows[2], rows[3]);
	const __m256i unpacked32_4 = _mm256_unpacklo_epi32(rows[4], rows[5]);
	const __m256i unpacked32_5 = _mm256_unpackhi_epi32(rows[4], rows[5]);
	const __m256i unpacked32_6 = _mm256_unpacklo_epi32(rows[6], rows[7]);
	const __m256i unpacked32_7 = _mm256_unpackhi_epi32(rows[6], rows[7]);

	const __m256i unpacked64_0 = _mm256_unpacklo_epi64(unpacked32_0, unpacked32_2);
	const __m256i unpacked64_1 = _mm256_unpackhi_epi64(unpacked32_0, unpacked32_2);
	const __m256i unpacked64_2 = _mm256_unpacklo_epi64(unpacked32_1, unpacked32_3);
	const __m256i unpacked64_3 = _mm256_unpackhi_epi64(unpacked32_1, unpacked32_3);
	const __m256i unpacked64_4 = _mm256_unpacklo_epi64(unpacked32_4, unpacked32_6);
	const __m256i unpacked64_5 = _mm256_unpackhi_epi64(unpacked32_4, unpacked32_6);
	const __m256i unpacked64_6 = _mm256_unpacklo_epi64(unpacked32_5, unpacked32_7);
	const __m256i unpacked64_7 = _mm256_unpackhi_epi64(unpacked32_5, unpacked32_7);

	columns[0] = _mm256_permute2x128_si256(unpacked64_0, unpacked64_4, 0x20);
	columns[1] = _mm256_permute2x128_si256(unpacked64_1, unpacked64_5, 0x20);
	columns[2] = _mm256_permute2x128_si256(unpacked64_2, unpacked64_6, 0x20);
	columns[3] = _mm256_permute2x128_si256(unpacked64_3, unpacked64_7, 0x20);
	columns[4] = _mm256_permute2x128_si256(unpacked64_0, unpacked64_4, 0x31);
	columns[5] = _mm256_permute2x128_si256(unpacked64_1, unpacked64_5, 0x31);
	columns[6] = _mm256_permute2x128_si256(unpacked64_2, unpacked64_6, 0x31);
	columns[7] = _mm256_permute2x128_si256(unpacked64_3, unpacked64_7, 0x31);
}

// AVX2 version of SS-CDC phase one. AVX2 has no scatter instruction, so sparse lane results are stored individually.
// Precondition: cutpoint_bitmap is allocated with at least (file_size / 8) (+1 if not divisible) bytes. (ideally aligned to 32bytes=256bits)
static void cdcz_chunking_phase_one_avx2_gear(
	uint32_t mask,
	uint64_t avg_block_size,
	uint64_t min_block_size,
	uint64_t max_block_size,
	const unsigned char* RESTRICT file_data,
	uint64_t file_size,
	uint8_t* RESTRICT cutpoint_bitmap,
	std::vector<uint64_t>& cutpoints
) {
	static constexpr uint32_t LANE_COUNT = 8;
	(void)cutpoint_bitmap;
	std::array<std::vector<CutPointCandidate>, LANE_COUNT> lane_results{};
	std::array<bool, LANE_COUNT> lane_achieved_chunk_invariance{};
	lane_achieved_chunk_invariance.fill(false);
	lane_achieved_chunk_invariance[0] = true;

	const CdczMasks masks = make_level_one_masks(mask, avg_block_size);
	const uint32_t hard_mask = masks.hard;
	const uint32_t easy_mask = masks.easy;
	const uint32_t backup_mask = masks.backup;
	const __m256i mm_break_mark_hard = _mm256_set1_epi32(hard_mask);
	const __m256i mm_break_mark_easy = _mm256_set1_epi32(easy_mask);
	const __m256i cmask = _mm256_set1_epi32(0xff);
	const __m256i zero_vec = _mm256_setzero_si256();
	const __m256i high_bit_vec = _mm256_set1_epi32(static_cast<int32_t>(1u << 31));
	uint64_t file_data_offset = 0;
	uint64_t total_bytes_left = file_size;
	uint64_t prev_cut_offset = 0;

	const auto consume_lane_results = [&]() {
		for (auto& lane_result : lane_results) {
			for (const auto& candidate : lane_result) {
				if (candidate.offset <= prev_cut_offset) {
					continue;
				}

				while (candidate.offset - prev_cut_offset >= max_block_size) {
					const uint64_t backup_pos = find_supercdc_backup_cut(
						file_data, prev_cut_offset, avg_block_size, max_block_size, backup_mask);
					if (backup_pos < prev_cut_offset + max_block_size) {
						prev_cut_offset = backup_pos;
						cutpoints.emplace_back(prev_cut_offset);
					}
					else if (candidate.offset != prev_cut_offset + max_block_size) {
						prev_cut_offset += max_block_size;
						cutpoints.emplace_back(prev_cut_offset);
					}
					else break;
				}

				const uint64_t dist_with_prev = candidate.offset - prev_cut_offset;
				const bool candidate_is_eligible =
					(candidate.type == CutPointCandidateType::HARD_CUT_MASK && dist_with_prev >= min_block_size) ||
					(candidate.type == CutPointCandidateType::EASY_CUT_MASK && dist_with_prev >= avg_block_size) ||
					candidate.type == CutPointCandidateType::SUPERCDC_BACKUP_MASK ||
					candidate.type == CutPointCandidateType::MAX_SIZE ||
					candidate.type == CutPointCandidateType::EOF_CUT;
				if (candidate_is_eligible) {
					prev_cut_offset = candidate.offset;
					cutpoints.emplace_back(prev_cut_offset);
				}

			}
			lane_result.clear();
		}
	};

	while (total_bytes_left > std::max<uint64_t>(2ull * LANE_COUNT * GEAR_HASHLEN, min_block_size)) {
		// Ensure the window offsets don't overflow (2GB is the max i32 value, 1GB is good enough)
		uint64_t window_bytes = std::min<uint64_t>(1 << 30, total_bytes_left);
		const uint64_t bytes_per_lane_without_overlap = (((window_bytes - GEAR_HASHLEN) / LANE_COUNT) / GEAR_HASHLEN) * GEAR_HASHLEN;
		const uint64_t bytes_per_lane = bytes_per_lane_without_overlap + GEAR_HASHLEN;
		window_bytes = (bytes_per_lane_without_overlap * LANE_COUNT) + GEAR_HASHLEN;

		const uint64_t expected_candidates_per_lane =
			((bytes_per_lane + avg_block_size - 1) / avg_block_size) + 2;
		for (auto& lane_result : lane_results) {
			lane_result.reserve(expected_candidates_per_lane);
		}

		__m256i vindex = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
		vindex = _mm256_mullo_epi32(vindex, _mm256_set1_epi32(bytes_per_lane_without_overlap));
		const __m256i vindex_end = _mm256_add_epi32(
			vindex,
			_mm256_set1_epi32(static_cast<int32_t>(bytes_per_lane))
		);
		alignas(32) uint32_t lane_end_positions[LANE_COUNT];
		_mm256_store_si256(reinterpret_cast<__m256i*>(lane_end_positions), vindex_end);
		const __m256i all_lanes_mask = _mm256_set1_epi32(-1);
		std::array<uint64_t, LANE_COUNT> lane_candidate_floor{};

		__m256i hash = zero_vec;
		__m256i candidates_hard_vmask = zero_vec;
		__m256i candidates_easy_vmask = zero_vec;
		for (int warmup_iter = 0; warmup_iter < 8; warmup_iter++) {
			__m256i cbytes = _mm256_i32gather_epi32(
				reinterpret_cast<const int*>(file_data + file_data_offset + (4 * warmup_iter)),
				vindex,
				1
			);
			for (uint64_t j = 0; j < sizeof(int32_t); j++) {
				doGearAvx2(hash, cbytes);
			}
		}

		vindex = _mm256_add_epi32(vindex, _mm256_set1_epi32(GEAR_HASHLEN));
		__m256i lanes_at_end = _mm256_cmpeq_epi32(vindex, vindex_end);
		while (_mm256_movemask_epi8(lanes_at_end) != -1) {
			const __m256i active_lanes_mask = _mm256_xor_si256(lanes_at_end, all_lanes_mask);
			const __m256i candidate_high_bit_vec = _mm256_andnot_si256(lanes_at_end, high_bit_vec);
			for (int inner_gather_i = 0; inner_gather_i < 8; inner_gather_i++) {
				__m256i cbytes = _mm256_mask_i32gather_epi32(
					zero_vec,
					reinterpret_cast<const int*>(file_data + file_data_offset + (4 * inner_gather_i)),
					vindex,
					active_lanes_mask,
					1
				);

				for (uint64_t j = 0; j < sizeof(int32_t); j++) {
					doGearAvx2(hash, cbytes);
					const __m256i lane_easy_mask = _mm256_cmpeq_epi32(
						_mm256_and_si256(hash, mm_break_mark_easy),
						zero_vec
					);
					candidates_easy_vmask = _mm256_srli_epi32(candidates_easy_vmask, 1);
					candidates_hard_vmask = _mm256_srli_epi32(candidates_hard_vmask, 1);
					if (_mm256_testz_si256(lane_easy_mask, lane_easy_mask)) {
						continue;
					}
					candidates_easy_vmask = _mm256_or_si256(
						candidates_easy_vmask,
						_mm256_and_si256(lane_easy_mask, candidate_high_bit_vec)
					);
					const __m256i lane_hard_mask = _mm256_cmpeq_epi32(
						_mm256_and_si256(hash, mm_break_mark_hard),
						zero_vec
					);
					candidates_hard_vmask = _mm256_or_si256(
						candidates_hard_vmask,
						_mm256_and_si256(lane_hard_mask, candidate_high_bit_vec)
					);
				}
			}

			if (_mm256_testz_si256(candidates_easy_vmask, candidates_easy_vmask)) {
				candidates_easy_vmask = zero_vec;
				candidates_hard_vmask = zero_vec;
				vindex = _mm256_min_epi32(
					_mm256_add_epi32(vindex, _mm256_set1_epi32(GEAR_HASHLEN)),
					vindex_end
				);
				lanes_at_end = _mm256_cmpeq_epi32(vindex, vindex_end);
				continue;
			}

			alignas(32) uint32_t hard_bitmap_words[LANE_COUNT];
			alignas(32) uint32_t easy_bitmap_words[LANE_COUNT];
			_mm256_store_si256(reinterpret_cast<__m256i*>(hard_bitmap_words), candidates_hard_vmask);
			_mm256_store_si256(reinterpret_cast<__m256i*>(easy_bitmap_words), candidates_easy_vmask);

			alignas(32) uint32_t current_lane_positions[LANE_COUNT];
			alignas(32) uint32_t next_lane_positions[LANE_COUNT];
			_mm256_store_si256(reinterpret_cast<__m256i*>(current_lane_positions), vindex);
			std::copy(
				std::begin(current_lane_positions),
				std::end(current_lane_positions),
				std::begin(next_lane_positions)
			);

			for (uint32_t lane_i = 0; lane_i < LANE_COUNT; lane_i++) {
				if (current_lane_positions[lane_i] == lane_end_positions[lane_i]) {
					continue;
				}

				const uint64_t current_batch_end =
					static_cast<uint64_t>(current_lane_positions[lane_i]) + GEAR_HASHLEN;
				const uint64_t normal_next_pos = std::min<uint64_t>(
					current_batch_end,
					lane_end_positions[lane_i]
				);
				uint64_t desired_next_pos = normal_next_pos;
				bool lane_accepted_candidate = false;

				{
					uint32_t candidates_easy_bits = easy_bitmap_words[lane_i];
					while (candidates_easy_bits != 0) {
						const uint32_t bit = _tzcnt_u32(candidates_easy_bits);
						candidates_easy_bits &= candidates_easy_bits - 1;
						const uint64_t candidate_relative_pos =
							static_cast<uint64_t>(current_lane_positions[lane_i]) + bit;
						if (candidate_relative_pos < lane_candidate_floor[lane_i]) {
							continue;
						}

						const uint64_t candidate_pos = file_data_offset + candidate_relative_pos;
					const auto result_type = ((hard_bitmap_words[lane_i] >> bit) & 1u) != 0
							? CutPointCandidateType::HARD_CUT_MASK
							: CutPointCandidateType::EASY_CUT_MASK;
						if (!lane_achieved_chunk_invariance[lane_i]) {
							if (!lane_results[lane_i].empty()) {
								const auto& prev_cut_candidate = lane_results[lane_i].back();
								const uint64_t dist_with_prev = candidate_pos - prev_cut_candidate.offset;
								const bool is_prev_candidate_hard = prev_cut_candidate.type == CutPointCandidateType::HARD_CUT_MASK;
										if (is_chunk_invariance_condition_satisfied(is_prev_candidate_hard, dist_with_prev, result_type, min_block_size, avg_block_size, max_block_size)) {
									lane_achieved_chunk_invariance[lane_i] = true;
								}
							}
							lane_results[lane_i].push_back({
								result_type,
								candidate_pos
							});
						}
						else {
							uint64_t prev_cut_pos = lane_results[lane_i].empty() ? prev_cut_offset : lane_results[lane_i].back().offset;
							uint64_t dist_with_prev = candidate_pos - prev_cut_pos;

							while (dist_with_prev >= max_block_size) {
								const uint64_t backup_pos = find_supercdc_backup_cut(
									file_data, prev_cut_pos, avg_block_size, max_block_size, backup_mask);
								if (backup_pos < prev_cut_pos + max_block_size) {
									prev_cut_pos = backup_pos;
									lane_results[lane_i].push_back({CutPointCandidateType::SUPERCDC_BACKUP_MASK, prev_cut_pos});
								}
								else {
									prev_cut_pos += max_block_size;
									lane_results[lane_i].push_back({CutPointCandidateType::MAX_SIZE, prev_cut_pos});
								}
								dist_with_prev = candidate_pos - prev_cut_pos;
							}

							if ((result_type == CutPointCandidateType::HARD_CUT_MASK && dist_with_prev >= min_block_size) ||
							(result_type == CutPointCandidateType::EASY_CUT_MASK && dist_with_prev >= avg_block_size)) {
								lane_results[lane_i].push_back({
									result_type,
									candidate_pos
								});
								lane_candidate_floor[lane_i] = candidate_relative_pos + min_block_size;
								uint64_t washout_start =
									lane_candidate_floor[lane_i] >= GEAR_HASHLEN
										? lane_candidate_floor[lane_i] - GEAR_HASHLEN
										: 0;
								// Jump to the previous aligned position for better CPU load performance.
								washout_start &= ~(static_cast<uint64_t>(GEAR_HASHLEN) - 1);
								desired_next_pos = std::max<uint64_t>(
									normal_next_pos,
									washout_start
								);
								lane_accepted_candidate = true;
							}
						}
					}
				}

				uint64_t next_lane_pos = normal_next_pos;
				if (lane_accepted_candidate) {
					next_lane_pos = std::min<uint64_t>(desired_next_pos, lane_end_positions[lane_i]);
				}

				next_lane_positions[lane_i] = static_cast<uint32_t>(next_lane_pos);
			}


			vindex = _mm256_load_si256(reinterpret_cast<const __m256i*>(next_lane_positions));

			candidates_easy_vmask = zero_vec;
			candidates_hard_vmask = zero_vec;
			lanes_at_end = _mm256_cmpeq_epi32(vindex, vindex_end);
		}

		consume_lane_results();
		lane_achieved_chunk_invariance.fill(false);
		lane_achieved_chunk_invariance[0] = true;

		file_data_offset += window_bytes - GEAR_HASHLEN;
		total_bytes_left -= window_bytes - GEAR_HASHLEN;
	}

	file_data_offset = file_data_offset >= GEAR_HASHLEN ? file_data_offset - GEAR_HASHLEN : 0;
	uint32_t pattern = 0;
	for (int j = 0; j < GEAR_HASHLEN && file_data_offset < file_size; j++) {
		pattern = (pattern << 1) + gear_common::table_32[file_data[file_data_offset]];
		file_data_offset++;
	}
	while (file_data_offset < file_size) {
		pattern = (pattern << 1) + gear_common::table_32[file_data[file_data_offset]];
		if (!(pattern & easy_mask)) {
			const auto result_type = !(pattern & hard_mask)
				? CutPointCandidateType::HARD_CUT_MASK
				: CutPointCandidateType::EASY_CUT_MASK;
			lane_results[LANE_COUNT - 1].push_back({result_type, file_data_offset});
		}
		file_data_offset++;
	}

	consume_lane_results();

	// Add as many max size chunks as needed after the last detected cutpoint
	while (file_size - prev_cut_offset > max_block_size) {
		const uint64_t backup_pos = find_supercdc_backup_cut(
			file_data, prev_cut_offset, avg_block_size, max_block_size, backup_mask);
		prev_cut_offset = backup_pos < prev_cut_offset + max_block_size
			? backup_pos : prev_cut_offset + max_block_size;
		cutpoints.emplace_back(prev_cut_offset);
	}
	// Add a final cut for the end of the file
	if (prev_cut_offset < file_size) {
		cutpoints.emplace_back(file_size);
	}
}

// Shared AVX2 CDCZ phase one for scheduled gathers and contiguous lane loads with transpose.
// AVX2 has no scatter instruction, so sparse lane results are stored individually.
// Precondition: cutpoint_bitmap is allocated with at least (file_size / 8) (+1 if not divisible) bytes. (ideally aligned to 32bytes=256bits)
template<bool USE_LOAD_TRANSPOSE>
static void cdcz_chunking_phase_one_avx2_gear_scheduled_impl(
	uint32_t mask,
	uint64_t avg_block_size,
	uint64_t min_block_size,
	uint64_t max_block_size,
	const unsigned char* RESTRICT file_data,
	uint64_t file_size,
	uint8_t* RESTRICT cutpoint_bitmap,
	std::vector<uint64_t>& cutpoints
) {
	static constexpr uint32_t LANE_COUNT = 8;
	(void)cutpoint_bitmap;
	std::array<std::vector<CutPointCandidate>, LANE_COUNT> lane_results{};
	std::array<bool, LANE_COUNT> lane_achieved_chunk_invariance{};
	lane_achieved_chunk_invariance.fill(false);
	lane_achieved_chunk_invariance[0] = true;

	const CdczMasks masks = make_level_one_masks(mask, avg_block_size);
	const uint32_t hard_mask = masks.hard;
	const uint32_t easy_mask = masks.easy;
	const uint32_t backup_mask = masks.backup;
	const __m256i mm_break_mark_hard = _mm256_set1_epi32(hard_mask);
	const __m256i mm_break_mark_easy = _mm256_set1_epi32(easy_mask);
	const __m256i cmask = _mm256_set1_epi32(0xff);
	const __m256i zero_vec = _mm256_setzero_si256();
	const __m256i high_bit_vec = _mm256_set1_epi32(static_cast<int32_t>(1u << 31));
	uint64_t file_data_offset = 0;
	uint64_t total_bytes_left = file_size;
	uint64_t prev_cut_offset = 0;

	const auto consume_lane_results = [&]() {
		for (auto& lane_result : lane_results) {
			for (const auto& candidate : lane_result) {
				if (candidate.offset <= prev_cut_offset) {
					continue;
				}

				while (candidate.offset - prev_cut_offset >= max_block_size) {
					const uint64_t backup_pos = find_supercdc_backup_cut(
						file_data, prev_cut_offset, avg_block_size, max_block_size, backup_mask);
					if (backup_pos < prev_cut_offset + max_block_size) {
						prev_cut_offset = backup_pos;
						cutpoints.emplace_back(prev_cut_offset);
					}
					else if (candidate.offset != prev_cut_offset + max_block_size) {
						prev_cut_offset += max_block_size;
						cutpoints.emplace_back(prev_cut_offset);
					}
					else break;
				}

				const uint64_t dist_with_prev = candidate.offset - prev_cut_offset;
				const bool candidate_is_eligible =
					(candidate.type == CutPointCandidateType::HARD_CUT_MASK && dist_with_prev >= min_block_size) ||
					(candidate.type == CutPointCandidateType::EASY_CUT_MASK && dist_with_prev >= avg_block_size) ||
					candidate.type == CutPointCandidateType::SUPERCDC_BACKUP_MASK ||
					candidate.type == CutPointCandidateType::MAX_SIZE ||
					candidate.type == CutPointCandidateType::EOF_CUT;
				if (candidate_is_eligible) {
					prev_cut_offset = candidate.offset;
					cutpoints.emplace_back(prev_cut_offset);
				}
			}
			lane_result.clear();
		}
	};

	while (total_bytes_left > std::max<uint64_t>(2ull * LANE_COUNT * GEAR_HASHLEN, min_block_size)) {
		// Ensure the window offsets don't overflow (2GB is the max i32 value, 1GB is good enough)
		uint64_t window_bytes = std::min<uint64_t>(1 << 30, total_bytes_left);
		const uint64_t bytes_per_lane_without_overlap = (((window_bytes - GEAR_HASHLEN) / LANE_COUNT) / GEAR_HASHLEN) * GEAR_HASHLEN;
		const uint64_t bytes_per_lane = bytes_per_lane_without_overlap + GEAR_HASHLEN;
		window_bytes = (bytes_per_lane_without_overlap * LANE_COUNT) + GEAR_HASHLEN;

		const uint64_t expected_candidates_per_lane =
			((bytes_per_lane + avg_block_size - 1) / avg_block_size) + 2;
		for (auto& lane_result : lane_results) {
			lane_result.reserve(expected_candidates_per_lane);
		}

		__m256i vindex = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
		vindex = _mm256_mullo_epi32(vindex, _mm256_set1_epi32(bytes_per_lane_without_overlap));
		const __m256i vindex_end = _mm256_add_epi32(
			vindex,
			_mm256_set1_epi32(static_cast<int32_t>(bytes_per_lane))
		);
		alignas(32) uint32_t lane_end_positions[LANE_COUNT];
		_mm256_store_si256(reinterpret_cast<__m256i*>(lane_end_positions), vindex_end);
		const __m256i all_lanes_mask = _mm256_set1_epi32(-1);
		std::array<uint64_t, LANE_COUNT> lane_candidate_floor{};

		__m256i hash = zero_vec;
		__m256i candidates_hard_vmask = zero_vec;
		__m256i candidates_easy_vmask = zero_vec;
		__m256i warmup_cbytes[8]{};
		if constexpr (USE_LOAD_TRANSPOSE) {
			alignas(32) uint32_t lane_positions[LANE_COUNT];
			_mm256_store_si256(reinterpret_cast<__m256i*>(lane_positions), vindex);
			load_and_transpose_8x8_epi32_avx2(
				file_data + file_data_offset,
				lane_positions,
				lane_end_positions,
				warmup_cbytes
			);
		}
		for (int warmup_iter = 0; warmup_iter < 8; warmup_iter++) {
			const __m256i cbytes = USE_LOAD_TRANSPOSE
				? warmup_cbytes[warmup_iter]
				: _mm256_i32gather_epi32(
					reinterpret_cast<const int*>(file_data + file_data_offset + (4 * warmup_iter)),
					vindex,
					1
				);
			__m256i tentry0;
			__m256i tentry1;
			__m256i tentry2;
			__m256i tentry3;
			gather_gear_entries_avx2(cbytes, cmask, tentry0, tentry1, tentry2, tentry3);
			roll_gear_avx2(hash, tentry0);
			roll_gear_avx2(hash, tentry1);
			roll_gear_avx2(hash, tentry2);
			roll_gear_avx2(hash, tentry3);
		}

		vindex = _mm256_add_epi32(vindex, _mm256_set1_epi32(GEAR_HASHLEN));
		__m256i cbytes_by_gather[8]{};
		__m256i tentries_by_gather[8][4]{};
		__m256i hashes_by_gather[8][4]{};
		__m256i lanes_at_end = _mm256_cmpeq_epi32(vindex, vindex_end);
		while (_mm256_movemask_epi8(lanes_at_end) != -1) {
			const __m256i candidate_high_bit_vec = _mm256_andnot_si256(lanes_at_end, high_bit_vec);
			if constexpr (USE_LOAD_TRANSPOSE) {
				alignas(32) uint32_t lane_positions[LANE_COUNT];
				_mm256_store_si256(reinterpret_cast<__m256i*>(lane_positions), vindex);
				load_and_transpose_8x8_epi32_avx2(
					file_data + file_data_offset,
					lane_positions,
					lane_end_positions,
					cbytes_by_gather
				);
			}
			else {
				const __m256i active_lanes_mask = _mm256_xor_si256(lanes_at_end, all_lanes_mask);
				for (int inner_gather_i = 0; inner_gather_i < 8; inner_gather_i++) {
					cbytes_by_gather[inner_gather_i] = _mm256_mask_i32gather_epi32(
						zero_vec,
						reinterpret_cast<const int*>(file_data + file_data_offset + (4 * inner_gather_i)),
						vindex,
						active_lanes_mask,

						1
					);
				}
			}

			for (int inner_gather_i = 0; inner_gather_i < 8; inner_gather_i++) {
				auto& tentry = tentries_by_gather[inner_gather_i];
				gather_gear_entries_avx2(cbytes_by_gather[inner_gather_i], cmask, tentry[0], tentry[1], tentry[2], tentry[3]);
			}

			for (int inner_gather_i = 0; inner_gather_i < 8; inner_gather_i++) {
				roll_gear_avx2(hash, tentries_by_gather[inner_gather_i][0]);
				hashes_by_gather[inner_gather_i][0] = hash;
				roll_gear_avx2(hash, tentries_by_gather[inner_gather_i][1]);
				hashes_by_gather[inner_gather_i][1] = hash;
				roll_gear_avx2(hash, tentries_by_gather[inner_gather_i][2]);
				hashes_by_gather[inner_gather_i][2] = hash;
				roll_gear_avx2(hash, tentries_by_gather[inner_gather_i][3]);
				hashes_by_gather[inner_gather_i][3] = hash;
			}

			for (int inner_gather_i = 0; inner_gather_i < 8; inner_gather_i++) {
				for (uint32_t j = 0; j < 4; j++) {
					const __m256i lane_easy_mask = _mm256_cmpeq_epi32(
						_mm256_and_si256(hashes_by_gather[inner_gather_i][j], mm_break_mark_easy),
						zero_vec
					);
					candidates_easy_vmask = _mm256_srli_epi32(candidates_easy_vmask, 1);
					candidates_hard_vmask = _mm256_srli_epi32(candidates_hard_vmask, 1);
					if (_mm256_testz_si256(lane_easy_mask, lane_easy_mask)) {
						continue;
					}
					candidates_easy_vmask = _mm256_or_si256(
						candidates_easy_vmask,
						_mm256_and_si256(lane_easy_mask, candidate_high_bit_vec)
					);
					const __m256i lane_hard_mask = _mm256_cmpeq_epi32(
						_mm256_and_si256(hashes_by_gather[inner_gather_i][j], mm_break_mark_hard),
						zero_vec
					);
					candidates_hard_vmask = _mm256_or_si256(
						candidates_hard_vmask,
						_mm256_and_si256(lane_hard_mask, candidate_high_bit_vec)
					);
				}
			}

			if (_mm256_testz_si256(candidates_easy_vmask, candidates_easy_vmask)) {
				candidates_easy_vmask = zero_vec;
				candidates_hard_vmask = zero_vec;
				vindex = _mm256_min_epi32(
					_mm256_add_epi32(vindex, _mm256_set1_epi32(GEAR_HASHLEN)),
					vindex_end
				);
				lanes_at_end = _mm256_cmpeq_epi32(vindex, vindex_end);
				continue;
			}

			alignas(32) uint32_t hard_bitmap_words[LANE_COUNT];
			alignas(32) uint32_t easy_bitmap_words[LANE_COUNT];
			_mm256_store_si256(reinterpret_cast<__m256i*>(hard_bitmap_words), candidates_hard_vmask);
			_mm256_store_si256(reinterpret_cast<__m256i*>(easy_bitmap_words), candidates_easy_vmask);

			alignas(32) uint32_t current_lane_positions[LANE_COUNT];
			alignas(32) uint32_t next_lane_positions[LANE_COUNT];
			_mm256_store_si256(reinterpret_cast<__m256i*>(current_lane_positions), vindex);
			std::copy(
				std::begin(current_lane_positions),
				std::end(current_lane_positions),
				std::begin(next_lane_positions)
			);

			for (uint32_t lane_i = 0; lane_i < LANE_COUNT; lane_i++) {
				if (current_lane_positions[lane_i] == lane_end_positions[lane_i]) {
					continue;
				}

				const uint64_t current_batch_end =
					static_cast<uint64_t>(current_lane_positions[lane_i]) + GEAR_HASHLEN;
				const uint64_t normal_next_pos = std::min<uint64_t>(
					current_batch_end,
					lane_end_positions[lane_i]
				);
				uint64_t desired_next_pos = normal_next_pos;
				bool lane_accepted_candidate = false;

				uint32_t candidates_easy_bits = easy_bitmap_words[lane_i];
				while (candidates_easy_bits != 0) {
					const uint32_t bit = _tzcnt_u32(candidates_easy_bits);
					candidates_easy_bits &= candidates_easy_bits - 1;
					const uint64_t candidate_relative_pos =
						static_cast<uint64_t>(current_lane_positions[lane_i]) + bit;
					if (candidate_relative_pos < lane_candidate_floor[lane_i]) {
						continue;
					}

					const uint64_t candidate_pos = file_data_offset + candidate_relative_pos;
					const auto result_type = ((hard_bitmap_words[lane_i] >> bit) & 1u) != 0
							? CutPointCandidateType::HARD_CUT_MASK
							: CutPointCandidateType::EASY_CUT_MASK;
					if (!lane_achieved_chunk_invariance[lane_i]) {
						if (!lane_results[lane_i].empty()) {
							const auto& prev_cut_candidate = lane_results[lane_i].back();
							const uint64_t dist_with_prev = candidate_pos - prev_cut_candidate.offset;
							const bool is_prev_candidate_hard = prev_cut_candidate.type == CutPointCandidateType::HARD_CUT_MASK;
								if (is_chunk_invariance_condition_satisfied(is_prev_candidate_hard, dist_with_prev, result_type, min_block_size, avg_block_size, max_block_size)) {
								lane_achieved_chunk_invariance[lane_i] = true;
							}
						}
						lane_results[lane_i].push_back({result_type, candidate_pos});
					}
					else {
						uint64_t prev_cut_pos = lane_results[lane_i].empty() ? prev_cut_offset : lane_results[lane_i].back().offset;
						uint64_t dist_with_prev = candidate_pos - prev_cut_pos;

						while (dist_with_prev >= max_block_size) {
							const uint64_t backup_pos = find_supercdc_backup_cut(
								file_data, prev_cut_pos, avg_block_size, max_block_size, backup_mask);
							if (backup_pos < prev_cut_pos + max_block_size) {
								prev_cut_pos = backup_pos;
								lane_results[lane_i].push_back({CutPointCandidateType::SUPERCDC_BACKUP_MASK, prev_cut_pos});
							}
							else {
								prev_cut_pos += max_block_size;
								lane_results[lane_i].push_back({CutPointCandidateType::MAX_SIZE, prev_cut_pos});
							}
							dist_with_prev = candidate_pos - prev_cut_pos;
						}

						if ((result_type == CutPointCandidateType::HARD_CUT_MASK && dist_with_prev >= min_block_size) ||
							(result_type == CutPointCandidateType::EASY_CUT_MASK && dist_with_prev >= avg_block_size)) {
							lane_results[lane_i].push_back({result_type, candidate_pos});
							lane_candidate_floor[lane_i] = candidate_relative_pos + min_block_size;
							uint64_t washout_start =
								lane_candidate_floor[lane_i] >= GEAR_HASHLEN
									? lane_candidate_floor[lane_i] - GEAR_HASHLEN
									: 0;
							// Jump to the previous aligned position for better CPU load performance.
							washout_start &= ~(static_cast<uint64_t>(GEAR_HASHLEN) - 1);
							desired_next_pos = std::max<uint64_t>(normal_next_pos, washout_start);
							lane_accepted_candidate = true;
						}
					}
				}

				uint64_t next_lane_pos = normal_next_pos;
				if (lane_accepted_candidate) {
					next_lane_pos = std::min<uint64_t>(desired_next_pos, lane_end_positions[lane_i]);
				}
				next_lane_positions[lane_i] = static_cast<uint32_t>(next_lane_pos);
			}

			vindex = _mm256_load_si256(reinterpret_cast<const __m256i*>(next_lane_positions));
			candidates_easy_vmask = zero_vec;
			candidates_hard_vmask = zero_vec;
			lanes_at_end = _mm256_cmpeq_epi32(vindex, vindex_end);
		}

		consume_lane_results();
		lane_achieved_chunk_invariance.fill(false);
		lane_achieved_chunk_invariance[0] = true;

		file_data_offset += window_bytes - GEAR_HASHLEN;
		total_bytes_left -= window_bytes - GEAR_HASHLEN;
	}

	file_data_offset = file_data_offset >= GEAR_HASHLEN ? file_data_offset - GEAR_HASHLEN : 0;
	uint32_t pattern = 0;
	for (int j = 0; j < GEAR_HASHLEN && file_data_offset < file_size; j++) {
		pattern = (pattern << 1) + gear_common::table_32[file_data[file_data_offset]];
		file_data_offset++;
	}
	while (file_data_offset < file_size) {
		pattern = (pattern << 1) + gear_common::table_32[file_data[file_data_offset]];
		if (!(pattern & easy_mask)) {
			const auto result_type = !(pattern & hard_mask)
				? CutPointCandidateType::HARD_CUT_MASK
				: CutPointCandidateType::EASY_CUT_MASK;
			lane_results[LANE_COUNT - 1].push_back({result_type, file_data_offset});
		}
		file_data_offset++;
	}

	consume_lane_results();

	while (file_size - prev_cut_offset > max_block_size) {
		const uint64_t backup_pos = find_supercdc_backup_cut(
			file_data, prev_cut_offset, avg_block_size, max_block_size, backup_mask);
		prev_cut_offset = backup_pos < prev_cut_offset + max_block_size
			? backup_pos : prev_cut_offset + max_block_size;
		cutpoints.emplace_back(prev_cut_offset);
	}
	if (prev_cut_offset < file_size) {
		cutpoints.emplace_back(file_size);
	}
}

static void cdcz_chunking_phase_one_avx2_gear_with_gather_scheduling(
	uint32_t mask,
	uint64_t avg_block_size,

	uint64_t min_block_size,
	uint64_t max_block_size,
	const unsigned char* RESTRICT file_data,
	uint64_t file_size,
	uint8_t* RESTRICT cutpoint_bitmap,
	std::vector<uint64_t>& cutpoints
) {
	cdcz_chunking_phase_one_avx2_gear_scheduled_impl<false>(
		mask, avg_block_size, min_block_size, max_block_size,
		file_data, file_size, cutpoint_bitmap, cutpoints
	);
}

// AVX2 CDCZ phase one that replaces input gathers with contiguous lane loads and an 8x8 dword transpose.
// Gear-table lookups retain the gather scheduling used by the optimization-level-one implementation.
static void cdcz_chunking_phase_one_avx2_gear_with_load_transpose(
	uint32_t mask,
	uint64_t avg_block_size,
	uint64_t min_block_size,
	uint64_t max_block_size,
	const unsigned char* RESTRICT file_data,
	uint64_t file_size,
	uint8_t* RESTRICT cutpoint_bitmap,
	std::vector<uint64_t>& cutpoints
) {
	cdcz_chunking_phase_one_avx2_gear_scheduled_impl<true>(
		mask, avg_block_size, min_block_size, max_block_size,
		file_data, file_size, cutpoint_bitmap, cutpoints
	);
}
#undef doGearAvx2
#endif

#if defined(__AVX512F__)
static inline void load_and_transpose_16x8_epi32_avx512(
		const unsigned char* RESTRICT file_data,
		const uint32_t (&lane_positions)[16],
		const uint32_t (&lane_end_positions)[16],
		__m512i (&columns)[8]) {
	constexpr __mmask16 LOW_EIGHT_DWORDS = 0x00ff;
	__m512i rows[16];
	for (uint32_t lane_i = 0; lane_i < 16; lane_i++) {
		assert((lane_positions[lane_i] % GEAR_HASHLEN) == 0);
		assert((lane_end_positions[lane_i] % GEAR_HASHLEN) == 0);
		rows[lane_i] = lane_positions[lane_i] == lane_end_positions[lane_i]
			? _mm512_setzero_si512()
			: _mm512_maskz_loadu_epi32(
				LOW_EIGHT_DWORDS,
				file_data + lane_positions[lane_i]);
	}

	const __m512i pair_indices = _mm512_setr_epi32(
		0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23
	);
	const __m512i group4_low_indices = _mm512_setr_epi32(
		0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23
	);
	const __m512i group4_high_indices = _mm512_setr_epi32(
		8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31
	);
	const __m512i group8_low_indices = _mm512_setr_epi32(
		0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23
	);
	const __m512i group8_high_indices = _mm512_setr_epi32(
		8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31
	);
	const __m512i group16_low_indices = _mm512_setr_epi32(
		0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23
	);
	const __m512i group16_high_indices = _mm512_setr_epi32(
		8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31
	);

	__m512i row_pairs[8];
	for (uint32_t pair_i = 0; pair_i < 8; pair_i++) {
		row_pairs[pair_i] = _mm512_permutex2var_epi32(rows[2 * pair_i], pair_indices, rows[2 * pair_i + 1]);
	}

	__m512i row_groups4[8];
	for (uint32_t group_i = 0; group_i < 4; group_i++) {
		row_groups4[2 * group_i] = _mm512_permutex2var_epi32(
			row_pairs[2 * group_i], group4_low_indices, row_pairs[2 * group_i + 1]
		);
		row_groups4[2 * group_i + 1] = _mm512_permutex2var_epi32(
			row_pairs[2 * group_i], group4_high_indices, row_pairs[2 * group_i + 1]
		);
	}

	__m512i row_groups8[8];
	for (uint32_t half_i = 0; half_i < 2; half_i++) {
		const uint32_t input_base = 4 * half_i;
		const uint32_t output_base = 4 * half_i;
		for (uint32_t column_group_i = 0; column_group_i < 2; column_group_i++) {
			const __m512i first_rows = row_groups4[input_base + column_group_i];
			const __m512i second_rows = row_groups4[input_base + 2 + column_group_i];
			row_groups8[output_base + 2 * column_group_i] = _mm512_permutex2var_epi32(
				first_rows, group8_low_indices, second_rows
			);
			row_groups8[output_base + 2 * column_group_i + 1] = _mm512_permutex2var_epi32(
				first_rows, group8_high_indices, second_rows
			);
		}
	}

	for (uint32_t column_pair_i = 0; column_pair_i < 4; column_pair_i++) {
		columns[2 * column_pair_i] = _mm512_permutex2var_epi32(
			row_groups8[column_pair_i], group16_low_indices, row_groups8[4 + column_pair_i]
		);
		columns[2 * column_pair_i + 1] = _mm512_permutex2var_epi32(
			row_groups8[column_pair_i], group16_high_indices, row_groups8[4 + column_pair_i]
		);
	}
}

// AVX-512 CDCZ phase one that replaces input gathers with contiguous lane loads and a 16x8 dword transpose.
// Gear-table lookups retain the gather scheduling used by the optimization-level-one implementation.
static void cdcz_chunking_phase_one_avx512_gear_with_load_transpose(
	uint32_t mask,
	uint64_t avg_block_size,
	uint64_t min_block_size,
	uint64_t max_block_size,
	const unsigned char* RESTRICT file_data,
	uint64_t file_size,
	uint8_t* RESTRICT cutpoint_bitmap,
	std::vector<uint64_t>& cutpoints
) {
	cdcz_chunking_phase_one_avx512_gear_impl<true, true>(
		mask, avg_block_size, min_block_size, max_block_size,
		file_data, file_size, cutpoint_bitmap, cutpoints);
}
#endif

static inline bool is_breakpoint(const uint8_t* cutpoint_bitmap, uint64_t idx) {
	uint8_t b = cutpoint_bitmap[idx / 8];
	return ((b >> (idx % 8)) & 0x1) == 1;
}

#if defined(__AVX512F__)
static void next_chunk_avx512_v1(uint64_t min_block_size, uint64_t max_block_size, uint64_t prev_cut_offset, uint64_t file_size, uint64_t& current_chunk_size, const uint8_t* cutpoint_bitmap) {
	__m512i zero_v = _mm512_set1_epi64(0);
	const uint64_t chunk_size_limit = file_size >= prev_cut_offset + max_block_size ? max_block_size : file_size - prev_cut_offset;
	current_chunk_size = std::min(min_block_size, chunk_size_limit);
	// First align ourselves so the current chunk offset is at the start of a byte on the bitmask
	const uint8_t disaligned_bits = (prev_cut_offset + current_chunk_size) % 8;
	const uint8_t bits_to_alignment = disaligned_bits == 0 ? 0 : 8 - disaligned_bits;
	const uint64_t chunk_size_limit_realigned = std::min(current_chunk_size + bits_to_alignment, chunk_size_limit);
	while (current_chunk_size < chunk_size_limit_realigned) {
		if (is_breakpoint(cutpoint_bitmap, prev_cut_offset + current_chunk_size)) {
			//printf("idx: %lu\n", prev_cut_offset + current_chunk_size);
			break;
		}
		current_chunk_size++;
	}

	// Now that we are aligned on the bitmask we figure out how many 512bit vectors we can fit inside and use them to advance 512bits at a time,
	// at least until an actual cutpoint is found
	const uint64_t chunk_size_limit_multiple512 = current_chunk_size + (((chunk_size_limit - current_chunk_size) / 512u) * 512u);
	while (current_chunk_size < chunk_size_limit_multiple512) {
		__m512i bits = _mm512_loadu_si512(&cutpoint_bitmap[(prev_cut_offset + current_chunk_size) / 8]);
		__mmask8 lane_has_result_bitmask = _mm512_cmpneq_epi64_mask(bits, zero_v);
		if (lane_has_result_bitmask == 0) {
			current_chunk_size += 512ull;
			continue;
		}
		uint8_t lane_i = 0;
		while (((lane_has_result_bitmask >> lane_i) & 0x1) == 0) {
			lane_i++;
		}
		uint8_t bit = 0;
		for (; bit < 64; bit++) {
			if (is_breakpoint(cutpoint_bitmap, prev_cut_offset + current_chunk_size + lane_i * 64ull + bit)) {
				//printf("idx: %lu\n", prev_cut_offset + current_chunk_size);
				break;
			}
		}
		current_chunk_size += lane_i * 64ull + bit;
		break;
	}

	// If we still didn't find a cutpoint we attempt on any leftover bits that didn't fit into a 512bit vector.
	// If we don't find a cutpoint here, we will end up cutting using the max_chunksize.
	if (current_chunk_size == chunk_size_limit_multiple512) {
		while (current_chunk_size < chunk_size_limit) {
			if (is_breakpoint(cutpoint_bitmap, prev_cut_offset + current_chunk_size)) {
				//printf("idx: %lu\n", prev_cut_offset + current_chunk_size);
				break;
			}
			current_chunk_size++;
		}
	}
}
#endif

#if defined(__AVX2__)
static void next_chunk_avx256_v1(uint64_t min_block_size, uint64_t max_block_size, uint64_t prev_cut_offset, uint64_t file_size, uint64_t& current_chunk_size, const uint8_t* cutpoint_bitmap) {
	const __m256i zero_v = _mm256_setzero_si256();
	const uint64_t chunk_size_limit = file_size >= prev_cut_offset + max_block_size ? max_block_size : file_size - prev_cut_offset;
	current_chunk_size = std::min(min_block_size, chunk_size_limit);
	// First align ourselves so the current chunk offset is at the start of a byte on the bitmap.
	const uint8_t disaligned_bits = (prev_cut_offset + current_chunk_size) % 8;

	const uint8_t bits_to_alignment = disaligned_bits == 0 ? 0 : 8 - disaligned_bits;
	const uint64_t chunk_size_limit_realigned = std::min(current_chunk_size + bits_to_alignment, chunk_size_limit);
	while (current_chunk_size < chunk_size_limit_realigned) {
		if (is_breakpoint(cutpoint_bitmap, prev_cut_offset + current_chunk_size)) {
			break;
		}
		current_chunk_size++;
	}

	// Advance 256 source positions at a time until a vector containing a cutpoint is found.
	const uint64_t chunk_size_limit_multiple256 = current_chunk_size + (((chunk_size_limit - current_chunk_size) / 256u) * 256u);
	while (current_chunk_size < chunk_size_limit_multiple256) {
		const __m256i bits = _mm256_loadu_si256(
			reinterpret_cast<const __m256i*>(&cutpoint_bitmap[(prev_cut_offset + current_chunk_size) / 8])
		);
		if (_mm256_testz_si256(bits, bits)) {
			current_chunk_size += 256ull;
			continue;
		}

		const __m256i zero_lanes = _mm256_cmpeq_epi64(bits, zero_v);
		const uint8_t lane_has_result_bitmask = static_cast<uint8_t>(
			(~_mm256_movemask_pd(_mm256_castsi256_pd(zero_lanes))) & 0x0f
		);
		uint8_t lane_i = 0;
		while (((lane_has_result_bitmask >> lane_i) & 0x1) == 0) {
			lane_i++;
		}
		uint8_t bit = 0;
		for (; bit < 64; bit++) {
			if (is_breakpoint(cutpoint_bitmap, prev_cut_offset + current_chunk_size + lane_i * 64ull + bit)) {
				break;
			}
		}
		current_chunk_size += lane_i * 64ull + bit;
		break;
	}

	// Scan any positions that did not fit into a complete 256-bit bitmap vector.
	if (current_chunk_size == chunk_size_limit_multiple256) {
		while (current_chunk_size < chunk_size_limit) {
			if (is_breakpoint(cutpoint_bitmap, prev_cut_offset + current_chunk_size)) {
				break;
			}
			current_chunk_size++;
		}
	}
}
#endif

static void next_chunk_serial(uint64_t min_block_size, uint64_t max_block_size, uint64_t prev_cut_offset, uint64_t file_size, uint64_t& current_chunk_size, const uint8_t* cutpoint_bitmap) {
	// You might argue this implementation is not truly serial as we essentially use 64bit unsigned ints as vectors.
	// To that I say: I don't care. It still uses no SIMD instructions and is much faster, so I think a comparison to an
	// artificially serialized algorithm that would never be used by anyone has no value, because an implementation like this
	// is what would be used if SIMD is not available.

	const uint64_t chunk_size_limit = file_size >= prev_cut_offset + max_block_size ? max_block_size : file_size - prev_cut_offset;
	current_chunk_size = std::min(min_block_size, chunk_size_limit);
	// First align ourselves so the current chunk offset is at the start of a byte on the bitmask
	const uint8_t disaligned_bits = (prev_cut_offset + current_chunk_size) % 8;
	const uint8_t bits_to_alignment = disaligned_bits == 0 ? 0 : 8 - disaligned_bits;
	const uint64_t chunk_size_limit_realigned = std::min(current_chunk_size + bits_to_alignment, chunk_size_limit);
	while (current_chunk_size < chunk_size_limit_realigned) {
		if (is_breakpoint(cutpoint_bitmap, prev_cut_offset + current_chunk_size)) {
			//printf("idx: %lu\n", prev_cut_offset + current_chunk_size);
			break;
		}
		current_chunk_size++;
	}

	// Now that we are aligned on the bitmask we figure out how many 64bit "vectors" we can fit inside and use them to advance 64bits at a time,
	// at least until an actual cutpoint is found
	const uint64_t chunk_size_limit_multiple64 = current_chunk_size + (((chunk_size_limit - current_chunk_size) / 64u) * 64u);
	while (current_chunk_size < chunk_size_limit_multiple64) {
		uint64_t bits = *reinterpret_cast<const uint64_t*>(&cutpoint_bitmap[(prev_cut_offset + current_chunk_size) / 8]);
		if (bits == 0) {
			current_chunk_size += 64;
			continue;
		}
		uint8_t bit = 0;
		for (; bit < 64; bit++) {
			if (is_breakpoint(cutpoint_bitmap, prev_cut_offset + current_chunk_size + bit)) {
				//printf("idx: %lu\n", prev_cut_offset + current_chunk_size);
				break;
			}
		}
		current_chunk_size += bit;
		break;
	}

	// If we still didn't find a cutpoint we attempt on any leftover bits that didn't fit into a 64bit vector.
	// If we don't find a cutpoint here, we will end up cutting using the max_chunksize.
	if (current_chunk_size == chunk_size_limit_multiple64) {
		while (current_chunk_size < chunk_size_limit) {
			if (is_breakpoint(cutpoint_bitmap, prev_cut_offset + current_chunk_size)) {
				//printf("idx: %lu\n", prev_cut_offset + current_chunk_size);
				break;
			}
			current_chunk_size++;
		}
	}
}

static bool next_chunk(
	uint64_t min_block_size,
	uint64_t max_block_size,
	uint64_t file_size,
	chunk_boundary& cb,
	SIMD_Mode simd_mode,
	const uint8_t* cutpoint_bitmap,
	std::vector<uint64_t>& cutpoints,
	uint64_t cutpoints_i
) {
	uint64_t current_chunk_size = 0;
	uint64_t prev_cut_offset = cb.right;

	if (!cutpoints.empty()) {
		if (cutpoints.size() <= cutpoints_i) {
			return false;
		}
		cb.left = prev_cut_offset;
		cb.right = cutpoints[cutpoints_i];
		return true;
	}

	if (file_size <= prev_cut_offset)
		return false;

	switch (simd_mode) {
		case SIMD_Mode::NONE:
			next_chunk_serial(min_block_size, max_block_size, prev_cut_offset, file_size, current_chunk_size, cutpoint_bitmap);
			break;
		case SIMD_Mode::AVX256:
		#if defined(__AVX2__)
			next_chunk_avx256_v1(min_block_size, max_block_size, prev_cut_offset, file_size, current_chunk_size, cutpoint_bitmap);
			break;
		#else
			std::cerr << "AVX256 chunk scanning requested, but AVX2 support is not compiled in" << std::endl;
			exit(EXIT_FAILURE);
		#endif
		case SIMD_Mode::AVX512:
		#if defined(__AVX512F__)
			next_chunk_avx512_v1(min_block_size, max_block_size, prev_cut_offset, file_size, current_chunk_size, cutpoint_bitmap);
			break;
		#else
			std::cerr << "AVX512 chunk scanning requested, but AVX-512 support is not compiled in" << std::endl;
			exit(EXIT_FAILURE);
		#endif
		default:
			std::cerr << "Unsupported SIMD mode requested for SS-CDC chunk scanning" << std::endl;
			exit(EXIT_FAILURE);
	}

	cb.left = prev_cut_offset;
	cb.right = prev_cut_offset + current_chunk_size;
	return true;
}

std::vector<std::string> Cdcz_Chunking::chunk_file(std::string file_path) {
	if (simd_mode == SIMD_Mode::NONE) {
		return Chunking_Technique::chunk_file(file_path);
	}

	using phase_one_function = void (*)(uint32_t, uint64_t, uint64_t, uint64_t, const unsigned char*, uint64_t, uint8_t*, std::vector<uint64_t>&);
	phase_one_function phase_one = nullptr;
	switch (simd_mode) {
	#if defined(__AVX2__)
		case SIMD_Mode::AVX256: {
			switch (optimization_level) {
				case 1:
					phase_one = cdcz_chunking_phase_one_avx2_gear_with_gather_scheduling;
					break;
				case 2:
					phase_one = cdcz_chunking_phase_one_avx2_gear_with_load_transpose;
					break;
				default:
					phase_one = cdcz_chunking_phase_one_avx2_gear;
					break;
			}
			break;
		}
	#endif
	#if defined(__AVX512F__)
		case SIMD_Mode::AVX512: {
			switch (optimization_level) {
				case 1:
					phase_one = cdcz_chunking_phase_one_avx512_gear_with_gather_scheduling;
					break;
				case 2:
					phase_one = cdcz_chunking_phase_one_avx512_gear_with_load_transpose;
					break;
				default:
					phase_one = cdcz_chunking_phase_one_avx512_gear;
					break;
			}
			break;
		}
	#endif
		default:
			std::cout << "SIMD mode unsupported for chosen chunking technique" << std::endl;

			exit(1);
	}

	std::ifstream file_ptr;
	file_ptr.open(file_path, std::ios::in | std::ios::binary);
	if (!file_ptr.is_open()) {
		throw std::runtime_error("Failed to open file: " + file_path);
	}

	file_ptr.seekg(0, std::ios_base::end);
	const std::streampos end_position = file_ptr.tellg();
	if (end_position < 0) {
		throw std::runtime_error("Failed to determine file size: " + file_path);
	}
	file_size = static_cast<uint64_t>(end_position);
	file_ptr.seekg(0, std::ios_base::beg);
	if (!file_ptr) {
		throw std::runtime_error("Failed to seek file: " + file_path);
	}

	const auto bitmap_size = (file_size + 7) / 8 + HASHLEN / 8;
	if (cutpoint_bitmap) {
		portable_aligned_free(cutpoint_bitmap);
	}
	cutpoint_bitmap = static_cast<uint8_t*>(portable_aligned_alloc(64, bitmap_size));
	if (!cutpoint_bitmap) {
		printf("Error: allocate breakpoint bitmap failed, exit\n");
		exit(2);
	}
	std::memset(cutpoint_bitmap, 0, bitmap_size);

	std::vector<uint64_t> cutpoints{};

	using aligned_buffer = std::unique_ptr<unsigned char, decltype(&portable_aligned_free)>;
	aligned_buffer file_data_owner(
		static_cast<unsigned char*>(portable_aligned_alloc(64, file_size)),
		&portable_aligned_free
	);
	if (!file_data_owner) {
		printf("Error: allocate aligned file buffer failed, exit\n");
		exit(2);
	}
	unsigned char* const file_data = file_data_owner.get();
	if (!read_exact(file_ptr, reinterpret_cast<char*>(file_data), file_size)) {
		throw std::runtime_error("Failed to read complete file: " + file_path);
	}

	auto begin_chunking = std::chrono::high_resolution_clock::now();
	// mask was made for 64bits on the most significant bits, shift and cast to 32bit
	const uint32_t avg_mask = static_cast<uint32_t>(mask >> 32);
#ifndef NDEBUG
	const CdczMasks debug_masks = make_level_one_masks(avg_mask, avg_block_size);
	const uint32_t hard_mask = debug_masks.hard;
	const uint32_t easy_mask = debug_masks.easy;
	const uint32_t backup_mask = debug_masks.backup;
#endif
	phase_one(avg_mask, avg_block_size, min_block_size, max_block_size, file_data, file_size, cutpoint_bitmap, cutpoints);
	auto end_chunking = std::chrono::high_resolution_clock::now();
	total_time_chunking += (end_chunking - begin_chunking);

	std::memset(&cb, 0, sizeof(chunk_boundary));
	std::vector<std::string> hashes{};
	uint64_t prev_cut_offset = cb.right;
	uint64_t cutpoints_i = 0;
	while (next_chunk(min_block_size, max_block_size, file_size, cb, SIMD_Mode::NONE, cutpoint_bitmap, cutpoints, cutpoints_i)) {
		cutpoints_i++;
#ifndef NDEBUG
		// DEBUG: serially check next chunk size, crash if there is a mismatch
		auto debug_curr_pos = prev_cut_offset;
		uint64_t debug_backup_pos = std::numeric_limits<uint64_t>::max();
		uint32_t hash = 0;
		for (; debug_curr_pos < file_size; debug_curr_pos++) {
			hash = (hash << 1) + gear_common::table_32[file_data[debug_curr_pos]];
			const uint64_t debug_chunk_size = debug_curr_pos - prev_cut_offset;
			if (debug_chunk_size >= avg_block_size && debug_chunk_size < max_block_size &&
				debug_backup_pos == std::numeric_limits<uint64_t>::max() && !(hash & backup_mask)) debug_backup_pos = debug_curr_pos;
			const bool normalized_match =
				(debug_chunk_size >= min_block_size && debug_chunk_size < avg_block_size && !(hash & hard_mask)) ||
				(debug_chunk_size >= avg_block_size && !(hash & easy_mask));
			if (debug_chunk_size == max_block_size) {
				if (debug_backup_pos != std::numeric_limits<uint64_t>::max()) debug_curr_pos = debug_backup_pos;
				break;
			}
			if (normalized_match) break;
		}
		uint64_t debug_chunk_size = debug_curr_pos - prev_cut_offset;
#endif

		uint64_t chunk_size = cb.right - prev_cut_offset;

#ifndef NDEBUG
		if (chunk_size != debug_chunk_size) {
			fprintf(stderr, "FIXING BAD CHUNK WITH SERIAL VERSION!\n");
			chunk_size = debug_chunk_size;
			cb.right = debug_curr_pos;
		}
#endif

		File_Chunk new_chunk{ chunk_size };
		memcpy(new_chunk.get_data(), file_data + prev_cut_offset, chunk_size);
		if (!disable_hashing) {
			auto begin_hashing = std::chrono::high_resolution_clock::now();
			hash_method->hash_chunk(new_chunk);
			auto end_hashing = std::chrono::high_resolution_clock::now();
			total_time_hashing += (end_hashing - begin_hashing);
		}
		hashes.emplace_back(new_chunk.to_string());

		prev_cut_offset = cb.right;
	}
	return hashes;
}

uint64_t Cdcz_Chunking::find_cutpoint(char* data, uint64_t size) {
    if (simd_mode == SIMD_Mode::NONE)
      return find_cutpoint_serial(data, size);
	else if (simd_mode == SIMD_Mode::AVX256 || simd_mode == SIMD_Mode::AVX512) {
		uint64_t prev_cut_offset = cb.right;
		std::vector<uint64_t> cutpoints;
		next_chunk(min_block_size, max_block_size, file_size, cb, simd_mode, cutpoint_bitmap, cutpoints, 0);
		return cb.right - prev_cut_offset;
		//return find_cutpoint_avx512(data, size);
	}
    else {
      std::cout << "SIMD mode unsupported for chosen chunking technique" << std::endl;
      exit(1);
    }
}

uint64_t Cdcz_Chunking::find_cutpoint_serial(char* data, uint64_t size) {
  uint32_t hash = 0;
  uint64_t idx = min_block_size;
  const uint32_t mask_32 = static_cast<uint32_t>(mask >> 32);

  // If given data is lower than the minimum chunk size, return data length.
  if (size <= min_block_size) {
    return size;
  }

  while (idx < size && idx < max_block_size) {
    hash = gear_common::roll(hash, static_cast<uint8_t>(data[idx]));
    if (!(hash & mask_32)) {
      return idx;
    }
    idx += 1;
  }

  return idx;
}

uint64_t Cdcz_Chunking::find_cutpoint_avx512(char* data, uint64_t size) {
  return find_cutpoint_serial(data, size);
}
