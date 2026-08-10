
/**
 * @file gear_chunking.cpp
 * @author WASL
 * @brief Implementations for gear chunking technique
 * @version 0.1
 * @date 2023-4-1
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "gear_chunking.hpp"

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

Gear_Chunking::Gear_Chunking(const Config& config) {
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
	use_low_bit_mask = config.get_gear_use_low_bit_mask();
	if (simd_mode != SIMD_Mode::NONE && use_64bit_gear) {
		throw ConfigError("64-bit Gear is unsupported for SSCDC");
	}
	if (simd_mode != SIMD_Mode::NONE && use_low_bit_mask) {
		throw ConfigError(
			"Low-bit Gear masks are always enabled on SIMD modes");
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

// Precondition: cutpoint_bitmap is allocated with at least (file_size / 8) (+1 if not divisible) bytes
// Serial version of SS-CDC phase one to have a simpler, slower but correct implementation for reference
static void sscdc_chunking_phase_one_serial_gear(uint32_t mask, uint64_t min_block_size, unsigned char* RESTRICT file_data, uint64_t file_size, uint8_t* RESTRICT cutpoint_bitmap) {
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

#if defined(__AVX512F__)
#define doGearAvx512(hash, cbytes) {\
					hash = _mm512_slli_epi32(hash, 1);\
					__m512i idx = _mm512_and_epi32(cbytes, cmask);\
					cbytes = _mm512_srli_epi32(cbytes, 8);\
					__m512i tentry = _mm512_i32gather_epi32(idx, gear_common::table_32, 4);\
					hash = _mm512_add_epi32(hash, tentry);}

// Code adapted from NetApp's SSCDC repo
// Precondition: cutpoint_bitmap is allocated with at least (file_size / 8) (+1 if not divisible) bytes (ideally aligned to 64bytes=512bits)
static void sscdc_chunking_phase_one_avx512_gear(uint32_t mask, uint64_t min_block_size, const unsigned char* RESTRICT file_data, uint64_t file_size, uint8_t* RESTRICT cutpoint_bitmap) {
	static constexpr uint32_t LANE_COUNT = 16;
	const __m512i mm_break_mark = _mm512_set1_epi32(mask);
	const __m512i cmask = _mm512_set1_epi32(0xff);
	uint64_t file_data_offset = 0;
	uint64_t total_bytes_left = file_size;

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

		__m512i vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
		// bytes_per_lane_without_overlap used for each lane's start pos, lanes overlap at the end of their data into next lane's data.
		// This applies even for the last lane, the next iteration will start at the start of the last lane's overlap data.
		vindex = _mm512_mullo_epi32(vindex, _mm512_set1_epi32(bytes_per_lane_without_overlap));

		const auto zero_vec = _mm512_set1_epi32(0);

		__m512i hash = zero_vec;
		__m512i cutpoint_bitmap_vmask = zero_vec;
		// We know exactly how many Gathers we will need to process all data in the window as each Gather gets 32bits/4bytes of data
		// per lane, and bytes_per_lane is divisible by GEAR_HASHLEN=8 Gather instructions.

		const unsigned int gather_count = (bytes_per_lane / GEAR_HASHLEN) * 8;
		// First do the "warmup" so each lane gets the valid GEAR hash pattern after the overlap with the previous lane's data
		// ( for the first lane on the first segment we assume the minimum chunk size >= GEAR_HASHLEN )
		for (int warmup_iter = 0; warmup_iter < 8; warmup_iter++) {
			__m512i cbytes = _mm512_i32gather_epi32(vindex, file_data + file_data_offset + (4 * warmup_iter), 1);
			for (uint64_t j = 0; j < sizeof(int32_t); j++) {
				doGearAvx512(hash, cbytes);
			}
		}
		unsigned int gather_i = 8;
		vindex = _mm512_add_epi32(vindex, _mm512_set1_epi32(32));
		while (gather_i < gather_count) {
			for (int inner_gather_i = 0; inner_gather_i < 8; inner_gather_i++) {
				//printf("Processing data stream at [0X%x-0X%x-1]: 0x%x bytes/thread parallel\n", offset, offset+cur_segsize*LANE_COUNT, cur_segsize);
				__m512i cbytes = _mm512_i32gather_epi32(vindex, file_data + file_data_offset + (4 * inner_gather_i), 1);

				for (uint64_t j = 0; j < sizeof(int32_t); j++) {
					doGearAvx512(hash, cbytes);

					//__m512i cut_judgment_vmask = _mm512_and_epi32(hash, mm_break_mark);
					//__mmask16 lane_cutpoint_mask = _mm512_cmpeq_epi32_mask(cut_judgment_vmask, zero_vec);
					const __mmask16 lane_cutpoint_mask = _mm512_testn_epi32_mask(hash, mm_break_mark);
					cutpoint_bitmap_vmask = _mm512_srli_epi32(cutpoint_bitmap_vmask, 1);
					if (lane_cutpoint_mask > 0) {
						const __m512i ret = _mm512_maskz_set1_epi32(lane_cutpoint_mask, static_cast<int32_t>(1u << 31));
						cutpoint_bitmap_vmask = _mm512_or_epi32(cutpoint_bitmap_vmask, ret);
					}
				}
			}

			// Only scatter results if there is any result to scatter (avoid expensive scatter if unneeded).
			// Also do masked scatter to prevent scattering on lanes without results.
			const __mmask16 lanes_with_results_mask = _mm512_cmpneq_epi32_mask(cutpoint_bitmap_vmask, zero_vec);
			if (lanes_with_results_mask != 0) {
				const auto cutpoint_bitmap_vindex = _mm512_srli_epi32(vindex, 3);  // vindex / 8
				const auto cutpoint_bitmap_offset = file_data_offset >> 3;  // idem
				_mm512_mask_i32scatter_epi32(
					cutpoint_bitmap + cutpoint_bitmap_offset,
					lanes_with_results_mask,
					cutpoint_bitmap_vindex,
					cutpoint_bitmap_vmask,
					1
				);
			}

			cutpoint_bitmap_vmask = zero_vec;
			vindex = _mm512_add_epi32(vindex, _mm512_set1_epi32(32));
			gather_i += 8;
		}

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
		if (!(pattern & mask)) {
			const uint32_t byte_pos = file_data_offset / 8u;
			const uint8_t bit_pos = file_data_offset % 8u;
			cutpoint_bitmap[byte_pos] = cutpoint_bitmap[byte_pos] | static_cast<uint8_t>(1u << bit_pos);
		}
		file_data_offset++;
	}
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

// Code adapted from NetApp's SSCDC repo
// Precondition: cutpoint_bitmap is allocated with at least (file_size / 8) (+1 if not divisible) bytes (ideally aligned to 64bytes=512bits)
static void sscdc_chunking_phase_one_avx512_gear_with_gather_scheduling(uint32_t mask, uint64_t min_block_size, const unsigned char* RESTRICT file_data, uint64_t file_size, uint8_t* RESTRICT cutpoint_bitmap) {
	static constexpr uint32_t LANE_COUNT = 16;
	const __m512i mm_break_mark = _mm512_set1_epi32(mask);
	const __m512i cmask = _mm512_set1_epi32(0xff);
	uint64_t file_data_offset = 0;
	uint64_t total_bytes_left = file_size;

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

		__m512i vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
		// bytes_per_lane_without_overlap used for each lane's start pos, lanes overlap at the end of their data into next lane's data.
		// This applies even for the last lane, the next iteration will start at the start of the last lane's overlap data.
		vindex = _mm512_mullo_epi32(vindex, _mm512_set1_epi32(bytes_per_lane_without_overlap));

		const auto zero_vec = _mm512_set1_epi32(0);

		__m512i hash = zero_vec;
		__m512i cutpoint_bitmap_vmask = zero_vec;
		// We know exactly how many Gathers we will need to process all data in the window as each Gather gets 32bits/4bytes of data
		// per lane, and bytes_per_lane is divisible by GEAR_HASHLEN=8 Gather instructions.
		const unsigned int gather_count = (bytes_per_lane / GEAR_HASHLEN) * 8;
		// First do the "warmup" so each lane gets the valid GEAR hash pattern after the overlap with the previous lane's data
		// ( for the first lane on the first segment we assume the minimum chunk size >= GEAR_HASHLEN )
		for (int warmup_iter = 0; warmup_iter < 8; warmup_iter++) {
			const __m512i cbytes = _mm512_i32gather_epi32(vindex, file_data + file_data_offset + (4 * warmup_iter), 1);
			__m512i tentry0;
			__m512i tentry1;
			__m512i tentry2;
			__m512i tentry3;
			gather_gear_entries_avx512(cbytes, cmask, tentry0, tentry1, tentry2, tentry3);
			roll_gear_avx512(hash, tentry0);
			roll_gear_avx512(hash, tentry1);
			roll_gear_avx512(hash, tentry2);
			roll_gear_avx512(hash, tentry3);
		}
		unsigned int gather_i = 8;
		vindex = _mm512_add_epi32(vindex, _mm512_set1_epi32(32));
		__m512i cbytes_by_gather[8]{};
		__m512i tentries_by_gather[8][4]{};
		__m512i hashes_by_gather[8][4]{};
		while (gather_i < gather_count) {
			// We will execute batches of 8 x 4 bytes, but scheduling the Gathers for better performance
			for (int inner_gather_i = 0; inner_gather_i < 8; inner_gather_i++) {
				cbytes_by_gather[inner_gather_i] = _mm512_i32gather_epi32(vindex, file_data + file_data_offset + (4 * inner_gather_i), 1);
			}

			for (int inner_gather_i = 0; inner_gather_i < 8; inner_gather_i++) {
				auto& tentry = tentries_by_gather[inner_gather_i];
				gather_gear_entries_avx512(cbytes_by_gather[inner_gather_i], cmask, tentry[0], tentry[1], tentry[2], tentry[3]);
			}

			for (int inner_gather_i = 0; inner_gather_i < 8; inner_gather_i++) {
				//printf("Processing data stream at [0X%x-0X%x-1]: 0x%x bytes/thread parallel\n", offset, offset+cur_segsize*LANE_COUNT, cur_segsize);
				roll_gear_avx512(hash, tentries_by_gather[inner_gather_i][0]);
				hashes_by_gather[inner_gather_i][0] = hash;
				roll_gear_avx512(hash, tentries_by_gather[inner_gather_i][1]);
				hashes_by_gather[inner_gather_i][1] = hash;
				roll_gear_avx512(hash, tentries_by_gather[inner_gather_i][2]);
				hashes_by_gather[inner_gather_i][2] = hash;
				roll_gear_avx512(hash, tentries_by_gather[inner_gather_i][3]);
				hashes_by_gather[inner_gather_i][3] = hash;
			}

			for (int inner_gather_i = 0; inner_gather_i < 8; inner_gather_i++) {
				record_cutpoints_avx512(hashes_by_gather[inner_gather_i][0], mm_break_mark, cutpoint_bitmap_vmask);
				record_cutpoints_avx512(hashes_by_gather[inner_gather_i][1], mm_break_mark, cutpoint_bitmap_vmask);
				record_cutpoints_avx512(hashes_by_gather[inner_gather_i][2], mm_break_mark, cutpoint_bitmap_vmask);
				record_cutpoints_avx512(hashes_by_gather[inner_gather_i][3], mm_break_mark, cutpoint_bitmap_vmask);
			}

			// Only scatter results if there is any result to scatter (avoid expensive scatter if unneeded).
			// Also do masked scatter to prevent scattering on lanes without results.
			const __mmask16 lanes_with_results_mask = _mm512_cmpneq_epi32_mask(cutpoint_bitmap_vmask, zero_vec);
			if (lanes_with_results_mask != 0) {
				const auto cutpoint_bitmap_vindex = _mm512_srli_epi32(vindex, 3);  // vindex / 8
				const auto cutpoint_bitmap_offset = file_data_offset >> 3;  // idem
				_mm512_mask_i32scatter_epi32(
					cutpoint_bitmap + cutpoint_bitmap_offset,
					lanes_with_results_mask,

					cutpoint_bitmap_vindex,
					cutpoint_bitmap_vmask,
					1
				);
			}

			cutpoint_bitmap_vmask = zero_vec;
			vindex = _mm512_add_epi32(vindex, _mm512_set1_epi32(32));
			gather_i += 8;
		}

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
		if (!(pattern & mask)) {
			const uint32_t byte_pos = file_data_offset / 8u;
			const uint8_t bit_pos = file_data_offset % 8u;
			cutpoint_bitmap[byte_pos] = cutpoint_bitmap[byte_pos] | static_cast<uint8_t>(1u << bit_pos);
		}
		file_data_offset++;
	}
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
		uint64_t bytes_per_lane,
		__m256i (&columns)[8]) {
	assert((reinterpret_cast<std::uintptr_t>(file_data) % 32) == 0);
	assert((bytes_per_lane % 32) == 0);
	__m256i rows[8];
	for (uint32_t lane_i = 0; lane_i < 8; lane_i++) {
		rows[lane_i] = _mm256_load_si256(reinterpret_cast<const __m256i*>(file_data + lane_i * bytes_per_lane));
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
static void sscdc_chunking_phase_one_avx2_gear(uint32_t mask, uint64_t min_block_size, const unsigned char* RESTRICT file_data, uint64_t file_size, uint8_t* RESTRICT cutpoint_bitmap) {
	static constexpr uint32_t LANE_COUNT = 8;
	const __m256i mm_break_mark = _mm256_set1_epi32(mask);
	const __m256i cmask = _mm256_set1_epi32(0xff);
	const __m256i zero_vec = _mm256_setzero_si256();
	uint64_t file_data_offset = 0;
	uint64_t total_bytes_left = file_size;

	while (total_bytes_left > std::max<uint64_t>(2ull * LANE_COUNT * GEAR_HASHLEN, min_block_size)) {
		// Ensure the window offsets don't overflow (2GB is the max i32 value, 1GB is good enough)
		uint64_t window_bytes = std::min<uint64_t>(1 << 30, total_bytes_left);
		const uint64_t bytes_per_lane_without_overlap = (((window_bytes - GEAR_HASHLEN) / LANE_COUNT) / GEAR_HASHLEN) * GEAR_HASHLEN;
		const uint64_t bytes_per_lane = bytes_per_lane_without_overlap + GEAR_HASHLEN;
		window_bytes = (bytes_per_lane_without_overlap * LANE_COUNT) + GEAR_HASHLEN;

		__m256i vindex = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
		vindex = _mm256_mullo_epi32(vindex, _mm256_set1_epi32(bytes_per_lane_without_overlap));

		__m256i hash = zero_vec;
		const unsigned int gather_count = (bytes_per_lane / GEAR_HASHLEN) * 8;
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

		unsigned int gather_i = 8;
		vindex = _mm256_add_epi32(vindex, _mm256_set1_epi32(GEAR_HASHLEN));
		while (gather_i < gather_count) {
			for (int inner_gather_i = 0; inner_gather_i < 8; inner_gather_i++) {
				__m256i cbytes = _mm256_i32gather_epi32(
					reinterpret_cast<const int*>(file_data + file_data_offset + (4 * inner_gather_i)),
					vindex,
					1
				);

				for (uint64_t j = 0; j < sizeof(int32_t); j++) {
					doGearAvx2(hash, cbytes);
					const __m256i lane_cutpoint_mask = _mm256_cmpeq_epi32(
						_mm256_and_si256(hash, mm_break_mark),
						zero_vec
					);
					uint32_t candidate_lanes = static_cast<uint32_t>(
						_mm256_movemask_ps(_mm256_castsi256_ps(lane_cutpoint_mask))
					);
					while (candidate_lanes != 0) {
						const uint32_t lane_i = _tzcnt_u32(candidate_lanes);
						const uint64_t candidate_pos = file_data_offset
							+ lane_i * bytes_per_lane_without_overlap
							+ gather_i * sizeof(int32_t)
							+ inner_gather_i * sizeof(int32_t)
							+ j;
						cutpoint_bitmap[candidate_pos >> 3] |= static_cast<uint8_t>(1u << (candidate_pos & 7));
						candidate_lanes &= candidate_lanes - 1;
					}
				}
			}
			vindex = _mm256_add_epi32(vindex, _mm256_set1_epi32(GEAR_HASHLEN));
			gather_i += 8;
		}

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
		if (!(pattern & mask)) {
			const uint32_t byte_pos = file_data_offset / 8u;
			const uint8_t bit_pos = file_data_offset % 8u;
			cutpoint_bitmap[byte_pos] = cutpoint_bitmap[byte_pos] | static_cast<uint8_t>(1u << bit_pos);
		}
		file_data_offset++;
	}
}

// AVX2 version of SS-CDC phase one with gathers scheduled in batches to expose more instruction-level parallelism.
// AVX2 has no scatter instruction, so sparse lane results are stored individually.
// Precondition: cutpoint_bitmap is allocated with at least (file_size / 8) (+1 if not divisible) bytes. (ideally aligned to 32bytes=256bits)
static void sscdc_chunking_phase_one_avx2_gear_with_gather_scheduling(uint32_t mask, uint64_t min_block_size, const unsigned char* RESTRICT file_data, uint64_t file_size, uint8_t* RESTRICT cutpoint_bitmap) {
	static constexpr uint32_t LANE_COUNT = 8;
	const __m256i mm_break_mark = _mm256_set1_epi32(mask);
	const __m256i cmask = _mm256_set1_epi32(0xff);
	const __m256i zero_vec = _mm256_setzero_si256();
	const __m256i high_bit_vec = _mm256_set1_epi32(static_cast<int32_t>(1u << 31));
	uint64_t file_data_offset = 0;
	uint64_t total_bytes_left = file_size;

	while (total_bytes_left > std::max<uint64_t>(2ull * LANE_COUNT * GEAR_HASHLEN, min_block_size)) {
		// Ensure the window offsets don't overflow (2GB is the max i32 value, 1GB is good enough)
		uint64_t window_bytes = std::min<uint64_t>(1 << 30, total_bytes_left);
		const uint64_t bytes_per_lane_without_overlap = (((window_bytes - GEAR_HASHLEN) / LANE_COUNT) / GEAR_HASHLEN) * GEAR_HASHLEN;
		const uint64_t bytes_per_lane = bytes_per_lane_without_overlap + GEAR_HASHLEN;
		window_bytes = (bytes_per_lane_without_overlap * LANE_COUNT) + GEAR_HASHLEN;

		__m256i vindex = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
		vindex = _mm256_mullo_epi32(vindex, _mm256_set1_epi32(bytes_per_lane_without_overlap));

		__m256i hash = zero_vec;
		__m256i cutpoint_bitmap_vmask = zero_vec;
		const unsigned int gather_count = (bytes_per_lane / GEAR_HASHLEN) * 8;
		for (int warmup_iter = 0; warmup_iter < 8; warmup_iter++) {
			const __m256i cbytes = _mm256_i32gather_epi32(
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

		unsigned int gather_i = 8;
		vindex = _mm256_add_epi32(vindex, _mm256_set1_epi32(GEAR_HASHLEN));
		__m256i cbytes_by_gather[8]{};
		__m256i tentries_by_gather[8][4]{};
		__m256i hashes_by_gather[8][4]{};
		while (gather_i < gather_count) {
			for (int inner_gather_i = 0; inner_gather_i < 8; inner_gather_i++) {
				cbytes_by_gather[inner_gather_i] = _mm256_i32gather_epi32(
					reinterpret_cast<const int*>(file_data + file_data_offset + (4 * inner_gather_i)),
					vindex,
					1
				);
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
				record_cutpoints_avx2(hashes_by_gather[inner_gather_i][0], mm_break_mark, zero_vec, high_bit_vec, cutpoint_bitmap_vmask);
				record_cutpoints_avx2(hashes_by_gather[inner_gather_i][1], mm_break_mark, zero_vec, high_bit_vec, cutpoint_bitmap_vmask);
				record_cutpoints_avx2(hashes_by_gather[inner_gather_i][2], mm_break_mark, zero_vec, high_bit_vec, cutpoint_bitmap_vmask);
				record_cutpoints_avx2(hashes_by_gather[inner_gather_i][3], mm_break_mark, zero_vec, high_bit_vec, cutpoint_bitmap_vmask);
			}

			if (!_mm256_testz_si256(cutpoint_bitmap_vmask, cutpoint_bitmap_vmask)) {
				alignas(32) uint32_t bitmap_words[LANE_COUNT];
				_mm256_store_si256(reinterpret_cast<__m256i*>(bitmap_words), cutpoint_bitmap_vmask);
				const uint64_t bitmap_batch_offset =
					(file_data_offset + gather_i * sizeof(int32_t)) >> 3;
				const uint64_t bitmap_bytes_per_lane = bytes_per_lane_without_overlap >> 3;
				for (uint32_t lane_i = 0; lane_i < LANE_COUNT; lane_i++) {
					const uint64_t bitmap_offset = bitmap_batch_offset + lane_i * bitmap_bytes_per_lane;
					std::memcpy(cutpoint_bitmap + bitmap_offset, &bitmap_words[lane_i], sizeof(uint32_t));
				}
			}

			cutpoint_bitmap_vmask = zero_vec;
			vindex = _mm256_add_epi32(vindex, _mm256_set1_epi32(GEAR_HASHLEN));
			gather_i += 8;
		}

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
		if (!(pattern & mask)) {
			const uint32_t byte_pos = file_data_offset / 8u;
			const uint8_t bit_pos = file_data_offset % 8u;
			cutpoint_bitmap[byte_pos] = cutpoint_bitmap[byte_pos] | static_cast<uint8_t>(1u << bit_pos);
		}
		file_data_offset++;
	}
}

// AVX2 version of SS-CDC phase one that replaces input gathers with contiguous lane loads and an 8x8 dword transpose.
// Gear-table lookups retain the gather scheduling used by the optimization-level-one implementation.
static void sscdc_chunking_phase_one_avx2_gear_with_load_transpose(uint32_t mask, uint64_t min_block_size, const unsigned char* RESTRICT file_data, uint64_t file_size, uint8_t* RESTRICT cutpoint_bitmap) {
	static constexpr uint32_t LANE_COUNT = 8;
	const __m256i mm_break_mark = _mm256_set1_epi32(mask);
	const __m256i cmask = _mm256_set1_epi32(0xff);
	const __m256i zero_vec = _mm256_setzero_si256();
	const __m256i high_bit_vec = _mm256_set1_epi32(static_cast<int32_t>(1u << 31));
	uint64_t file_data_offset = 0;
	uint64_t total_bytes_left = file_size;

	while (total_bytes_left > std::max<uint64_t>(2ull * LANE_COUNT * GEAR_HASHLEN, min_block_size)) {
		// Ensure the window offsets don't overflow (2GB is the max i32 value, 1GB is good enough)
		uint64_t window_bytes = std::min<uint64_t>(1 << 30, total_bytes_left);
		const uint64_t bytes_per_lane_without_overlap = (((window_bytes - GEAR_HASHLEN) / LANE_COUNT) / GEAR_HASHLEN) * GEAR_HASHLEN;
		const uint64_t bytes_per_lane = bytes_per_lane_without_overlap + GEAR_HASHLEN;
		window_bytes = (bytes_per_lane_without_overlap * LANE_COUNT) + GEAR_HASHLEN;

		__m256i hash = zero_vec;
		__m256i cutpoint_bitmap_vmask = zero_vec;
		const unsigned int gather_count = (bytes_per_lane / GEAR_HASHLEN) * 8;
		__m256i cbytes_by_gather[8]{};
		load_and_transpose_8x8_epi32_avx2(
			file_data + file_data_offset,
			bytes_per_lane_without_overlap,
			cbytes_by_gather
		);
		for (int warmup_iter = 0; warmup_iter < 8; warmup_iter++) {
			__m256i tentry0;
			__m256i tentry1;
			__m256i tentry2;
			__m256i tentry3;
			gather_gear_entries_avx2(cbytes_by_gather[warmup_iter], cmask, tentry0, tentry1, tentry2, tentry3);
			roll_gear_avx2(hash, tentry0);
			roll_gear_avx2(hash, tentry1);
			roll_gear_avx2(hash, tentry2);
			roll_gear_avx2(hash, tentry3);
		}

		unsigned int gather_i = 8;
		__m256i tentries_by_gather[8][4]{};
		__m256i hashes_by_gather[8][4]{};
		while (gather_i < gather_count) {
			load_and_transpose_8x8_epi32_avx2(
				file_data + file_data_offset + gather_i * sizeof(int32_t),
				bytes_per_lane_without_overlap,
				cbytes_by_gather
			);

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
				record_cutpoints_avx2(hashes_by_gather[inner_gather_i][0], mm_break_mark, zero_vec, high_bit_vec, cutpoint_bitmap_vmask);
				record_cutpoints_avx2(hashes_by_gather[inner_gather_i][1], mm_break_mark, zero_vec, high_bit_vec, cutpoint_bitmap_vmask);
				record_cutpoints_avx2(hashes_by_gather[inner_gather_i][2], mm_break_mark, zero_vec, high_bit_vec, cutpoint_bitmap_vmask);
				record_cutpoints_avx2(hashes_by_gather[inner_gather_i][3], mm_break_mark, zero_vec, high_bit_vec, cutpoint_bitmap_vmask);
			}


			if (!_mm256_testz_si256(cutpoint_bitmap_vmask, cutpoint_bitmap_vmask)) {
				alignas(32) uint32_t bitmap_words[LANE_COUNT];
				_mm256_store_si256(reinterpret_cast<__m256i*>(bitmap_words), cutpoint_bitmap_vmask);
				const uint64_t bitmap_batch_offset =
					(file_data_offset + gather_i * sizeof(int32_t)) >> 3;
				const uint64_t bitmap_bytes_per_lane = bytes_per_lane_without_overlap >> 3;
				for (uint32_t lane_i = 0; lane_i < LANE_COUNT; lane_i++) {
					const uint64_t bitmap_offset = bitmap_batch_offset + lane_i * bitmap_bytes_per_lane;
					std::memcpy(cutpoint_bitmap + bitmap_offset, &bitmap_words[lane_i], sizeof(uint32_t));
				}
			}

			cutpoint_bitmap_vmask = zero_vec;
			gather_i += 8;
		}

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
		if (!(pattern & mask)) {
			const uint32_t byte_pos = file_data_offset / 8u;
			const uint8_t bit_pos = file_data_offset % 8u;
			cutpoint_bitmap[byte_pos] = cutpoint_bitmap[byte_pos] | static_cast<uint8_t>(1u << bit_pos);
		}
		file_data_offset++;
	}
}
#undef doGearAvx2
#endif

#if defined(__AVX512F__)
static inline void load_and_transpose_16x8_epi32_avx512(
		const unsigned char* RESTRICT file_data,
		uint64_t bytes_per_lane,
		__m512i (&columns)[8]) {
	assert((reinterpret_cast<std::uintptr_t>(file_data) % 32) == 0);
	assert((bytes_per_lane % 32) == 0);
	constexpr __mmask16 LOW_EIGHT_DWORDS = 0x00ff;
	__m512i rows[16];
	for (uint32_t lane_i = 0; lane_i < 16; lane_i++) {
		rows[lane_i] = _mm512_maskz_loadu_epi32(
			LOW_EIGHT_DWORDS,
			file_data + lane_i * bytes_per_lane
		);
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

// AVX-512 version of SS-CDC phase one that replaces input gathers with contiguous lane loads and transposes.
// Gear-table lookups retain the gather scheduling used by the optimization-level-one implementation.
static void sscdc_chunking_phase_one_avx512_gear_with_load_transpose(uint32_t mask, uint64_t min_block_size, const unsigned char* RESTRICT file_data, uint64_t file_size, uint8_t* RESTRICT cutpoint_bitmap) {
	static constexpr uint32_t LANE_COUNT = 16;
	const __m512i mm_break_mark = _mm512_set1_epi32(mask);
	const __m512i cmask = _mm512_set1_epi32(0xff);
	const __m512i zero_vec = _mm512_setzero_si512();
	uint64_t file_data_offset = 0;
	uint64_t total_bytes_left = file_size;

	while (total_bytes_left > std::max<uint64_t>(2ull * LANE_COUNT * GEAR_HASHLEN, min_block_size)) {
		// Ensure the window offsets don't overflow (2GB is the max i32 value, 1GB is good enough)
		uint64_t window_bytes = std::min<uint64_t>(1 << 30, total_bytes_left);
		const uint64_t bytes_per_lane_without_overlap = (((window_bytes - GEAR_HASHLEN) / LANE_COUNT) / GEAR_HASHLEN) * GEAR_HASHLEN;
		const uint64_t bytes_per_lane = bytes_per_lane_without_overlap + GEAR_HASHLEN;
		window_bytes = (bytes_per_lane_without_overlap * LANE_COUNT) + GEAR_HASHLEN;

		__m512i vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
		vindex = _mm512_mullo_epi32(vindex, _mm512_set1_epi32(bytes_per_lane_without_overlap));

		__m512i hash = zero_vec;
		__m512i cutpoint_bitmap_vmask = zero_vec;
		const unsigned int gather_count = (bytes_per_lane / GEAR_HASHLEN) * 8;
		__m512i cbytes_by_gather[8]{};
		load_and_transpose_16x8_epi32_avx512(
			file_data + file_data_offset,
			bytes_per_lane_without_overlap,
			cbytes_by_gather
		);
		for (int warmup_iter = 0; warmup_iter < 8; warmup_iter++) {
			__m512i tentry0;
			__m512i tentry1;
			__m512i tentry2;
			__m512i tentry3;
			gather_gear_entries_avx512(cbytes_by_gather[warmup_iter], cmask, tentry0, tentry1, tentry2, tentry3);
			roll_gear_avx512(hash, tentry0);
			roll_gear_avx512(hash, tentry1);
			roll_gear_avx512(hash, tentry2);
			roll_gear_avx512(hash, tentry3);
		}

		unsigned int gather_i = 8;
		vindex = _mm512_add_epi32(vindex, _mm512_set1_epi32(GEAR_HASHLEN));
		__m512i tentries_by_gather[8][4]{};
		__m512i hashes_by_gather[8][4]{};
		while (gather_i < gather_count) {
			load_and_transpose_16x8_epi32_avx512(
				file_data + file_data_offset + gather_i * sizeof(int32_t),
				bytes_per_lane_without_overlap,
				cbytes_by_gather
			);

			for (int inner_gather_i = 0; inner_gather_i < 8; inner_gather_i++) {
				auto& tentry = tentries_by_gather[inner_gather_i];
				gather_gear_entries_avx512(cbytes_by_gather[inner_gather_i], cmask, tentry[0], tentry[1], tentry[2], tentry[3]);
			}

			for (int inner_gather_i = 0; inner_gather_i < 8; inner_gather_i++) {
				roll_gear_avx512(hash, tentries_by_gather[inner_gather_i][0]);
				hashes_by_gather[inner_gather_i][0] = hash;
				roll_gear_avx512(hash, tentries_by_gather[inner_gather_i][1]);
				hashes_by_gather[inner_gather_i][1] = hash;
				roll_gear_avx512(hash, tentries_by_gather[inner_gather_i][2]);
				hashes_by_gather[inner_gather_i][2] = hash;
				roll_gear_avx512(hash, tentries_by_gather[inner_gather_i][3]);
				hashes_by_gather[inner_gather_i][3] = hash;
			}

			for (int inner_gather_i = 0; inner_gather_i < 8; inner_gather_i++) {
				record_cutpoints_avx512(hashes_by_gather[inner_gather_i][0], mm_break_mark, cutpoint_bitmap_vmask);
				record_cutpoints_avx512(hashes_by_gather[inner_gather_i][1], mm_break_mark, cutpoint_bitmap_vmask);
				record_cutpoints_avx512(hashes_by_gather[inner_gather_i][2], mm_break_mark, cutpoint_bitmap_vmask);
				record_cutpoints_avx512(hashes_by_gather[inner_gather_i][3], mm_break_mark, cutpoint_bitmap_vmask);
			}

			const __mmask16 lanes_with_results_mask = _mm512_cmpneq_epi32_mask(cutpoint_bitmap_vmask, zero_vec);
			if (lanes_with_results_mask != 0) {
				const auto cutpoint_bitmap_vindex = _mm512_srli_epi32(vindex, 3);
				const auto cutpoint_bitmap_offset = file_data_offset >> 3;
				_mm512_mask_i32scatter_epi32(
					cutpoint_bitmap + cutpoint_bitmap_offset,
					lanes_with_results_mask,

					cutpoint_bitmap_vindex,
					cutpoint_bitmap_vmask,
					1
				);
			}

			cutpoint_bitmap_vmask = zero_vec;
			vindex = _mm512_add_epi32(vindex, _mm512_set1_epi32(GEAR_HASHLEN));
			gather_i += 8;
		}

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
		if (!(pattern & mask)) {
			const uint32_t byte_pos = file_data_offset / 8u;
			const uint8_t bit_pos = file_data_offset % 8u;
			cutpoint_bitmap[byte_pos] = cutpoint_bitmap[byte_pos] | static_cast<uint8_t>(1u << bit_pos);
		}
		file_data_offset++;
	}
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

static bool next_chunk(uint64_t min_block_size, uint64_t max_block_size, uint64_t file_size, chunk_boundary& cb, SIMD_Mode simd_mode, const uint8_t* cutpoint_bitmap) {
	uint64_t current_chunk_size = 0;
	uint64_t prev_cut_offset = cb.right;

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

std::vector<std::string> Gear_Chunking::chunk_file(std::string file_path) {
	if (simd_mode == SIMD_Mode::NONE) {
		return Chunking_Technique::chunk_file(file_path);
	}

	using phase_one_function = void (*)(uint32_t, uint64_t, const unsigned char*, uint64_t, uint8_t*);
	phase_one_function phase_one = nullptr;
	switch (simd_mode) {
	#if defined(__AVX2__)
		case SIMD_Mode::AVX256: {
			switch (optimization_level) {
				case 1:
					phase_one = sscdc_chunking_phase_one_avx2_gear_with_gather_scheduling;
					break;
				case 2:
					phase_one = sscdc_chunking_phase_one_avx2_gear_with_load_transpose;
					break;
				default:
					phase_one = sscdc_chunking_phase_one_avx2_gear;
					break;
			}
			break;
		}
	#endif
	#if defined(__AVX512F__)
		case SIMD_Mode::AVX512: {
			switch (optimization_level) {
				case 1:
					phase_one = sscdc_chunking_phase_one_avx512_gear_with_gather_scheduling;
					break;
				case 2:
					phase_one = sscdc_chunking_phase_one_avx512_gear_with_load_transpose;
					break;
				default:
					phase_one = sscdc_chunking_phase_one_avx512_gear;
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
	uint32_t avg_mask = static_cast<uint32_t>(mask >> 32);
	phase_one(avg_mask, min_block_size, file_data, file_size, cutpoint_bitmap);
	auto end_chunking = std::chrono::high_resolution_clock::now();
	total_time_chunking += (end_chunking - begin_chunking);

	std::memset(&cb, 0, sizeof(chunk_boundary));
	std::vector<std::string> hashes{};
	uint64_t prev_cut_offset = cb.right;
	while (next_chunk(min_block_size, max_block_size, file_size, cb, SIMD_Mode::NONE, cutpoint_bitmap)) {
#ifndef NDEBUG
		// DEBUG: serially check next chunk size, crash if there is a mismatch
		auto debug_curr_pos = prev_cut_offset;
		uint32_t hash = 0;
		for (; debug_curr_pos < file_size; debug_curr_pos++) {
			hash = (hash << 1) + gear_common::table_32[file_data[debug_curr_pos]];
			if (!(hash & avg_mask) && (debug_curr_pos - prev_cut_offset >= min_block_size)) {
				break;
			}
			if ((debug_curr_pos - prev_cut_offset == max_block_size)) {
				break;
			}
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

uint64_t Gear_Chunking::find_cutpoint(char* data, uint64_t size) {
    if (simd_mode == SIMD_Mode::NONE)
      return find_cutpoint_serial(data, size);
	else if (simd_mode == SIMD_Mode::AVX256 || simd_mode == SIMD_Mode::AVX512) {
		uint64_t prev_cut_offset = cb.right;
		next_chunk(min_block_size, max_block_size, file_size, cb, simd_mode, cutpoint_bitmap);
		return cb.right - prev_cut_offset;
		//return find_cutpoint_avx512(data, size);
	}
    else {
      std::cout << "SIMD mode unsupported for chosen chunking technique" << std::endl;
      exit(1);
    }
}

uint64_t Gear_Chunking::find_cutpoint_serial(char* data, uint64_t size) {
  uint64_t idx = 0;

  // If given data is lower than the minimum chunk size, return data length.

  if (size <= min_block_size) {
    return size;
  }

  const uint64_t limit = std::min(size, max_block_size);

  if (use_64bit_gear) {
    uint64_t hash = 0;
    const uint64_t serial_mask = use_low_bit_mask
        ? avg_block_size - 1
        : mask;
  	while (idx < min_block_size) {
  		hash = gear_common::roll(hash, static_cast<uint8_t>(data[idx]));
  		idx += 1;
  	}
    while (idx < limit) {
      hash = gear_common::roll(hash, static_cast<uint8_t>(data[idx]));
      if (!(hash & serial_mask)) {
        return idx;
      }
      idx += 1;
    }
  } else {
    uint32_t hash = 0;
    const uint32_t mask_32 = use_low_bit_mask
        ? static_cast<uint32_t>(avg_block_size - 1)
        : static_cast<uint32_t>(mask >> 32);
  	while (idx < min_block_size) {
  		hash = gear_common::roll(hash, static_cast<uint8_t>(data[idx]));
  		idx += 1;
  	}
    while (idx < limit) {
      hash = gear_common::roll(hash, static_cast<uint8_t>(data[idx]));
      if (!(hash & mask_32)) {
        return idx;
      }
      idx += 1;
    }
  }

  return idx;
}

uint64_t Gear_Chunking::find_cutpoint_avx512(char* data, uint64_t size) {
  return find_cutpoint_serial(data, size);
}
