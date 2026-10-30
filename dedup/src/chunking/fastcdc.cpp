/**
 * @file fastcdc.cpp
 * @author WASL
 * @brief Implementations for FastCDC chunking technique
 * @version 0.1
 * @date 2023-4-1
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "fastcdc.hpp"

#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>

#include "config_error.hpp"
#include "gear_common.hpp"

// Rolling two bytes each time (Xia et al., TPDS 2020, Section 3.7): one
// shift per byte pair. After the even byte, fp is the one-byte-at-a-time
// fingerprint shifted left by one, so it is judged with the mask shifted left
// by one; after the odd byte it is exact again. Cut-points are identical to
// rolling one byte at a time as long as the mask leaves the top bit clear.
// Returns the first cut-point in [i, end), or end if there is none, leaving
// fp exact.
template <typename Hash>
static uint64_t scan_two_bytes(const uint8_t* data, uint64_t i, uint64_t end,
                               Hash& fp, Hash mask, const Hash* gear,
                               const Hash* gear_ls) {
    const Hash mask_ls = mask << 1;
    for (; i + 2 <= end; i += 2) {
        fp = (fp << 2) + gear_ls[data[i]];
        if ((fp & mask_ls) == 0) {
            return i;
        }
        fp += gear[data[i + 1]];
        if ((fp & mask) == 0) {
            return i + 1;
        }
    }
    if (i < end) {
        fp = (fp << 1) + gear[data[i]];
        if ((fp & mask) == 0) {
            return i;
        }
    }
    return end;
}

template <typename Hash>
static uint64_t find_cutpoint_two_bytes(const uint8_t* data, uint64_t start,
                                        uint64_t first_phase, uint64_t length,
                                        Hash small_mask, Hash large_mask,
                                        const Hash* gear, const Hash* gear_ls) {
    Hash fp = 0;
    const uint64_t cut = scan_two_bytes(data, start, first_phase, fp,
                                        small_mask, gear, gear_ls);
    if (cut < first_phase) {
        return cut;
    }
    return scan_two_bytes(data, first_phase, length, fp, large_mask, gear,
                          gear_ls);
}

FastCDC::FastCDC(const Config& config) {
    min_block_size = config.get_fastcdc_min_block_size();
    avg_block_size = config.get_fastcdc_avg_block_size();
    max_block_size = config.get_fastcdc_max_block_size();
    use_64bit_gear = config.get_use_64bit_gear();
    bool disable_normalization = config.get_fastcdc_disable_normalization();
    if (disable_normalization) {
        normalization_level = 0;
    } else {
        normalization_level = config.get_fastcdc_normalization_level();
    }
    int mask_bits = int(round(log2(avg_block_size)));
    small_mask = (uint64_t{1} << (mask_bits + normalization_level)) - 1;
    large_mask = (uint64_t{1} << (mask_bits - normalization_level)) - 1;

    roll_two_bytes = config.get_fastcdc_roll_two_bytes();
    const int hash_bits = use_64bit_gear ? 64 : 32;
    if (roll_two_bytes && (small_mask >> (hash_bits - 1)) != 0) {
        throw ConfigError(
            "fastcdc_roll_two_bytes needs FastCDC masks that leave the "
            "hash's top bit clear");
    }
    for (int k = 0; k < 256; k++) {
        gear_ls_32[k] = gear_common::table_32[k] << 1;
        gear_ls_64[k] = gear_common::table_64[k] << 1;
    }
}

uint64_t FastCDC::find_cutpoint(char* data, uint64_t len) {
    uint64_t i = min_block_size;  // skip min block size
    if (len < min_block_size) {
        return len;
    }
    uint64_t length = std::min(len, max_block_size);
    uint64_t first_phase = std::min(length, avg_block_size);

    if (roll_two_bytes) {
        const auto* bytes = reinterpret_cast<const uint8_t*>(data);
        if (use_64bit_gear) {
            return find_cutpoint_two_bytes<uint64_t>(
                bytes, i, first_phase, length, small_mask, large_mask,
                gear_common::table_64, gear_ls_64.data());
        }
        return find_cutpoint_two_bytes<uint32_t>(
            bytes, i, first_phase, length, static_cast<uint32_t>(small_mask),
            static_cast<uint32_t>(large_mask), gear_common::table_32,
            gear_ls_32.data());
    }

    if (use_64bit_gear) {
        uint64_t fp = 0;
        for (; i < first_phase; i++) {
            fp = gear_common::roll(fp, static_cast<uint8_t>(data[i]));
            if ((fp & small_mask) == 0) {
                return i;
            }
        }

        for (; i < length; i++) {
            fp = gear_common::roll(fp, static_cast<uint8_t>(data[i]));
            if ((fp & large_mask) == 0) {
                return i;
            }
        }
    } else {
        uint32_t fp = 0;
        const uint32_t small_mask_32 = static_cast<uint32_t>(small_mask);
        const uint32_t large_mask_32 = static_cast<uint32_t>(large_mask);
        for (; i < first_phase; i++) {
            fp = gear_common::roll(fp, static_cast<uint8_t>(data[i]));
            if ((fp & small_mask_32) == 0) {
                return i;
            }
        }

        for (; i < length; i++) {
            fp = gear_common::roll(fp, static_cast<uint8_t>(data[i]));
            if ((fp & large_mask_32) == 0) {
                return i;
            }
        }
    }

    return length; //Double check that this is safe to return length here
}

FastCDC::~FastCDC() {}
