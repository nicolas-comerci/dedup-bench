/**
 * @file supercdc.cpp
 * @author NicolasComerci
 * @brief Implementations for SuperCDC chunking technique
 * @version 0.1
 * @date 2026-6-1
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "supercdc.hpp"

#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>

#include "gear_common.hpp"

SuperCDC::SuperCDC(const Config& config) {
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
    backup_mask = large_mask >> 1;
}

uint64_t SuperCDC::find_cutpoint(char* data, uint64_t len) {
    uint64_t i = min_block_size - (use_64bit_gear ? 64 : 32);  // skip min block size
    if (len < min_block_size) {
        return len;
    }
    uint64_t length = std::min(len, max_block_size);
    uint64_t first_phase = std::min(length, avg_block_size);
    uint64_t backup_i = 0;

    if (use_64bit_gear) {
        uint64_t fp = 0;
        // context repair
        for (int j = 0; j < 64; i++, j++) {
            fp = gear_common::roll(fp, static_cast<uint8_t>(data[i]));
        }

        // pre average hard cut condition
        for (; i < first_phase; i++) {
            fp = gear_common::roll(fp, static_cast<uint8_t>(data[i]));
            if ((fp & small_mask) == 0) {
                return i;
            }
        }

        // post average easy cut condition with backup condition
        for (; i < length; i++) {
            fp = gear_common::roll(fp, static_cast<uint8_t>(data[i]));
            if ((fp & backup_mask) == 0) {
                if ((fp & large_mask) == 0) {
                    return i;
                }
                backup_i = i;
                // We have our backup, move to faster loop without backup check
                i++;
                break;
            }
        }

        // remaining post average, post backup condition region
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
        const uint32_t backup_mask_32 = static_cast<uint32_t>(backup_mask);
        // context repair
        for (int j = 0; j < 32; i++, j++) {
            fp = gear_common::roll(fp, static_cast<uint8_t>(data[i]));
        }

        // pre average hard cut condition
        for (; i < first_phase; i++) {
            fp = gear_common::roll(fp, static_cast<uint8_t>(data[i]));
            if ((fp & small_mask_32) == 0) {
                return i;
            }
        }

        // post average easy cut condition with backup condition
        for (; i < length; i++) {
            fp = gear_common::roll(fp, static_cast<uint8_t>(data[i]));
            if ((fp & backup_mask_32) == 0) {
                if ((fp & large_mask_32) == 0) {
                    return i;
                }
                backup_i = i;
                // We have our backup, move to faster loop without backup check
                i++;
                break;
            }
        }

        // remaining post average, post backup condition region
        for (; i < length; i++) {
            fp = gear_common::roll(fp, static_cast<uint8_t>(data[i]));
            if ((fp & large_mask_32) == 0) {
                return i;
            }
        }
    }

    // Reached max_block_size, use SuperCDC backup if it exists, else cut with length=max_block_size
    return backup_i > 0 ? backup_i : length; //Double check that this is safe to return length here
}

SuperCDC::~SuperCDC() {}
