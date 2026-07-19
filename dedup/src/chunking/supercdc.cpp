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

SuperCDC::SuperCDC(const Config& config) {
    min_block_size = config.get_fastcdc_min_block_size();
    avg_block_size = config.get_fastcdc_avg_block_size();
    max_block_size = config.get_fastcdc_max_block_size();
    bool disable_normalization = config.get_fastcdc_disable_normalization();
    if (disable_normalization) {
        normalization_level = 0;
    } else {
        normalization_level = config.get_fastcdc_normalization_level();
    }
    int mask_bits = int(round(log2(avg_block_size)));
    small_mask = (1 << (mask_bits + normalization_level)) - 1 ;
    large_mask = (1 << (mask_bits - normalization_level)) - 1 ;
    backup_mask = large_mask >> 1;
}

uint64_t SuperCDC::find_cutpoint(char* data, uint64_t len) {
    uint64_t fp = 0;
    uint64_t i = min_block_size;  // skip min block size
    if (len < min_block_size) {
        return len;
    }
    uint64_t length = std::min(len, max_block_size);
    uint64_t first_phase = std::min(length, avg_block_size);
    uint64_t backup_i = 0;

    for (; i < first_phase; i++) {
        fp = (fp << 1) + GEAR_TABLE[(int)data[i]];
        if ((fp & small_mask) == 0) {
            return i ;
        }
    }

    // Get backup cut position in case we end up cutting because of max_block_size
    for (; i < length; i++) {
        fp = (fp << 1) + GEAR_TABLE[(int)data[i]];
        if ((fp & backup_mask) == 0) {
          // If this also satisfies the actual second half cut condition, just cut
          if ((fp & large_mask) == 0) {
            return i;
          }
          backup_i = i;
        }
    }

    for (; i < length; i++) {
      fp = (fp << 1) + GEAR_TABLE[(int)data[i]];
      if ((fp & large_mask) == 0) {
        return i;
      }
    }

    // Reached max_block_size, use SuperCDC backup if it exists, else cut with length=max_block_size
    return backup_i > 0 ? backup_i : length; //Double check that this is safe to return length here
}

SuperCDC::~SuperCDC() {}
