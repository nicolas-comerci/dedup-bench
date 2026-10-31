#ifndef _GEAR_COMMON_
#define _GEAR_COMMON_

#include <cstdint>

namespace gear_common {

extern const uint32_t table_32[256];
extern const uint64_t table_64[256];

// table_32 entries shifted left by 0 to 15 bits, entries[s][b] ==
// table_32[b] << s: the pre-shifted contributions of group-end judgment
// over 16-byte groups.
struct shifted_table_set {
    static constexpr int shift_count = 16;
    alignas(64) uint32_t entries[shift_count][256];
};
extern const shifted_table_set table_32_shifted;

inline uint32_t roll(uint32_t hash, uint8_t byte) {
    return (hash << 1) + table_32[byte];
}

inline uint64_t roll(uint64_t hash, uint8_t byte) {
    return (hash << 1) + table_64[byte];
}

}  // namespace gear_common

#endif

