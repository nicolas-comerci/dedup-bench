#ifndef _GEAR_COMMON_
#define _GEAR_COMMON_

#include <cstdint>

namespace gear_common {

extern const uint32_t table_32[256];
extern const uint64_t table_64[256];

inline uint32_t roll(uint32_t hash, uint8_t byte) {
    return (hash << 1) + table_32[byte];
}

inline uint64_t roll(uint64_t hash, uint8_t byte) {
    return (hash << 1) + table_64[byte];
}

}  // namespace gear_common

#endif

