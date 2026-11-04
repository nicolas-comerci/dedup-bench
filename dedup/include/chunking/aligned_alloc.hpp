#ifndef _ALIGNED_ALLOC_
#define _ALIGNED_ALLOC_

#include <cstddef>
#include <cstdlib>
#include <limits>

#ifdef _WIN32
#include <malloc.h>
#endif

/**
 * @brief Allocate `size` bytes with 64-byte alignment (sufficient for the
 * SIMD loads used by the chunkers). Returns nullptr on failure or when size
 * is 0. Free the result with aligned_free_64().
 */
inline void* aligned_alloc_64(std::size_t size) {
    if (size == 0) {
        return nullptr;
    }
    const std::size_t minimum_size = size > 64 ? size : static_cast<std::size_t>(64);
    if (minimum_size > std::numeric_limits<std::size_t>::max() - 63) {
        return nullptr;
    }
    const std::size_t aligned_size = ((minimum_size + 63) / 64) * 64;
#ifdef _WIN32
    return _aligned_malloc(aligned_size, 64);
#else
    return std::aligned_alloc(64, aligned_size);
#endif
}

/**
 * @brief Free a buffer returned by aligned_alloc_64(). nullptr is a no-op.
 */
inline void aligned_free_64(void* ptr) {
    if (ptr == nullptr) {
        return;
    }
#ifdef _WIN32
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

#endif
