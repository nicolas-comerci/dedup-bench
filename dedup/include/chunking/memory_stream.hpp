/**
 * @file memory_stream.hpp
 * @brief A seekable in-memory std::istream
 */
#ifndef _MEMORY_STREAM_
#define _MEMORY_STREAM_

#include <cstddef>
#include <cstdint>
#include <istream>

#include "aligned_alloc.hpp"

/**
 * @brief A seekable std::istream backed by an in-memory buffer.
 *
 * Two construction modes:
 *  - Owning: the stream takes a 64-byte-aligned buffer and frees it on
 *    destruction (used for decompressed .xz datasets).
 *  - Non-owning: a read-only view over externally-owned memory that is never
 *    freed here (used to run chunk_stream() over an already-materialized
 *    buffer without copying it).
 *
 * No extra copy of the data is made, so memory usage is exactly the size of
 * the data. Seeking is supported so that chunk_stream() can query the total
 * byte count and re-read windows.
 */
class MemoryInputStream : public std::istream {
public:
    /**
     * @brief Owning constructor. Takes a 64-byte-aligned buffer that this
     * stream will free when destroyed.
     */
    MemoryInputStream(char* data, std::size_t size);

    /** Tag type marking the non-owning constructor. */
    struct NonOwning {};

    /**
     * @brief Non-owning view over externally-owned memory (never freed).
     */
    MemoryInputStream(const char* data, std::size_t size, NonOwning);

    /**
     * @brief Total number of bytes held by the stream.
     * @return uint64_t
     */
    uint64_t size() const;

    /**
     * @brief Pointer to the first byte of the underlying buffer.
     * @return const char*
     */
    const char* data() const;

private:
    class Buffer : public std::streambuf {
    public:
        Buffer(char* data, std::size_t size, bool owns)
            : m_data(data), m_size(size), m_owns(owns) {
            // Expose the whole buffer as the get area up front; no refilling is
            // ever needed because all of the data is already in memory.
            setg(m_data, m_data, m_data + m_size);
        }

        ~Buffer() override {
            if (m_owns) {
                aligned_free_64(m_data);
            }
        }

        std::size_t size() const { return m_size; }
        const char* data() const { return m_data; }

    protected:
        int underflow() override;
        std::streamsize showmanyc() override;
        pos_type seekoff(off_type off, std::ios_base::seekdir dir,
                         std::ios_base::openmode which) override;
        pos_type seekpos(pos_type pos, std::ios_base::openmode which) override;

    private:
        char* m_data;
        std::size_t m_size;
        bool m_owns;
    };
};

#endif
