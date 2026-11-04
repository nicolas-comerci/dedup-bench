/**
 * @file memory_stream.cpp
 * @brief A seekable in-memory std::istream
 */
#include "memory_stream.hpp"

int MemoryInputStream::Buffer::underflow() {
    // The get area is fully populated; signal end-of-data once it is consumed.
    return traits_type::eof();
}

std::streamsize MemoryInputStream::Buffer::showmanyc() {
    return static_cast<std::streamsize>(egptr() - gptr());
}

MemoryInputStream::Buffer::pos_type
MemoryInputStream::Buffer::seekoff(off_type off, std::ios_base::seekdir dir,
                                   std::ios_base::openmode) {
    pos_type base;
    if (dir == std::ios_base::beg) {
        base = 0;
    } else if (dir == std::ios_base::cur) {
        base = static_cast<pos_type>(gptr() - m_data);
    } else {  // std::ios_base::end
        base = static_cast<pos_type>(m_size);
    }
    const pos_type target = base + off;
    if (target < 0 || target > static_cast<pos_type>(m_size)) {
        return pos_type(traits_type::eof());
    }
    setg(m_data, m_data + target, m_data + m_size);
    return target;
}

MemoryInputStream::Buffer::pos_type
MemoryInputStream::Buffer::seekpos(pos_type pos, std::ios_base::openmode) {
    if (pos < 0 || pos > static_cast<pos_type>(m_size)) {
        return pos_type(traits_type::eof());
    }
    setg(m_data, m_data + pos, m_data + m_size);
    return pos;
}

MemoryInputStream::MemoryInputStream(char* data, std::size_t size)
    : std::istream(new Buffer(data, size, true)) {}

MemoryInputStream::MemoryInputStream(const char* data, std::size_t size, NonOwning)
    : std::istream(new Buffer(const_cast<char*>(data), size, false)) {}

uint64_t MemoryInputStream::size() const {
    return static_cast<const Buffer*>(rdbuf())->size();
}

const char* MemoryInputStream::data() const {
    return static_cast<const Buffer*>(rdbuf())->data();
}
