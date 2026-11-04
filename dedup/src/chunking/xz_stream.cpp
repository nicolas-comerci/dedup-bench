/**
 * @file xz_stream.cpp
 * @brief In-memory dataset streams, including .xz decompression
 */
#include "xz_stream.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include <lzma.h>

#include "aligned_alloc.hpp"

// ========== .xz decompression =============

namespace {

bool ends_with_xz(const std::string& path) {
    constexpr const char* ext = ".xz";
    constexpr std::size_t ext_len = 3;
    if (path.size() < ext_len) {
        return false;
    }
    for (std::size_t i = 0; i < ext_len; ++i) {
        const char c = path[path.size() - ext_len + i];
        if (std::tolower(static_cast<unsigned char>(c)) != ext[i]) {
            return false;
        }
    }
    return true;
}

constexpr uint64_t kXzStreamHeaderSize = 12;
constexpr uint64_t kXzStreamFooterSize = 12;
constexpr uint8_t kXzHeaderMagic[6] = {0xFD, 0x37, 0x7A, 0x58, 0x5A, 0x00};
constexpr uint8_t kXzFooterMagic[2] = {0x59, 0x5A};

uint32_t read32le(const uint8_t* data) {
    return static_cast<uint32_t>(data[0])
        | (static_cast<uint32_t>(data[1]) << 8)
        | (static_cast<uint32_t>(data[2]) << 16)
        | (static_cast<uint32_t>(data[3]) << 24);
}

bool read_stream_bytes(std::ifstream& in, uint8_t* data, uint64_t count) {
    constexpr uint64_t max_read_size = 1ull << 20;
    uint64_t total = 0;
    while (total < count) {
        const uint64_t bytes_left = count - total;
        const auto bytes_to_read = static_cast<std::streamsize>(
            std::min(max_read_size, bytes_left));
        in.read(reinterpret_cast<char*>(data + total), bytes_to_read);
        const std::streamsize bytes_read = in.gcount();
        if (bytes_read != bytes_to_read) {
            return false;
        }
        total += static_cast<uint64_t>(bytes_read);
    }
    return true;
}

bool read_xz_vli(const uint8_t* data, std::size_t size, std::size_t& pos,
                 uint64_t& value) {
    uint64_t result = 0;
    for (int i = 0; i < 9; ++i) {
        if (pos >= size) {
            return false;
        }
        const uint8_t byte = data[pos++];
        result |= static_cast<uint64_t>(byte & 0x7F) << (i * 7);
        if ((byte & 0x80) == 0) {
            value = result;
            return true;
        }
    }
    return false;
}

bool parse_xz_index(const uint8_t* index, std::size_t size,
                    uint64_t& uncompressed_total, uint64_t& blocks_total) {
    if (size < 5 || index[0] != 0x00) {
        return false;
    }

    std::size_t pos = 1;
    uint64_t records = 0;
    if (!read_xz_vli(index, size, pos, records)) {
        return false;
    }

    uint64_t uncompressed_sum = 0;
    uint64_t padded_blocks_sum = 0;
    for (uint64_t i = 0; i < records; ++i) {
        // Index records store Compressed Size first, then Uncompressed Size.
        uint64_t record_compressed = 0;
        uint64_t record_uncompressed = 0;
        if (!read_xz_vli(index, size, pos, record_compressed)
            || !read_xz_vli(index, size, pos, record_uncompressed)) {
            return false;
        }

        // The recorded compressed size excludes the block's alignment padding.
        const uint64_t padded_block_size = (record_compressed + 3) & ~3ull;
        if (uncompressed_sum > std::numeric_limits<uint64_t>::max() - record_uncompressed
            || padded_blocks_sum > std::numeric_limits<uint64_t>::max() - padded_block_size) {
            return false;
        }
        uncompressed_sum += record_uncompressed;
        padded_blocks_sum += padded_block_size;
    }

    // The index must contain its trailing CRC32. Padding is not needed for
    // sizing, so it is intentionally not validated here.
    if (pos + 4 > size) {
        return false;
    }

    uncompressed_total = uncompressed_sum;
    blocks_total = padded_blocks_sum;
    return true;
}

bool read_xz_uncompressed_size(const std::string& path, uint64_t& out_size) {
    std::ifstream in(path, std::ios::in | std::ios::binary);
    if (!in.is_open()) {
        return false;
    }

    in.seekg(0, std::ios_base::end);
    const uint64_t file_size = static_cast<uint64_t>(in.tellg());
    in.seekg(0, std::ios_base::beg);
    if (file_size < kXzStreamHeaderSize + kXzStreamFooterSize) {
        return false;
    }

    uint64_t pos = file_size;
    uint64_t total_uncompressed = 0;

    // Walk backwards through Stream Footer / Index records. Each index record
    // also contains the block's compressed size, which lets us locate the
    // preceding Stream Header and continue with concatenated streams.
    while (pos > 0) {
        if (pos < kXzStreamHeaderSize + kXzStreamFooterSize) {
            return false;
        }

        const uint64_t footer_pos = pos - kXzStreamFooterSize;
        uint8_t footer[kXzStreamFooterSize]{};
        in.seekg(static_cast<std::streamoff>(footer_pos), std::ios_base::beg);
        if (!read_stream_bytes(in, footer, sizeof(footer))) {
            return false;
        }
        if (std::memcmp(footer + 10, kXzFooterMagic,
                        sizeof(kXzFooterMagic)) != 0) {
            return false;
        }

        const uint64_t index_size =
            (static_cast<uint64_t>(read32le(footer + 4)) + 1) * 4;
        if (index_size < 4 || footer_pos < index_size
            || index_size > std::numeric_limits<std::size_t>::max()) {
            return false;
        }

        const uint64_t index_start = footer_pos - index_size;
        std::vector<uint8_t> index(static_cast<std::size_t>(index_size));
        in.seekg(static_cast<std::streamoff>(index_start), std::ios_base::beg);
        if (!read_stream_bytes(in, index.data(), index.size())) {
            return false;
        }

        uint64_t stream_uncompressed = 0;
        uint64_t stream_blocks = 0;
        if (!parse_xz_index(index.data(), index.size(), stream_uncompressed,
                            stream_blocks)) {
            return false;
        }
        if (index_start < kXzStreamHeaderSize + stream_blocks) {
            return false;
        }

        const uint64_t stream_start =
            index_start - stream_blocks - kXzStreamHeaderSize;
        uint8_t header_magic[sizeof(kXzHeaderMagic)]{};
        in.seekg(static_cast<std::streamoff>(stream_start), std::ios_base::beg);
        if (!read_stream_bytes(in, header_magic, sizeof(header_magic))
            || std::memcmp(header_magic, kXzHeaderMagic,
                           sizeof(kXzHeaderMagic)) != 0) {
            return false;
        }

        if (total_uncompressed >
            std::numeric_limits<uint64_t>::max() - stream_uncompressed) {
            return false;
        }
        total_uncompressed += stream_uncompressed;
        pos = stream_start;
    }

    out_size = total_uncompressed;
    return true;
}

// Owned, 64-byte-aligned decompression result. `ok` distinguishes a successful
// empty stream from a failed decompression.
struct XzDecompressed {
    char* data = nullptr;
    uint64_t size = 0;
    bool ok = false;
};

constexpr uint64_t kInputBlockSize = 1ull << 20;

XzDecompressed xz_decompress_to_fixed(const std::string& path,
                                      uint64_t declared_size) {
    XzDecompressed result;

    std::ifstream in(path, std::ios::in | std::ios::binary);
    if (!in.is_open()) {
        return result;
    }

    lzma_stream ls = LZMA_STREAM_INIT;
    if (lzma_stream_decoder(&ls, UINT64_MAX, LZMA_CONCATENATED) != LZMA_OK) {
        lzma_end(&ls);
        return result;
    }

    if (declared_size == std::numeric_limits<uint64_t>::max()) {
        return result;
    }

    // One extra byte lets liblzma always have output space available, which
    // makes detecting a stream that exceeds the declared size straightforward.
    const uint64_t capacity = declared_size + 1;
    char* out = static_cast<char*>(aligned_alloc_64(capacity));
    if (!out) {
        lzma_end(&ls);
        return result;
    }

    std::unique_ptr<uint8_t[]> in_block(new uint8_t[kInputBlockSize]);
    bool input_eof = false;
    uint64_t out_pos = 0;

    while (true) {
        if (ls.avail_in == 0) {
            in.read(reinterpret_cast<char*>(in_block.get()),
                    static_cast<std::streamsize>(kInputBlockSize));
            const std::streamsize bytes_read = in.gcount();
            ls.next_in = in_block.get();
            ls.avail_in = bytes_read > 0
                ? static_cast<std::size_t>(bytes_read) : 0;
            if (bytes_read <= 0) {
                input_eof = true;
            }
        }

        const uint64_t space = capacity - out_pos;
        ls.next_out = reinterpret_cast<uint8_t*>(out + out_pos);
        ls.avail_out = static_cast<std::size_t>(space);

        const lzma_action action = input_eof ? LZMA_FINISH : LZMA_RUN;
        const lzma_ret ret = lzma_code(&ls, action);
        out_pos += space - static_cast<uint64_t>(ls.avail_out);

        if (ret == LZMA_STREAM_END) {
            if (out_pos == declared_size) {
                result.data = out;
                result.size = out_pos;
                result.ok = true;
            }
            lzma_end(&ls);
            if (!result.ok) {
                aligned_free_64(out);
            }
            return result;
        }
        if (ret != LZMA_OK || out_pos > declared_size || out_pos == capacity) {
            break;
        }
        // No input left and nothing produced: the stream is truncated.
        if (input_eof && ls.avail_in == 0 && ls.avail_out == space) {
            break;
        }
    }

    aligned_free_64(out);
    lzma_end(&ls);
    return result;
}

XzDecompressed xz_decompress_to_memory(const std::string& path) {
    uint64_t declared_size = 0;
    if (!read_xz_uncompressed_size(path, declared_size)) {
        std::cerr << "Failed to determine the uncompressed size of "
                  << path << std::endl;
        return {};
    }

    XzDecompressed result = xz_decompress_to_fixed(path, declared_size);
    if (!result.ok) {
        std::cerr << "Failed to decompress " << path
                  << " (expected " << declared_size
                  << " uncompressed bytes)" << std::endl;
    }
    return result;
}

}  // namespace

// ========== open_dataset_stream =============

DatasetStream open_dataset_stream(const std::string& file_path) {
    DatasetStream ds;

    if (ends_with_xz(file_path)) {
        XzDecompressed decompressed = xz_decompress_to_memory(file_path);
        if (!decompressed.ok) {
            return DatasetStream{};
        }
        auto mem = std::make_unique<MemoryInputStream>(decompressed.data,
                                                       decompressed.size);
        ds.size = decompressed.size;
        ds.data = reinterpret_cast<const unsigned char*>(mem->data());
        ds.stream = std::move(mem);
        return ds;
    }

    auto file = std::make_unique<std::ifstream>();
    file->open(file_path, std::ios::in | std::ios::binary);
    if (!file->is_open()) {
        std::cerr << "Failed to open " << file_path << " for reading"
                  << std::endl;
        return DatasetStream{};
    }
    file->seekg(0, std::ios::end);
    ds.size = static_cast<uint64_t>(file->tellg());
    file->seekg(0, std::ios::beg);
    ds.stream = std::move(file);
    return ds;
}
