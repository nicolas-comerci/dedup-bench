/**
 * @file xz_stream.hpp
 * @brief In-memory dataset streams, including .xz decompression
 *
 * Provides a helper that opens a dataset file for chunking. Files ending in
 * ".xz" are decompressed fully into memory (via MemoryInputStream) so that
 * large datasets can be stored compressed on disk.
 */
#ifndef _XZ_STREAM_
#define _XZ_STREAM_

#include <cstdint>
#include <istream>
#include <memory>
#include <string>

#include "memory_stream.hpp"

/**
 * @brief Result of opening a dataset file for chunking.
 */
struct DatasetStream {
    std::unique_ptr<std::istream> stream;  // null for in-memory datasets / on failure
    const unsigned char* data = nullptr;   // non-owning view, valid while `stream` lives
    uint64_t size = 0;                     // logical (decompressed) byte count
};

/**
 * @brief Open `file_path` for chunking.
 *
 * If the path ends in ".xz" (case-insensitive) the file is decompressed fully
 * into memory and returned as a MemoryInputStream. Otherwise it is opened as a
 * plain binary file. On any failure an error is printed to stderr and a
 * DatasetStream with a null `stream` is returned.
 *
 * @param file_path: Path to the dataset file
 * @return DatasetStream holding the input stream and its byte count
 */
DatasetStream open_dataset_stream(const std::string& file_path);

#endif
