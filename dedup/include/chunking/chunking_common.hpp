/**
 * @file chunking.hpp
 * @author WASL
 * @brief Declarations for all chunking techniques
 * @version 0.1
 * @date 2023-01-31
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#ifndef _COMMON_CHUNKING_
#define _COMMON_CHUNKING_

#include <chrono>
#include <string>
#include <vector>
#include <memory>
#include <sstream>
#include <istream>
#include <cstdint>
#include <iostream>
#include <functional>

#include "hash.hpp"
#include "config.hpp"
#include "file_chunk.hpp"
#include "hashing_common.hpp"

class Chunking_Technique{
    /**
     * @brief Interface for all chunking techniques
     * 
     */

    protected:
        /**
         * Read exactly byte_count bytes into a preallocated buffer using
         * bounded requests that are safe across supported standard-library
         * implementations.
         */
        static bool read_exact(std::istream& stream, char* buffer,
                               uint64_t byte_count);

        /**
         * @brief a helper function to create a chunk from a data buffer
         * @param file_chunks: chunks vector
         * @param data: the data buffer
         * @param buffer_end: the logical size of the buffer in bytes
         * @return: size of the chunk
         */
        int64_t create_chunk(std::vector<std::string>& hashes, char* data, uint64_t buffer_end);

        // Whether the buffer passed to find_cutpoint holds the end of the
        // input, set by chunk_stream before each call. Lets a chunker tell a
        // short final buffer apart from a full one.
        bool input_ends_in_buffer = false;

    public:
        std::string technique_name;
        std::unique_ptr<Hashing_Technique> hash_method;
        uint64_t stream_buffer_size;
        std::chrono::duration<double, std::milli> total_time_chunking =
        std::chrono::duration<double, std::milli>::zero();
        std::chrono::duration<double, std::milli> total_time_hashing =
        std::chrono::duration<double, std::milli>::zero();
        /**
         * @brief Chunk a buffer using a chunking technique and return a single chunk boundary from this operation
         * 
         * @param buffer: Data stream of bytes
         * @param buffer_size: Size of buffer
         * @return: uint64_t indicating boundary position
         */
        virtual uint64_t find_cutpoint(char* buffer, uint64_t buffer_size){
            if(buffer != nullptr)
                return buffer_size;
            else
                std::cout << "Null buffer received" << std::endl; 
            return 0;
        }

        /**
         * @brief calculates the size of the given file
         * 
         * @param file_ptr: ifstream pointer to the file   
         * @return: the size of the file
         */
        uint64_t get_file_size(std::istream* file_ptr);

        /**
         * @brief Chunk a file using a chunking technique and return the struct File_Chunks from this operation
         * 
         * @param file_path: String containing path to file
         * @return: Vector of struct File_Chunk
         */
        virtual std::vector<std::string> chunk_file(std::string file_path);

        using Chunk_Result_Consumer = std::function<void(const std::string&)>;

        virtual uint64_t chunk_file(
            std::string file_path, const Chunk_Result_Consumer& consumer);
        /**
         * @brief Chunk a stream using a chunking technique and append the struct File_Chunks from this operation
         * to the vector passed in
         * 
         * @param stream: input stream containing the data to be chunked
         * @return: void
         */
        virtual void chunk_stream(std::vector<std::string>& hashes, std::istream& stream);

        /**
         * @brief Chunk data that is already materialized in a contiguous buffer.
         *
         * Used for in-memory datasets (e.g. decompressed .xz) so that a
         * chunker can process the buffer directly without an extra copy. The
         * default implementation reuses chunk_stream() over a non-owning view
         * of the buffer; whole-file/windowed SIMD chunkers override it.
         *
         * @param hashes: Output vector to append the chunk hashes to
         * @param data: Pointer to the data to chunk (not owned by this object)
         * @param size: Number of bytes in `data`
         */
        virtual void chunk_buffer(std::vector<std::string>& hashes,
                                  const unsigned char* data, uint64_t size);

        virtual ~Chunking_Technique() {};

        /**
         * @brief Read a file from disk and store it in memory buffer
         * 
         * @param file_path : String containing path to the file. Must be a valid file path
         * @return std::istringstream
         */
        static std::unique_ptr<std::istream> read_file_to_buffer(std::string file_path);
};

#endif
