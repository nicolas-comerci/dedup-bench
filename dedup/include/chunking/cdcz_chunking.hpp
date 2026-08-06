#ifndef _CDCZ_CHUNKING_
#define _CDCZ_CHUNKING_

#include "chunking_common.hpp"
#include "config.hpp"
#include "gear_chunking.hpp"

class Cdcz_Chunking : public virtual Chunking_Technique {
    /**
     * @brief Class implementing gear's based chunking
     *
     */

   private:
    uint64_t min_block_size;
    uint64_t max_block_size;
    uint64_t avg_block_size;
    uint64_t mask;
    SIMD_Mode simd_mode;
    bool use_64bit_gear;

    // for ss-cdc
    uint64_t file_size;
    uint8_t* cutpoint_bitmap = nullptr;
    chunk_boundary cb{};
    uint64_t optimization_level;



    /**
     * @brief calculate the next gear hash
     * @param h the current hash value
     * @param ch the next byte to add
     * @return the new hash value
     */
    /**
     * @brief finds the next cut point in an array of bytes
     * @param buff: the buff to find the cutpoint in.
     * @param size: the size of the buffer
     * @return: cutpoint position in the buffer 
     */
    uint64_t find_cutpoint(char* buff, uint64_t size) override;

    uint64_t find_cutpoint_serial(char* buff, uint64_t size);
    uint64_t find_cutpoint_avx512(char* buff, uint64_t size);


   public:
    /**
     * @brief Default constructor. defines all parameters to defualt values
     * @return: void
     */
    Cdcz_Chunking();

    /**
     * @brief Defines all parameters based on values from the config file
     * @return: void
     */
    Cdcz_Chunking(const Config& config);

    /**
     * @brief chunk a file using gear hasing
     * @param data Data stream to chunk.
     * @return A vector of File_Chuncks
     */
    std::vector<std::string> chunk_file(std::string file_path) override;
};

#endif
