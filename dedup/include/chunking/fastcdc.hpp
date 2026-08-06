#ifndef _FastCDC_CHUNKING_
#define _FastCDC_CHUNKING_

#include "chunking_common.hpp"
#include "config.hpp"

class FastCDC : public virtual Chunking_Technique {
    /**
     * @brief Class implementing FastCDC: gear's based chunking with
     * subminimum skipping and normalized chunking.
     *
     */

   private:
    uint64_t min_block_size;
    uint64_t max_block_size;
    uint64_t avg_block_size;
    uint64_t normalization_level;
    bool use_64bit_gear;

    uint64_t small_mask;
    uint64_t large_mask;

    /**
     * @brief finds the next cut point in an array of bytes
     * @param buff: the buff to find the cutpoint in.
     * @param size: the size of the buffer
     * @return: cutpoint position in the buffer
     */
    uint64_t find_cutpoint(char* buff, uint64_t size) override;


   public:
    /**
     * @brief Default constructor. defines all parameters to defualt values
     * @return: void
     */
    FastCDC();

    /**
     * @brief Defines all parameters based on values from the config file
     * @return: void
     */
    FastCDC(const Config& config);

    ~FastCDC();
};

#endif
