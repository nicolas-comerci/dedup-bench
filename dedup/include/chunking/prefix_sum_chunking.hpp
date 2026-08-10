#ifndef _PREFIX_SUM_CHUNKING_
#define _PREFIX_SUM_CHUNKING_

#include "chunking_common.hpp"
#include "config.hpp"

class Prefix_Sum_Chunking : public virtual Chunking_Technique {
   private:
    uint64_t min_block_size;
    uint64_t max_block_size;
    uint64_t avg_block_size;
    uint64_t mask;
    uint64_t small_mask;
    uint64_t large_mask;
    uint64_t backup_mask;
    uint64_t normalization_level;
    bool use_64bit_prefix_sum;
    bool use_gear_table_lookup;
    bool use_subminimum_skipping;
    bool use_context_repair;
    bool use_supercdc_backup;
    bool contribution_lookahead;
    SIMD_Mode simd_mode;

    uint64_t find_cutpoint(char* data, uint64_t size) override;
    template <bool UseGearTable, bool SkipSubminimum, bool RepairContext,
              bool Normalize, bool UseBackup>
    uint64_t find_cutpoint_serial(
        const unsigned char* data, uint64_t size) const;

#if defined(__AVX2__)
    template <bool UseGearTable, bool SkipSubminimum, bool RepairContext,
              bool Lookahead, bool Normalize, bool UseBackup>
    uint64_t find_cutpoint_avx2(
        const unsigned char* data, uint64_t size) const;
#endif

#if defined(__AVX512F__)
    template <bool UseGearTable, bool SkipSubminimum, bool RepairContext,
              bool Lookahead, bool Normalize, bool UseBackup>
    uint64_t find_cutpoint_avx512(
        const unsigned char* data, uint64_t size) const;
#endif

   public:
    explicit Prefix_Sum_Chunking(const Config& config);
};

#endif
