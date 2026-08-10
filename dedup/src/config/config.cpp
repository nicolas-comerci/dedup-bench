#include "config.hpp"

#include <sstream>
#include <string>

#include "config_error.hpp"
#include "parser.hpp"

Config::Config(std::string config_file_path) : parser{config_file_path} {}

ChunkingTech Config::get_chunking_tech() const {
    try {
        std::string value = parser.get_property(CHUNKING_TECH);
        if (value == "fixed") {
            return ChunkingTech::FIXED;
        } else if (value == "rabins") {
            return ChunkingTech::RABINS;
        } else if (value == "ae") {
            return ChunkingTech::AE;
        } else if (value == "gear") {
            return ChunkingTech::GEAR;
        } else if (value == "prefix_sum") {
            return ChunkingTech::PREFIX_SUM;
        } else if (value == "fastcdc") {
            return ChunkingTech::FASTCDC;
        } else if (value == "supercdc") {
            return ChunkingTech::SUPERCDC;
        } else if (value == "cdcz") {
            return ChunkingTech::CDCZ;
        } else if (value == "ram") {
            return ChunkingTech::RAM;
        } else if(value == "maxp"){
            return ChunkingTech::MAXP;
        } else if (value == "crc") {
            return ChunkingTech::CRC;
        } else if (value == "seq") {
            return ChunkingTech::SEQ;
        } else if (value == "tttd"){
            return ChunkingTech::TTTD;
        }
    } catch (...) {
    }
    throw ConfigError(
        "The configuration file does not specify a valid chunking technique");
}

HashingTech Config::get_hashing_tech() const {
    try {
        std::string value = parser.get_property(HASHING_TECH);
        if (value == "sha1") {
            return HashingTech::SHA1;
        } else if (value == "sha256") {
            return HashingTech::SHA256;
        } else if (value == "sha512") {
            return HashingTech::SHA512;
        } else if (value == "md5") {
            return HashingTech::MD5;
        } else if (value == "xxhash128") {
            return HashingTech::XXHASH128;
        } else if (value == "murmurhash3") {
            return HashingTech::MURMURHASH3;
        } 
    } catch (...) {
    }
    throw ConfigError(
        "The configuration file does not specify a valid hashing technique");
}

SIMD_Mode Config::get_simd_mode() const {
    try {
        std::string value = parser.get_property(SIMD_MODE_STRING);
        if (value == "none") {
            return SIMD_Mode::NONE;
        } 
        #ifdef __SSE3__
            else if (value == "sse128") {
                return SIMD_Mode::SSE128;
            } 
            else if (value == "sse128_noslide"){
                return SIMD_Mode::SSE128_NOSLIDE;
            }
        #endif
        
        #ifdef __AVX2__
            else if (value == "avx256") {
                return SIMD_Mode::AVX256;
            }
        #endif
        
        #ifdef __AVX512F__
            else if (value == "avx512") {
                return SIMD_Mode::AVX512;
            }
        #endif
        #ifdef __ARM_NEON
            else if (value == "neon128") {
                return SIMD_Mode::NEON;
            }
        #endif
        #ifdef __ALTIVEC__
            else if (value == "altivec128") {
                return SIMD_Mode::ALTIVEC;
            }
        #endif
        else {
            throw ConfigError("Unsupported SIMD mode");
        }
    } catch (...) {
    }
    throw ConfigError(
        "Unsupported SIMD mode. Please check compilation flags and configuration file.");
}

uint64_t Config::get_fc_size() const {
    try {
        std::string value = parser.get_property(FC_SIZE);
        return std::stoull(value);
    } catch (...) {
    }
    throw ConfigError(
        "The configuration file does not specify a valid size for fixed size "
        "chunking");
}

uint64_t Config::get_rabinc_window_size() const {
    try {
        std::string value = parser.get_property(RABINC_WINDOW_SIZE);
        return std::stoull(value);
    } catch (...) {
    }
    throw ConfigError(
        "The configuration file does not specify a valid size for the sliding "
        "window");
}

uint64_t Config::get_rabinc_min_block_size() const {
    try {
        std::string value = parser.get_property(RABINC_MIN_BLOCK_SIZE);
        return std::stoull(value);
    } catch (...) {
    }
    throw ConfigError(
        "The configuration file does not specify a valid minimum block size");
}

uint64_t Config::get_rabinc_avg_block_size() const {
    try {
        std::string value = parser.get_property(RABINC_AVG_BLOCK_SIZE);
        return std::stoull(value);
    } catch (...) {
    }
    throw ConfigError(
        "The configuration file does not specify a valid avarage block size");
}

uint64_t Config::get_rabinc_max_block_size() const {
    try {
        std::string value = parser.get_property(RABINC_MAX_BLOCK_SIZE);
        return std::stoull(value);
    } catch (...) {
    }
    throw ConfigError(
        "The configuration file does not specify a valid maximum block size");
}

std::string Config::get_output_file() const {
    try {
        std::string value = parser.get_property(OUTPUT_FILE);
        return value;
    } catch (...) {
    }
    return "hashes.out";
}

uint64_t Config::get_ae_avg_block_size() const {
    try {
        std::string value = parser.get_property(AE_AVG_BLOCK_SIZE);
        return std::stoull(value);
    } catch (...) {
    }
    throw ConfigError(
        "The configuration file does not specify a valid AE avarage block "
        "size");
}

AE_Mode Config::get_ae_extreme_mode() const {
    try {
        std::string value = parser.get_property(AE_EXTREME_MODE);
        if (value == "min") {
            return AE_Mode::MIN;
        } else if (value == "max") {
            return AE_Mode::MAX;
        }
    } catch (...) {
    }
    throw ConfigError(
        "The configuration file does not specify a valid AE extreme mode");
}

uint64_t Config::get_gear_min_block_size() const {
    try {
        std::string value = parser.get_property(GEAR_MIN_BLOCK_SIZE);
        return std::stoull(value);
    } catch (...) {
    }
    throw ConfigError(
        "The configuration file does not specify a valid gear minimum block "
        "size for gear hash");
}

uint64_t Config::get_gear_max_block_size() const {
    try {
        std::string value = parser.get_property(GEAR_MAX_BLOCK_SIZE);
        return std::stoull(value);
    } catch (...) {
    }
    throw ConfigError(
        "The configuration file does not specify a valid gear maximum block "
        "size for gear hash");
}

bool Config::get_use_64bit_gear() const {
    std::string value;
    try {
        value = parser.get_property(USE_64BIT_GEAR);
    } catch (...) {
        return false;
    }

    if (value == "true") {
        return true;
    }
    if (value == "false") {
        return false;
    }
    throw ConfigError(
        "The use_64bit_gear option must be either 'true' or 'false'");
}

bool Config::get_gear_use_low_bit_mask() const {
    std::string value;
    try {
        value = parser.get_property(GEAR_USE_LOW_BIT_MASK);
    } catch (...) {
        return false;
    }
    if (value == "true") {
        return true;
    }
    if (value == "false") {
        return false;
    }
    throw ConfigError(
        "The gear_use_low_bit_mask option must be either 'true' or 'false'");
}

uint64_t Config::get_prefix_sum_min_block_size() const {
    try {
        return std::stoull(parser.get_property(PREFIX_SUM_MIN_BLOCK_SIZE));
    } catch (...) {
    }
    throw ConfigError(
        "The configuration file does not specify a valid prefix-sum minimum block size");
}

uint64_t Config::get_prefix_sum_avg_block_size() const {
    try {
        return std::stoull(parser.get_property(PREFIX_SUM_AVG_BLOCK_SIZE));
    } catch (...) {
    }
    throw ConfigError(
        "The configuration file does not specify a valid prefix-sum average block size");
}

uint64_t Config::get_prefix_sum_max_block_size() const {
    try {
        return std::stoull(parser.get_property(PREFIX_SUM_MAX_BLOCK_SIZE));
    } catch (...) {
    }
    throw ConfigError(
        "The configuration file does not specify a valid prefix-sum maximum block size");
}

bool Config::get_use_64bit_prefix_sum() const {
    std::string value;
    try {
        value = parser.get_property(USE_64BIT_PREFIX_SUM);
    } catch (...) {
        return false;
    }
    if (value == "true") {
        return true;
    }
    if (value == "false") {
        return false;
    }
    throw ConfigError(
        "The use_64bit_prefix_sum option must be either 'true' or 'false'");
}

bool Config::get_use_gear_table_lookup() const {
    std::string value;
    try {
        value = parser.get_property(USE_GEAR_TABLE_LOOKUP);
    } catch (...) {
        return false;
    }
    if (value == "true") {
        return true;
    }
    if (value == "false") {
        return false;
    }
    throw ConfigError(
        "The use_gear_table_lookup option must be either 'true' or 'false'");
}

static bool get_optional_bool(const Parser& parser, const char* property,
                              bool default_value) {
    std::string value;
    try {
        value = parser.get_property(property);
    } catch (...) {
        return default_value;
    }
    if (value == "true") {
        return true;
    }
    if (value == "false") {
        return false;
    }
    throw ConfigError(std::string("The ") + property
                      + " option must be either 'true' or 'false'");
}

bool Config::get_prefix_sum_contribution_lookahead(bool default_value) const {
    return get_optional_bool(
        parser, PREFIX_SUM_CONTRIBUTION_LOOKAHEAD, default_value);
}

bool Config::get_use_subminimum_skipping() const {
    return get_optional_bool(parser, USE_SUBMINIMUM_SKIPPING, true);
}

bool Config::get_use_context_repair() const {
    return get_optional_bool(parser, USE_CONTEXT_REPAIR, true);
}

bool Config::get_use_supercdc_backup() const {
    return get_optional_bool(parser, USE_SUPERCDC_BACKUP, false);
}

uint64_t Config::get_sscdc_optimization_level() const {
    std::string value{};
    try {
        value = parser.get_property("optimization_level");
    } catch (...) {
        return 0;
    }
    try {
        return !value.empty() ? std::stoull(value) : 0;
    } catch (...) {
    }
    throw ConfigError(
        "The configuration file does not specify a valid sscdc optimization "
        "level");
}

uint64_t Config::get_gear_avg_block_size() const {
    try {
        std::string value = parser.get_property(GEAR_AVG_BLOCK_SIZE);
        return std::stoull(value);
    } catch (...) {
    }
    throw ConfigError(
        "The configuration file does not specify a valid gear maximum block "
        "size for gear hash");
}

uint64_t Config::get_fastcdc_min_block_size() const {
    try {
        std::string value = parser.get_property(FASTCDC_MIN_BLOCK_SIZE);
        return std::stoull(value);
    } catch (...) {
    }
    throw ConfigError(
        "The configuration file does not specify a valid fastcdc minimum block "
        "size");
}

uint64_t Config::get_fastcdc_max_block_size() const {
    try {
        std::string value = parser.get_property(FASTCDC_MAX_BLOCK_SIZE);
        return std::stoull(value);
    } catch (...) {
    }
    throw ConfigError(
        "The configuration file does not specify a valid fastcdc maximum block "
        "size");
}

uint64_t Config::get_fastcdc_avg_block_size() const {
    try {
        std::string value = parser.get_property(FASTCDC_AVG_BLOCK_SIZE);
        return std::stoull(value);
    } catch (...) {
    }
    throw ConfigError(
        "The configuration file does not specify a valid fastcdc avarge block "
        "size");
}

uint64_t Config::get_fastcdc_normalization_level() const {
    try {
        std::string value = parser.get_property(FASTCDC_NORMALIZATION_LEVEL);
        return std::stoull(value);
    } catch (...) {
    }
    throw ConfigError(
        "The configuration file does not specify a valid fastcdc normaliztion "
        "level");
}

uint64_t Config::get_optional_fastcdc_normalization_level() const {
    std::string disabled;
    try {
        disabled = parser.get_property(FASTCDC_DISABLE_NORMALIZATION);
    } catch (...) {
        return 0;
    }
    if (disabled == "true") {
        return 0;
    }
    if (disabled != "false") {
        throw ConfigError(
            "The fastcdc disable normalization option must be either 'true' "
            "or 'false'");
    }
    const uint64_t level = get_fastcdc_normalization_level();
    if (level < 1 || level > 3) {
        throw ConfigError(
            "The fastcdc normalization level must be between 1 and 3");
    }
    return level;
}

bool Config::get_fastcdc_disable_normalization() const {
    try {
        std::string value = parser.get_property(FASTCDC_DISABLE_NORMALIZATION);
        return value == "true" ? true : false;
    } catch (...) {
    }
    throw ConfigError(
        "The configuration file does not specify a valid fastcdc disable "
        "normalization option");
}

uint64_t Config::get_ram_avg_block_size() const {
    try {
        std::string value = parser.get_property(RAM_AVG_BLOCK_SIZE);
        return std::stoull(value);
    } catch (...) {
    }
    throw ConfigError(
        "The configuration file does not specify a valid RAM avarage block "
        "size");
}

uint64_t Config::get_ram_max_block_size() const {
    try {
        std::string value = parser.get_property(RAM_MAX_BLOCK_SIZE);
        return std::stoull(value);
    } catch (...) {
    }
    throw ConfigError(
        "The configuration file does not specify a valid RAM max block size");
}

uint64_t Config::get_crc_avg_block_size() const {
    try {
        std::string value = parser.get_property(CRC_AVG_BLOCK_SIZE);
        return std::stoull(value);
    } catch (...) {
    }
    throw ConfigError(
        "The configuration file does not specify a valid crc average block "
        "size");
}

uint64_t Config::get_crc_min_block_size() const {
    try {
        std::string value = parser.get_property(CRC_MIN_BLOCK_SIZE);
        return std::stoull(value);
    } catch (...) {
    }
    throw ConfigError(
        "The configuration file does not specify a valid crc minimum block "
        "size");
}

uint64_t Config::get_crc_max_block_size() const {
    try {
        std::string value = parser.get_property(CRC_MAX_BLOCK_SIZE);
        return std::stoull(value);
    } catch (...) {
    }
    throw ConfigError(
        "The configuration file does not specify a valid crc maximum block "
        "size");
}

uint64_t Config::get_crc_window_size() const {
    try {
        std::string value = parser.get_property(CRC_WINDOW_SIZE);
        return std::stoull(value);
    } catch (...) {
    }
    throw ConfigError(
        "The configuration file does not specify a valid crc window size");
}

uint64_t Config::get_crc_window_step() const {
    try {
        std::string value = parser.get_property(CRC_WINDOW_STEP_SIZE);
        return std::stoull(value);
    } catch (...) {
    }
    throw ConfigError(
        "The configuration file does not specify a valid crc window step size");
}

uint32_t Config::get_crc_hash_bits() const {
    try {
        std::string value = parser.get_property(CRC_HASH_BITS);
        return std::stoull(value);
    } catch (...) {
    }
    throw ConfigError(
        "The configuration file does not specify a valid crc hash bit number");
}

uint64_t Config::get_buffer_size() const {
    try {
        std::string value = parser.get_property(BUFFER_SIZE);
        return std::stoull(value);
    } catch (...) {
    }
    throw ConfigError(
        "The configuration file does not specify a valid buffer size");
}

uint64_t Config::get_seq_jump_trigger() const {
    try {
        std::string value = parser.get_property(SEQ_JUMP_TRIGGER);
        return std::stoull(value);
    } catch (...) {
    }
    throw ConfigError(
        "The configuration file does not specify a valid seq jump trigger "
        "value");
}

uint64_t Config::get_seq_threshold() const {
    try {
        std::string value = parser.get_property(SEQ_CHUNK_BOUNDARY_THRESHOLD);
        return std::stoull(value);
    } catch (...) {
    }
    throw ConfigError(
        "The configuration file does not specify a valid seq sequence "
        "threshold");
}

uint64_t Config::get_seq_min_block_size() const {
    try {
        std::string value = parser.get_property(SEQ_MIN_BLOCK_SIZE);
        return std::stoull(value);
    } catch (...) {
    }
    throw ConfigError(
        "The configuration file does not specify a valid seq minimum block "
        "size");
}

uint64_t Config::get_seq_max_block_size() const {
    try {
        std::string value = parser.get_property(SEQ_MAX_BLOCK_SIZE);
        return std::stoull(value);
    } catch (...) {
    }
    throw ConfigError(
        "The configuration file does not specify a valid seq maximum block "
        "size");
}

uint64_t Config::get_seq_avg_block_size() const {
    try {
        std::string value = parser.get_property(SEQ_AVG_BLOCK_SIZE);
        return std::stoull(value);
    } catch (...) {
    }
    throw ConfigError(
        "The configuration file does not specify a valid seq average block "
        "size");
}

uint64_t Config::get_seq_jump_size() const {
    try {
        std::string value = parser.get_property(SEQ_JUMP_SIZE);
        return std::stoull(value);
    } catch (...) {
    }
    throw ConfigError(
        "The configuration file does not specify a valid max seq jump size");
}

Seq_Op_Mode Config::get_seq_op_mode() const {
    try {
        std::string value = parser.get_property(SEQ_OPERATION_MODE);
        if (value == "inc")
            return Seq_Op_Mode::INCREASING;
        else if (value == "dec")
            return Seq_Op_Mode::DECREASING;
    } catch (...) {
    }
    throw ConfigError(
        "The configuration file does not specify a valid seq operation mode");
}

uint64_t Config::get_exp_window_size() const {
    try {
        std::string value = parser.get_property(EXP_WINDOW_SIZE);
        return std::stoull(value);
    } catch (...) {
    }
    throw ConfigError(
        "The configuration file does not specify a valid window size");
}

uint64_t Config::get_exp_little_window_size() const {
    try {
        std::string value = parser.get_property(EXP_LITTLE_WINDOW_SIZE);
        return std::stoull(value);
    } catch (...) {
    }
    throw ConfigError(
        "The configuration file does not specify a valid little window size");
}

uint64_t Config::get_exp_jump_threshold() const {
    try {
        std::string value = parser.get_property(EXP_JUMP_THRESHOLD);
        return std::stoull(value);
    } catch (...) {
    }
    throw ConfigError(
        "The configuration file does not specify a valid jump threshold");
}

uint64_t Config::get_exp_jump_amount() const {
    try {
        std::string value = parser.get_property(EXP_JUMP_AMOUNT);
        return std::stoull(value);
    } catch (...) {
    }
    throw ConfigError(
        "The configuration file does not specify a valid jump amount");
}

uint64_t Config::get_exp_min_block_size() const {
    try {
        std::string value = parser.get_property(EXP_MIN_BLOCK_SIZE);
        return std::stoull(value);
    } catch (...) {
    }
    throw ConfigError(
        "The configuration file does not specify a valid min block size");
}

uint64_t Config::get_tttd_avg_block_size() const {
    try {
        std::string value = parser.get_property(TTTD_AVG_BLOCK_SIZE);
        return std::stoull(value);
    } catch (...) {
    }
    throw ConfigError(
        "The configuration file does not specify a valid TTTD average block "
        "size");
}

uint64_t Config::get_tttd_min_block_size() const {
    try {
        std::string value = parser.get_property(TTTD_MIN_BLOCK_SIZE);
        return std::stoull(value);
    } catch (...) {
    }
    throw ConfigError(
        "The configuration file does not specify a valid TTTD min block "
        "size");
}

uint64_t Config::get_tttd_max_block_size() const {
    try {
        std::string value = parser.get_property(TTTD_MAX_BLOCK_SIZE);
        return std::stoull(value);
    } catch (...) {
    }
    throw ConfigError(
        "The configuration file does not specify a valid TTTD max block "
        "size");
}

uint64_t Config::get_maxp_window_size() const {
    try {
        std::string value = parser.get_property(MAXP_WINDOW_SIZE);
        return std::stoull(value);
    } catch (...) {
    }
    throw ConfigError(
        "The configuration file does not specify a valid MAXP window "
        "size");
}

uint64_t Config::get_maxp_max_block_size() const {
    try {
        std::string value = parser.get_property(MAXP_MAX_BLOCK_SIZE);
        return std::stoull(value);
    } catch (...) {
    }
    throw ConfigError(
        "The configuration file does not specify a valid MAXP maximum chunk "
        "size");
}
