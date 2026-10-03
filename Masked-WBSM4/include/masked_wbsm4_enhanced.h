#ifndef MASKED_WBSM4_ENHANCED_H
#define MASKED_WBSM4_ENHANCED_H

#include <stddef.h>
#include <stdint.h>

#ifndef MASKED_PRECOMPOSE_MASK_BRANCH
#define MASKED_PRECOMPOSE_MASK_BRANCH 0
#endif
#ifndef MASKED_PRECOMPOSE_MASK_LOOKUP_U8
#define MASKED_PRECOMPOSE_MASK_LOOKUP_U8 0
#endif
#ifndef MASKED_PIPELINE_32WAY
#define MASKED_PIPELINE_32WAY 0
#endif
#ifndef MASKED_FUSE_U8_MASK_AFFINE
#define MASKED_FUSE_U8_MASK_AFFINE 0
#endif

#if MASKED_PRECOMPOSE_MASK_BRANCH && MASKED_PRECOMPOSE_MASK_LOOKUP_U8
#error "Select at most one mask precomposition strategy"
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct masked_wbsm4_enhanced_context
    masked_wbsm4_enhanced_context;

enum {
    MASKED_WBSM4_ENHANCED_OK = 0,
    MASKED_WBSM4_ENHANCED_BAD_ARGUMENT = -1,
    MASKED_WBSM4_ENHANCED_NO_MEMORY = -2,
    MASKED_WBSM4_ENHANCED_NOT_GENERATED = -3,
    MASKED_WBSM4_ENHANCED_INVALID_TABLES = -4,
    MASKED_WBSM4_ENHANCED_GFNI_MISMATCH = -5,
    MASKED_WBSM4_ENHANCED_GFNI_TABLE_MISMATCH = -6,
    MASKED_WBSM4_ENHANCED_GFNI_MASK_BRANCH_MISMATCH = -7
};

masked_wbsm4_enhanced_context *masked_wbsm4_enhanced_create(void);
void masked_wbsm4_enhanced_destroy(masked_wbsm4_enhanced_context *ctx);

int masked_wbsm4_enhanced_generate(masked_wbsm4_enhanced_context *ctx,
                                   const uint8_t key[16],
                                   uint32_t random_seed);

int masked_wbsm4_enhanced_encrypt(
    const masked_wbsm4_enhanced_context *ctx,
    const uint8_t input[16],
    uint8_t output[16]);

int masked_wbsm4_enhanced_encrypt_many(
    const masked_wbsm4_enhanced_context *ctx,
    const uint8_t *input,
    uint8_t *output,
    size_t block_count);

int masked_wbsm4_enhanced_encrypt_many_scalar(
    const masked_wbsm4_enhanced_context *ctx,
    const uint8_t *input,
    uint8_t *output,
    size_t block_count);

int masked_wbsm4_enhanced_validate(
    const masked_wbsm4_enhanced_context *ctx);

size_t masked_wbsm4_enhanced_lookup_table_bytes(void);
size_t masked_wbsm4_enhanced_allocated_bytes(
    const masked_wbsm4_enhanced_context *ctx);

#ifdef __cplusplus
}
#endif

#endif
