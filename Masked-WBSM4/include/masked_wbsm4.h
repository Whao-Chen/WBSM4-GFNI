#ifndef MASKED_WBSM4_H
#define MASKED_WBSM4_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct masked_wbsm4_context masked_wbsm4_context;

enum {
    MASKED_WBSM4_OK = 0,
    MASKED_WBSM4_BAD_ARGUMENT = -1,
    MASKED_WBSM4_NO_MEMORY = -2,
    MASKED_WBSM4_NOT_GENERATED = -3,
    MASKED_WBSM4_INVALID_TABLES = -4
};

masked_wbsm4_context *masked_wbsm4_create(void);
void masked_wbsm4_destroy(masked_wbsm4_context *ctx);

int masked_wbsm4_generate(masked_wbsm4_context *ctx,
                          const uint8_t key[16],
                          uint32_t random_seed);

int masked_wbsm4_encrypt(const masked_wbsm4_context *ctx,
                         const uint8_t input[16],
                         uint8_t output[16]);
int masked_wbsm4_encrypt_many(const masked_wbsm4_context *ctx,
                              const uint8_t *input,
                              uint8_t *output,
                              size_t block_count);
int masked_wbsm4_encrypt_many_scalar(const masked_wbsm4_context *ctx,
                                     const uint8_t *input,
                                     uint8_t *output,
                                     size_t block_count);

int masked_wbsm4_validate(const masked_wbsm4_context *ctx);

size_t masked_wbsm4_lookup_table_bytes(void);
size_t masked_wbsm4_allocated_bytes(const masked_wbsm4_context *ctx);

#ifdef __cplusplus
}
#endif

#endif
