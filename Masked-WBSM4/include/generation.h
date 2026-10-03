#ifndef MASKED_WBSM4_GENERATION_H
#define MASKED_WBSM4_GENERATION_H
#include "masked_wbsm4.h"
#include "masked_wbsm4_enhanced.h"
#ifdef __cplusplus
extern "C" {
#endif
/* decrypt: 0 for encryption parameters; 1 for decryption parameters. */
int masked_wbsm4_generate_direction(masked_wbsm4_context *ctx,
                          const uint8_t key[16],
                          uint32_t random_seed, int decrypt);
int masked_wbsm4_enhanced_generate_direction(masked_wbsm4_enhanced_context *ctx,
                                   const uint8_t key[16],
                                   uint32_t random_seed, int decrypt);
#ifdef __cplusplus
}
#endif
#endif
