#ifndef XL_WBSM4_GENERATION_H
#define XL_WBSM4_GENERATION_H
#include "wbsm4.h"
#ifdef __cplusplus
extern "C" {
#endif
/* decrypt: 0 for encryption parameters; 1 for decryption parameters. */
void sm4_wb_gen_tables_direction(uint8_t *key, Sm4Whitebox *sm4_wb_ctx, int decrypt);
#ifdef __cplusplus
}
#endif
#endif
