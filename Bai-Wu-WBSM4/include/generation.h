#ifndef BAI_WU_WBSM4_GENERATION_H
#define BAI_WU_WBSM4_GENERATION_H
#include "wbsm4.h"
#ifdef __cplusplus
extern "C" {
#endif
/* decrypt: 0 for encryption parameters; 1 for decryption parameters. */
void wbsm4_gen_direction(uint8_t *key, int decrypt);
#ifdef __cplusplus
}
#endif
#endif
