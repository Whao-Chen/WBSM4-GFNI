#ifndef CHEN_SE_WBSM4_GENERATION_H
#define CHEN_SE_WBSM4_GENERATION_H
#include "wbsm4_se_cl.h"
#ifdef __cplusplus
extern "C" {
#endif
/* decrypt: 0 for encryption parameters; 1 for decryption parameters. */
int wbsm4_se_cl_gen_table_with_external_direction(const unsigned char *key, int external_encoding, int decrypt);
#ifdef __cplusplus
}
#endif
#endif
