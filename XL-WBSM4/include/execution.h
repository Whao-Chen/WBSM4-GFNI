#ifndef XL_EXECUTION_H
#define XL_EXECUTION_H
#include "wbsm4.h"
void sm4_wb_enc(const uint8_t input[16], uint8_t output[16], Sm4Whitebox *context);
#endif
