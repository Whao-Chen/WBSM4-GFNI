#ifndef WSISE_WBSM4_ADAPTER_H
#define WSISE_WBSM4_ADAPTER_H

#include <stdint.h>

void wbsm4_adapter_generate(const uint8_t key[16]);
void wbsm4_adapter_encrypt(const uint8_t input[16], uint8_t output[16]);

#endif

