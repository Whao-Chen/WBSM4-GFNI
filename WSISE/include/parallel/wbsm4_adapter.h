#ifndef WSISE_WBSM4_ADAPTER_H
#define WSISE_WBSM4_ADAPTER_H

#include <stddef.h>
#include <stdint.h>

void wbsm4_adapter_generate(const uint8_t key[16]);
void wbsm4_adapter_generate_seeded(const uint8_t key[16], uint32_t seed);
void wbsm4_adapter_encrypt(const uint8_t input[16], uint8_t output[16]);
void wbsm4_adapter_encrypt_many(const uint8_t *input, uint8_t *output,
                                size_t block_count);
void wbsm4_adapter_encrypt_many_scalar(const uint8_t *input, uint8_t *output,
                                       size_t block_count);

#endif
