#ifndef BAIWU_GFNI_SIMD_H
#define BAIWU_GFNI_SIMD_H

#include <stdint.h>

void baiwu_gfni_prepare(void);
void baiwu_set_generation_seed(uint32_t seed);
void baiwu_gfni_encrypt16(const uint8_t input[16 * 16],
                          uint8_t output[16 * 16]);
void baiwu_interleaved_encrypt16(const uint8_t input[16 * 16],
                                 uint8_t output[16 * 16]);

#endif
