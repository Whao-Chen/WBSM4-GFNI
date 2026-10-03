#ifndef WSISE_GFNI_SIMD_H
#define WSISE_GFNI_SIMD_H

#include <stdint.h>

void wsise_gfni_prepare(void);
void wsise_set_generation_seed(uint32_t seed);
int wsise_gfni_selftest(void);
void wsise_gfni_encrypt16(const uint8_t input[16 * 16],
                          uint8_t output[16 * 16]);

#endif
