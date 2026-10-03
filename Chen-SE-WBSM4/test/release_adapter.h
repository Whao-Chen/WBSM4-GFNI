#ifndef RELEASE_TEST_ADAPTER_H
#define RELEASE_TEST_ADAPTER_H
#include <stddef.h>
#include <stdint.h>
int release_generate(const uint8_t key[16], uint32_t seed, int decrypt);
int release_prepare(uint32_t seed, int decrypt);
int release_construct(const uint8_t key[16], uint32_t seed, int decrypt);
void release_execute(const uint8_t *input, uint8_t *output, size_t blocks);
void release_cleanup(void);
int release_extra_validation(void);
#endif
