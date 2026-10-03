#ifndef GONG_GFNI_H
#define GONG_GFNI_H

#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>
#include "wbsm4_se.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GongAff32PreparedGFNI {
    __m256i mats[4];
    uint32_t vec;
} GongAff32PreparedGFNI;

typedef struct GongWbsm4SePreparedGFNI {
    GongAff32PreparedGFNI M[32][3];
    GongAff32PreparedGFNI C[32];
    GongAff32PreparedGFNI D[32];
    GongAff32PreparedGFNI SE[4];
    GongAff32PreparedGFNI FE[4];
} GongWbsm4SePreparedGFNI;




void gong_affineU32_gfni_prepare(const Aff32 *aff, GongAff32PreparedGFNI *prepared);
uint32_t gong_affineU32_gfni_prepared(const GongAff32PreparedGFNI *prepared, uint32_t arr);

void gong_affineU32_gfni_prepared_x8_avxpack(const GongAff32PreparedGFNI *prepared, const uint32_t in[8], uint32_t out[8]);
void wbsm4_se_prepare_gfni_context(const wbsm4_se_context *ctx, GongWbsm4SePreparedGFNI *prepared);
int wbsm4_se_encrypt_gfni_prepared(const unsigned char *input, unsigned char *output, const GongWbsm4SePreparedGFNI *prepared);



int wbsm4_se_encrypt_gfni_prepared_x8_avxpack(const unsigned char *input, unsigned char *output, const GongWbsm4SePreparedGFNI *prepared);
int wbsm4_se_ecb_encrypt_gfni_prepared_x8_avxpack(const unsigned char *input, unsigned char *output, size_t len, const GongWbsm4SePreparedGFNI *prepared);





#ifdef __cplusplus
}
#endif

#endif
