#include "gong_gfni.h"

#include <assert.h>
#include <immintrin.h>
#include <stdint.h>

#include "wbsm4_se_local.h"

static inline __m128i pack_m8_block_column(__m256i row_block, __m256i mask)
{
    const __m256i packed = _mm256_shuffle_epi8(row_block, mask);
    const __m128i lo = _mm256_castsi256_si128(packed);
    const __m128i hi = _mm256_extracti128_si256(packed, 1);

    return _mm_unpacklo_epi32(lo, hi);
}

static inline __m256i pack_m8_block_matrices(__m256i row_block)
{
    const __m256i mask3 = _mm256_setr_epi8(
        3, 7, 11, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        3, 7, 11, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i mask2 = _mm256_setr_epi8(
        2, 6, 10, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        2, 6, 10, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i mask1 = _mm256_setr_epi8(
        1, 5, 9, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        1, 5, 9, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i mask0 = _mm256_setr_epi8(
        0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m128i a0 = pack_m8_block_column(row_block, mask3);
    const __m128i a1 = pack_m8_block_column(row_block, mask2);
    const __m128i a2 = pack_m8_block_column(row_block, mask1);
    const __m128i a3 = pack_m8_block_column(row_block, mask0);
    const __m128i lo = _mm_unpacklo_epi64(a0, a1);
    const __m128i hi = _mm_unpacklo_epi64(a2, a3);

    return _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
}

static inline uint8_t xor_lane_low_bytes(__m256i y)
{
    const __m256i pair_xor = _mm256_xor_si256(y, _mm256_srli_si256(y, 8));
    const uint8_t lo = (uint8_t)_mm_cvtsi128_si32(_mm256_castsi256_si128(pair_xor));
    const uint8_t hi = (uint8_t)_mm_cvtsi128_si32(_mm256_extracti128_si256(pair_xor, 1));

    return (uint8_t)(lo ^ hi);
}





void gong_affineU32_gfni_prepare(const Aff32 *aff, GongAff32PreparedGFNI *prepared)
{
    for (int block = 0; block < 4; block++) {
        const __m256i row_block = _mm256_loadu_si256((const __m256i *)&aff->Mat.M[block * 8]);
        prepared->mats[block] = pack_m8_block_matrices(row_block);
    }
    prepared->vec = aff->Vec.V;
}

uint32_t gong_affineU32_gfni_prepared(const GongAff32PreparedGFNI *prepared, uint32_t arr)
{
    const __m256i x = _mm256_set_epi64x((long long)(uint8_t)arr,
                                        (long long)(uint8_t)(arr >> 8),
                                        (long long)(uint8_t)(arr >> 16),
                                        (long long)(uint8_t)(arr >> 24));
    uint32_t out = 0;

    const __m256i yv0 = _mm256_gf2p8affine_epi64_epi8(x, prepared->mats[0], 0);
    out |= (uint32_t)xor_lane_low_bytes(yv0) << 24;

    const __m256i yv1 = _mm256_gf2p8affine_epi64_epi8(x, prepared->mats[1], 0);
    out |= (uint32_t)xor_lane_low_bytes(yv1) << 16;

    const __m256i yv2 = _mm256_gf2p8affine_epi64_epi8(x, prepared->mats[2], 0);
    out |= (uint32_t)xor_lane_low_bytes(yv2) << 8;

    const __m256i yv3 = _mm256_gf2p8affine_epi64_epi8(x, prepared->mats[3], 0);
    out |= (uint32_t)xor_lane_low_bytes(yv3);

    return out ^ prepared->vec;
}



static inline uint64_t xor_lane_bytes_x8(__m256i y)
{
    const __m256i pair_xor = _mm256_xor_si256(y, _mm256_srli_si256(y, 8));
    const __m128i lo = _mm256_castsi256_si128(pair_xor);
    const __m128i hi = _mm256_extracti128_si256(pair_xor, 1);
    const __m128i all_xor = _mm_xor_si128(lo, hi);

    return (uint64_t)_mm_cvtsi128_si64(all_xor);
}



static inline __m256i pack_input_bytes_x8_avxpack(const uint32_t in[8])
{
    const __m256i words = _mm256_loadu_si256((const __m256i *)in);
    const __m256i shuf = _mm256_shuffle_epi8(words, _mm256_setr_epi8(
        3, 7, 11, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        3, 7, 11, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
    const __m256i shuf1 = _mm256_shuffle_epi8(words, _mm256_setr_epi8(
        2, 6, 10, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        2, 6, 10, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
    const __m256i shuf2 = _mm256_shuffle_epi8(words, _mm256_setr_epi8(
        1, 5, 9, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        1, 5, 9, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
    const __m256i shuf3 = _mm256_shuffle_epi8(words, _mm256_setr_epi8(
        0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));

    const uint64_t b0 = (uint64_t)(uint32_t)_mm_cvtsi128_si32(_mm256_castsi256_si128(shuf)) |
                        ((uint64_t)(uint32_t)_mm_cvtsi128_si32(_mm256_extracti128_si256(shuf, 1)) << 32);
    const uint64_t b1 = (uint64_t)(uint32_t)_mm_cvtsi128_si32(_mm256_castsi256_si128(shuf1)) |
                        ((uint64_t)(uint32_t)_mm_cvtsi128_si32(_mm256_extracti128_si256(shuf1, 1)) << 32);
    const uint64_t b2 = (uint64_t)(uint32_t)_mm_cvtsi128_si32(_mm256_castsi256_si128(shuf2)) |
                        ((uint64_t)(uint32_t)_mm_cvtsi128_si32(_mm256_extracti128_si256(shuf2, 1)) << 32);
    const uint64_t b3 = (uint64_t)(uint32_t)_mm_cvtsi128_si32(_mm256_castsi256_si128(shuf3)) |
                        ((uint64_t)(uint32_t)_mm_cvtsi128_si32(_mm256_extracti128_si256(shuf3, 1)) << 32);

    return _mm256_set_epi64x((long long)b3, (long long)b2, (long long)b1, (long long)b0);
}

static inline void unpack_output_bytes_x8_avxpack(uint64_t y0, uint64_t y1, uint64_t y2, uint64_t y3,
                                                  uint32_t vec, uint32_t out[8])
{
    const __m128i b0 = _mm_cvtsi64_si128((long long)y0);
    const __m128i b1 = _mm_cvtsi64_si128((long long)y1);
    const __m128i b2 = _mm_cvtsi64_si128((long long)y2);
    const __m128i b3 = _mm_cvtsi64_si128((long long)y3);
    const __m128i lo01 = _mm_unpacklo_epi8(b3, b2);
    const __m128i lo23 = _mm_unpacklo_epi8(b1, b0);
    const __m128i words_lo = _mm_unpacklo_epi16(lo01, lo23);
    const __m128i words_hi = _mm_unpackhi_epi16(lo01, lo23);
    const __m256i words = _mm256_inserti128_si256(_mm256_castsi128_si256(words_lo), words_hi, 1);
    const __m256i vx = _mm256_set1_epi32((int)vec);

    _mm256_storeu_si256((__m256i *)out, _mm256_xor_si256(words, vx));
}

void gong_affineU32_gfni_prepared_x8_avxpack(const GongAff32PreparedGFNI *prepared, const uint32_t in[8], uint32_t out[8])
{
    const __m256i x = pack_input_bytes_x8_avxpack(in);

    const uint64_t y0 = xor_lane_bytes_x8(_mm256_gf2p8affine_epi64_epi8(x, prepared->mats[0], 0));
    const uint64_t y1 = xor_lane_bytes_x8(_mm256_gf2p8affine_epi64_epi8(x, prepared->mats[1], 0));
    const uint64_t y2 = xor_lane_bytes_x8(_mm256_gf2p8affine_epi64_epi8(x, prepared->mats[2], 0));
    const uint64_t y3 = xor_lane_bytes_x8(_mm256_gf2p8affine_epi64_epi8(x, prepared->mats[3], 0));

    unpack_output_bytes_x8_avxpack(y0, y1, y2, y3, prepared->vec, out);
}

typedef struct GongPackedU32x8 {
    uint64_t b0;
    uint64_t b1;
    uint64_t b2;
    uint64_t b3;
} GongPackedU32x8;









void wbsm4_se_prepare_gfni_context(const wbsm4_se_context *ctx, GongWbsm4SePreparedGFNI *prepared)
{
    for (int i = 0; i < 32; i++) {
        gong_affineU32_gfni_prepare(&ctx->M[i][0], &prepared->M[i][0]);
        gong_affineU32_gfni_prepare(&ctx->M[i][1], &prepared->M[i][1]);
        gong_affineU32_gfni_prepare(&ctx->M[i][2], &prepared->M[i][2]);
        gong_affineU32_gfni_prepare(&ctx->C[i], &prepared->C[i]);
        gong_affineU32_gfni_prepare(&ctx->D[i], &prepared->D[i]);
    }

    for (int i = 0; i < 4; i++) {
        gong_affineU32_gfni_prepare(&ctx->SE[i], &prepared->SE[i]);
        gong_affineU32_gfni_prepare(&ctx->FE[i], &prepared->FE[i]);
    }
}



int wbsm4_se_encrypt_gfni_prepared(const unsigned char *input, unsigned char *output, const GongWbsm4SePreparedGFNI *prepared)
{
    uint32_t x0 = GET32(input);
    uint32_t x1 = GET32(input + 4);
    uint32_t x2 = GET32(input + 8);
    uint32_t x3 = GET32(input + 12);

    x0 = gong_affineU32_gfni_prepared(&prepared->SE[0], x0);
    x1 = gong_affineU32_gfni_prepared(&prepared->SE[1], x1);
    x2 = gong_affineU32_gfni_prepared(&prepared->SE[2], x2);
    x3 = gong_affineU32_gfni_prepared(&prepared->SE[3], x3);

    for (int i = 0; i < 32; i++) {
        uint32_t xt1 = gong_affineU32_gfni_prepared(&prepared->M[i][0], x1);
        uint32_t xt2 = gong_affineU32_gfni_prepared(&prepared->M[i][1], x2);
        uint32_t xt3 = gong_affineU32_gfni_prepared(&prepared->M[i][2], x3);
        uint32_t x4 = xt1 ^ xt2 ^ xt3;

        x4 = (sm4_sbox[(x4 >> 24) & 0xff] << 24) |
             (sm4_sbox[(x4 >> 16) & 0xff] << 16) |
             (sm4_sbox[(x4 >> 8) & 0xff] << 8) |
             sm4_sbox[x4 & 0xff];

        uint32_t xt0 = gong_affineU32_gfni_prepared(&prepared->C[i], x0);
        uint32_t xt4 = gong_affineU32_gfni_prepared(&prepared->D[i], x4);
        x4 = xt0 ^ xt4;

        x0 = x1;
        x1 = x2;
        x2 = x3;
        x3 = x4;
    }

    x0 = gong_affineU32_gfni_prepared(&prepared->FE[3], x0);
    x1 = gong_affineU32_gfni_prepared(&prepared->FE[2], x1);
    x2 = gong_affineU32_gfni_prepared(&prepared->FE[1], x2);
    x3 = gong_affineU32_gfni_prepared(&prepared->FE[0], x3);

    PUT32(x3, output);
    PUT32(x2, output + 4);
    PUT32(x1, output + 8);
    PUT32(x0, output + 12);

    return 1;
}



int wbsm4_se_encrypt_gfni_prepared_x8_avxpack(const unsigned char *input, unsigned char *output, const GongWbsm4SePreparedGFNI *prepared)
{
    uint32_t x0[8], x1[8], x2[8], x3[8];
    uint32_t xt0[8], xt1[8], xt2[8], xt3[8], xt4[8], x4[8];

    for (int block = 0; block < 8; block++) {
        const unsigned char *src = input + block * 16;
        x0[block] = GET32(src);
        x1[block] = GET32(src + 4);
        x2[block] = GET32(src + 8);
        x3[block] = GET32(src + 12);
    }

    gong_affineU32_gfni_prepared_x8_avxpack(&prepared->SE[0], x0, x0);
    gong_affineU32_gfni_prepared_x8_avxpack(&prepared->SE[1], x1, x1);
    gong_affineU32_gfni_prepared_x8_avxpack(&prepared->SE[2], x2, x2);
    gong_affineU32_gfni_prepared_x8_avxpack(&prepared->SE[3], x3, x3);

    for (int i = 0; i < 32; i++) {
        gong_affineU32_gfni_prepared_x8_avxpack(&prepared->M[i][0], x1, xt1);
        gong_affineU32_gfni_prepared_x8_avxpack(&prepared->M[i][1], x2, xt2);
        gong_affineU32_gfni_prepared_x8_avxpack(&prepared->M[i][2], x3, xt3);

        for (int block = 0; block < 8; block++) {
            x4[block] = xt1[block] ^ xt2[block] ^ xt3[block];
            x4[block] = (sm4_sbox[(x4[block] >> 24) & 0xff] << 24) |
                        (sm4_sbox[(x4[block] >> 16) & 0xff] << 16) |
                        (sm4_sbox[(x4[block] >> 8) & 0xff] << 8) |
                        sm4_sbox[x4[block] & 0xff];
        }

        gong_affineU32_gfni_prepared_x8_avxpack(&prepared->C[i], x0, xt0);
        gong_affineU32_gfni_prepared_x8_avxpack(&prepared->D[i], x4, xt4);

        for (int block = 0; block < 8; block++) {
            x4[block] = xt0[block] ^ xt4[block];
            x0[block] = x1[block];
            x1[block] = x2[block];
            x2[block] = x3[block];
            x3[block] = x4[block];
        }
    }

    gong_affineU32_gfni_prepared_x8_avxpack(&prepared->FE[3], x0, x0);
    gong_affineU32_gfni_prepared_x8_avxpack(&prepared->FE[2], x1, x1);
    gong_affineU32_gfni_prepared_x8_avxpack(&prepared->FE[1], x2, x2);
    gong_affineU32_gfni_prepared_x8_avxpack(&prepared->FE[0], x3, x3);

    for (int block = 0; block < 8; block++) {
        unsigned char *dst = output + block * 16;
        PUT32(x3[block], dst);
        PUT32(x2[block], dst + 4);
        PUT32(x1[block], dst + 8);
        PUT32(x0[block], dst + 12);
    }

    return 1;
}









int wbsm4_se_ecb_encrypt_gfni_prepared_x8_avxpack(const unsigned char *input, unsigned char *output, size_t len, const GongWbsm4SePreparedGFNI *prepared)
{
    assert(len % 16 == 0);
    while (len >= 128) {
        wbsm4_se_encrypt_gfni_prepared_x8_avxpack(input, output, prepared);
        len -= 128;
        input += 128;
        output += 128;
    }
    while (len) {
        wbsm4_se_encrypt_gfni_prepared(input, output, prepared);
        len -= 16;
        input += 16;
        output += 16;
    }
    return 1;
}






