#include "internal.h"


void MatMulVecM8(M8 Mat, V8 Vec, V8 *ans);
void MatMulVecM4(M4 Mat, V4 Vec, V4 *ans);
static inline uint64_t m128_reverse64(uint64_t x);
static inline uint64_t m128_vpopcntdq_pairfold_half_testmask(const M128 *Mat, uint64_t v0, uint64_t v1, int row_base);
static inline void m128_vpopcntdq_pairfold_core(const M128 *Mat, const V128 *Vec, V128 *ans);
void MatMulVecM128(M128 Mat, V128 Vec, V128 *ans);
static inline uint16_t m16_gfni_lowbyte_core(const M16 *Mat, uint16_t n);
void MatMulVecM16(M16 Mat, V16 Vec, V16 *ans);
uint16_t MatMulNumM16(M16 Mat, uint16_t n);
static inline uint64_t m256_reverse64(uint64_t x);
static inline uint32_t m256_compress2_from_0x11(uint32_t folded);
static inline uint32_t m256_pairrows_qword_movmask_nopext(__m512i rows, __m512i x);
static inline uint64_t m256_vpopcntdq_pairfold_half_4rowpack_movmask(const M256 *Mat, const V256 *Vec, int row_base);
static inline void m256_vpopcntdq_pairfold_4rowpack_movmask_core(const M256 *Mat, const V256 *Vec, V256 *ans);
void MatMulVecM256(M256 Mat, V256 Vec, V256 *ans);
static inline uint32_t m32_reverse32(uint32_t x);
static inline uint32_t m32_avx512_vpopcntdq_bitreverse32_core(const M32 *Mat, uint32_t n);
uint32_t MatMulNumM32(M32 Mat, uint32_t n);
static inline uint32_t m32_avx512vl_vpopcntdq_ymm_core(const M32 *Mat, uint32_t n);
void MatMulVecM32(M32 Mat, V32 Vec, V32 *ans);
static inline uint8_t m4_gfni_nomask_core(const M4 *Mat, uint8_t n);
uint8_t MatMulNumM4(M4 Mat, uint8_t n);
static inline uint64_t m64_reverse64(uint64_t x);
static inline uint64_t m64_avx512_vpopcntdq_bitreverse_core(const M64 *Mat, uint64_t n);
uint64_t MatMulNumM64(M64 Mat, uint64_t n);
static inline uint64_t m64_avx512_vpopcntdq_movmask_core(const M64 *Mat, uint64_t n);
void MatMulVecM64(M64 Mat, V64 Vec, V64 *ans);
static inline uint8_t m8_gfni_u64load_core(const M8 *Mat, uint8_t n);
uint8_t MatMulNumM8(M8 Mat, uint8_t n);

void MatMulVecM8(M8 Mat, V8 Vec, V8 *ans)
{
                                                                        
                                                          
    __m128i mat_reg = _mm_loadl_epi64((__m128i*)&Mat.M[0]);
    
                                     
                                                                            
    __m128i vec_reg = _mm_cvtsi32_si128(Vec.V);
    
                                                                     
    __m128i result = _mm_gf2p8affine_epi64_epi8(vec_reg, mat_reg, 0);
    
                                     
                                                                                  
    (*ans).V = (uint8_t)_mm_cvtsi128_si32(result);
}

void MatMulVecM4(M4 Mat, V4 Vec, V4 *ans)
{
                                                                                                       
    uint32_t mat_32;
    memcpy(&mat_32, Mat.M, 4);
    
                            
    __m128i mat_reg = _mm_cvtsi32_si128(mat_32);
    
                      
    __m128i vec_reg = _mm_cvtsi32_si128(Vec.V);
    
                    
                                              
                              
                                                                   
    __m128i result = _mm_gf2p8affine_epi64_epi8(vec_reg, mat_reg, 0);
    
                      
    (*ans).V = ((uint8_t)_mm_cvtsi128_si32(result) >> 4) & 0x0F;
}

static inline uint64_t m128_reverse64(uint64_t x)
{
    x = ((x & 0x5555555555555555ULL) << 1) | ((x >> 1) & 0x5555555555555555ULL);
    x = ((x & 0x3333333333333333ULL) << 2) | ((x >> 2) & 0x3333333333333333ULL);
    x = ((x & 0x0f0f0f0f0f0f0f0fULL) << 4) | ((x >> 4) & 0x0f0f0f0f0f0f0f0fULL);
    x = ((x & 0x00ff00ff00ff00ffULL) << 8) | ((x >> 8) & 0x00ff00ff00ff00ffULL);
    x = ((x & 0x0000ffff0000ffffULL) << 16) | ((x >> 16) & 0x0000ffff0000ffffULL);
    return (x << 32) | (x >> 32);
}

static inline uint64_t m128_vpopcntdq_pairfold_half_testmask(const M128 *Mat, uint64_t v0, uint64_t v1, int row_base)
{
    const __m512i x = _mm512_set_epi64(
        (long long)v1, (long long)v0, (long long)v1, (long long)v0,
        (long long)v1, (long long)v0, (long long)v1, (long long)v0);
    const __m512i ones = _mm512_set1_epi64(1);
    uint64_t raw = 0;

    for (int block = 0; block < 16; block++) {
        const __m512i rows = _mm512_loadu_si512((const void *)&Mat->M[row_base + block * 4][0]);
        const __m512i cnt = _mm512_popcnt_epi64(_mm512_and_si512(rows, x));
        const uint32_t mask8 = (uint32_t)_mm512_test_epi64_mask(cnt, ones);
        const uint32_t pair = (mask8 ^ (mask8 >> 1)) & 0x55U;
        raw |= (uint64_t)_pext_u32(pair, 0x55U) << (4 * block);
    }

    return m128_reverse64(raw);
}

static inline void m128_vpopcntdq_pairfold_core(const M128 *Mat, const V128 *Vec, V128 *ans)
{
    ans->V[0] = m128_vpopcntdq_pairfold_half_testmask(Mat, Vec->V[0], Vec->V[1], 0);
    ans->V[1] = m128_vpopcntdq_pairfold_half_testmask(Mat, Vec->V[0], Vec->V[1], 64);
}

void MatMulVecM128(M128 Mat, V128 Vec, V128 *ans)
{
    m128_vpopcntdq_pairfold_core(&Mat, &Vec, ans);
}

static inline uint16_t m16_gfni_lowbyte_core(const M16 *Mat, uint16_t n)
{
    const uint8_t vL = (uint8_t)n;
    const uint8_t vH = (uint8_t)(n >> 8);
    const __m128i x = _mm_set_epi64x((long long)(uint64_t)vH, (long long)(uint64_t)vL);
    const __m128i planar_mask = _mm_setr_epi8(
        0, 2, 4, 6, 8, 10, 12, 14,
        1, 3, 5, 7, 9, 11, 13, 15);

    const __m128i top_raw = _mm_loadu_si128((const __m128i *)&Mat->M[0]);
    const __m128i top_planar = _mm_shuffle_epi8(top_raw, planar_mask);
    const __m128i top_parts = _mm_gf2p8affine_epi64_epi8(x, top_planar, 0);
    const __m128i top_xor = _mm_xor_si128(top_parts, _mm_bsrli_si128(top_parts, 8));
    const uint8_t y_top = (uint8_t)_mm_cvtsi128_si32(top_xor);

    const __m128i bot_raw = _mm_loadu_si128((const __m128i *)&Mat->M[8]);
    const __m128i bot_planar = _mm_shuffle_epi8(bot_raw, planar_mask);
    const __m128i bot_parts = _mm_gf2p8affine_epi64_epi8(x, bot_planar, 0);
    const __m128i bot_xor = _mm_xor_si128(bot_parts, _mm_bsrli_si128(bot_parts, 8));
    const uint8_t y_bot = (uint8_t)_mm_cvtsi128_si32(bot_xor);

    return (uint16_t)(((uint16_t)y_top << 8) | y_bot);
}

void MatMulVecM16(M16 Mat, V16 Vec, V16 *ans)
{
    ans->V = m16_gfni_lowbyte_core(&Mat, Vec.V);
}

uint16_t MatMulNumM16(M16 Mat, uint16_t n)
{
    return m16_gfni_lowbyte_core(&Mat, n);
}

static inline uint64_t m256_reverse64(uint64_t x)
{
    x = ((x & 0x5555555555555555ULL) << 1) | ((x >> 1) & 0x5555555555555555ULL);
    x = ((x & 0x3333333333333333ULL) << 2) | ((x >> 2) & 0x3333333333333333ULL);
    x = ((x & 0x0f0f0f0f0f0f0f0fULL) << 4) | ((x >> 4) & 0x0f0f0f0f0f0f0f0fULL);
    x = ((x & 0x00ff00ff00ff00ffULL) << 8) | ((x >> 8) & 0x00ff00ff00ff00ffULL);
    x = ((x & 0x0000ffff0000ffffULL) << 16) | ((x >> 16) & 0x0000ffff0000ffffULL);
    return (x << 32) | (x >> 32);
}

static inline uint32_t m256_compress2_from_0x11(uint32_t folded)
{
    return (folded & 0x1U) | ((folded >> 3) & 0x2U);
}

static inline uint32_t m256_pairrows_qword_movmask_nopext(__m512i rows, __m512i x)
{
    const __m512i cnt = _mm512_popcnt_epi64(_mm512_and_si512(rows, x));
    const uint32_t mask = (uint32_t)_mm512_movepi64_mask(_mm512_slli_epi64(cnt, 63));
    const uint32_t folded = (mask ^ (mask >> 1) ^ (mask >> 2) ^ (mask >> 3)) & 0x11U;
    return m256_compress2_from_0x11(folded);
}

static inline uint64_t m256_vpopcntdq_pairfold_half_4rowpack_movmask(const M256 *Mat, const V256 *Vec, int row_base)
{
    const __m512i x = _mm512_setr_epi64(
        (long long)Vec->V[0], (long long)Vec->V[1], (long long)Vec->V[2], (long long)Vec->V[3],
        (long long)Vec->V[0], (long long)Vec->V[1], (long long)Vec->V[2], (long long)Vec->V[3]);
    uint64_t raw = 0;

    for (int block = 0; block < 16; block++) {
        const int row = row_base + block * 4;
        const __m512i rows0 = _mm512_loadu_si512((const void *)&Mat->M[row][0]);
        const __m512i rows1 = _mm512_loadu_si512((const void *)&Mat->M[row + 2][0]);
        const uint32_t lo = m256_pairrows_qword_movmask_nopext(rows0, x);
        const uint32_t hi = m256_pairrows_qword_movmask_nopext(rows1, x);
        raw |= (uint64_t)(lo | (hi << 2)) << (4 * block);
    }

    return m256_reverse64(raw);
}

static inline void m256_vpopcntdq_pairfold_4rowpack_movmask_core(const M256 *Mat, const V256 *Vec, V256 *ans)
{
    ans->V[0] = m256_vpopcntdq_pairfold_half_4rowpack_movmask(Mat, Vec, 0);
    ans->V[1] = m256_vpopcntdq_pairfold_half_4rowpack_movmask(Mat, Vec, 64);
    ans->V[2] = m256_vpopcntdq_pairfold_half_4rowpack_movmask(Mat, Vec, 128);
    ans->V[3] = m256_vpopcntdq_pairfold_half_4rowpack_movmask(Mat, Vec, 192);
}

void MatMulVecM256(M256 Mat, V256 Vec, V256 *ans)
{
    m256_vpopcntdq_pairfold_4rowpack_movmask_core(&Mat, &Vec, ans);
}

static inline uint32_t m32_reverse32(uint32_t x)
{
    x = ((x & 0x55555555U) << 1) | ((x >> 1) & 0x55555555U);
    x = ((x & 0x33333333U) << 2) | ((x >> 2) & 0x33333333U);
    x = ((x & 0x0f0f0f0fU) << 4) | ((x >> 4) & 0x0f0f0f0fU);
    x = ((x & 0x00ff00ffU) << 8) | ((x >> 8) & 0x00ff00ffU);
    return (x << 16) | (x >> 16);
}

static inline uint32_t m32_avx512_vpopcntdq_bitreverse32_core(const M32 *Mat, uint32_t n)
{
    const __m512i x = _mm512_set1_epi32((int)n);
    const __m512i ones = _mm512_set1_epi32(1);
    const __m512i rows0 = _mm512_loadu_si512((const void *)&Mat->M[0]);
    const __m512i rows1 = _mm512_loadu_si512((const void *)&Mat->M[16]);
    const __m512i cnt0 = _mm512_popcnt_epi32(_mm512_and_si512(rows0, x));
    const __m512i cnt1 = _mm512_popcnt_epi32(_mm512_and_si512(rows1, x));
    const uint32_t raw = (uint32_t)_mm512_test_epi32_mask(cnt0, ones) |
                         ((uint32_t)_mm512_test_epi32_mask(cnt1, ones) << 16);

    return m32_reverse32(raw);
}

uint32_t MatMulNumM32(M32 Mat, uint32_t n)
{
    return m32_avx512_vpopcntdq_bitreverse32_core(&Mat, n);
}

static inline uint32_t m32_avx512vl_vpopcntdq_ymm_core(const M32 *Mat, uint32_t n)
{
    const __m256i x = _mm256_set1_epi32((int)n);
    const __m256i ones = _mm256_set1_epi32(1);
    const __m256i rows0 = _mm256_loadu_si256((const __m256i *)&Mat->M[0]);
    const __m256i rows1 = _mm256_loadu_si256((const __m256i *)&Mat->M[8]);
    const __m256i rows2 = _mm256_loadu_si256((const __m256i *)&Mat->M[16]);
    const __m256i rows3 = _mm256_loadu_si256((const __m256i *)&Mat->M[24]);
    const uint32_t raw = (uint32_t)_mm256_test_epi32_mask(_mm256_popcnt_epi32(_mm256_and_si256(rows0, x)), ones) |
                         ((uint32_t)_mm256_test_epi32_mask(_mm256_popcnt_epi32(_mm256_and_si256(rows1, x)), ones) << 8) |
                         ((uint32_t)_mm256_test_epi32_mask(_mm256_popcnt_epi32(_mm256_and_si256(rows2, x)), ones) << 16) |
                         ((uint32_t)_mm256_test_epi32_mask(_mm256_popcnt_epi32(_mm256_and_si256(rows3, x)), ones) << 24);

    return m32_reverse32(raw);
}

void MatMulVecM32(M32 Mat, V32 Vec, V32 *ans)
{
    ans->V = m32_avx512vl_vpopcntdq_ymm_core(&Mat, Vec.V);
}

static inline uint8_t m4_gfni_nomask_core(const M4 *Mat, uint8_t n)
{
    uint32_t rows;
    memcpy(&rows, Mat->M, sizeof(rows));

    __m128i mat_reg = _mm_cvtsi32_si128((int)rows);
    __m128i vec_reg = _mm_cvtsi32_si128((int)(n & 0x0fU));
    __m128i out = _mm_gf2p8affine_epi64_epi8(vec_reg, mat_reg, 0);

    return (uint8_t)(((uint32_t)_mm_cvtsi128_si32(out) >> 4) & 0x0fU);
}

uint8_t MatMulNumM4(M4 Mat, uint8_t n)
{
    return m4_gfni_nomask_core(&Mat, n);
}

static inline uint64_t m64_reverse64(uint64_t x)
{
    x = ((x & 0x5555555555555555ULL) << 1) | ((x >> 1) & 0x5555555555555555ULL);
    x = ((x & 0x3333333333333333ULL) << 2) | ((x >> 2) & 0x3333333333333333ULL);
    x = ((x & 0x0f0f0f0f0f0f0f0fULL) << 4) | ((x >> 4) & 0x0f0f0f0f0f0f0f0fULL);
    x = ((x & 0x00ff00ff00ff00ffULL) << 8) | ((x >> 8) & 0x00ff00ff00ff00ffULL);
    x = ((x & 0x0000ffff0000ffffULL) << 16) | ((x >> 16) & 0x0000ffff0000ffffULL);
    return (x << 32) | (x >> 32);
}

static inline uint64_t m64_avx512_vpopcntdq_bitreverse_core(const M64 *Mat, uint64_t n)
{
    const __m512i x = _mm512_set1_epi64((long long)n);
    const __m512i ones = _mm512_set1_epi64(1);
    uint64_t raw = 0;

    const __m512i rows0 = _mm512_loadu_si512((const void *)&Mat->M[0]);
    const __m512i cnt0 = _mm512_popcnt_epi64(_mm512_and_si512(rows0, x));
    raw |= (uint64_t)(uint8_t)_mm512_test_epi64_mask(cnt0, ones) << 0;

    const __m512i rows1 = _mm512_loadu_si512((const void *)&Mat->M[8]);
    const __m512i cnt1 = _mm512_popcnt_epi64(_mm512_and_si512(rows1, x));
    raw |= (uint64_t)(uint8_t)_mm512_test_epi64_mask(cnt1, ones) << 8;

    const __m512i rows2 = _mm512_loadu_si512((const void *)&Mat->M[16]);
    const __m512i cnt2 = _mm512_popcnt_epi64(_mm512_and_si512(rows2, x));
    raw |= (uint64_t)(uint8_t)_mm512_test_epi64_mask(cnt2, ones) << 16;

    const __m512i rows3 = _mm512_loadu_si512((const void *)&Mat->M[24]);
    const __m512i cnt3 = _mm512_popcnt_epi64(_mm512_and_si512(rows3, x));
    raw |= (uint64_t)(uint8_t)_mm512_test_epi64_mask(cnt3, ones) << 24;

    const __m512i rows4 = _mm512_loadu_si512((const void *)&Mat->M[32]);
    const __m512i cnt4 = _mm512_popcnt_epi64(_mm512_and_si512(rows4, x));
    raw |= (uint64_t)(uint8_t)_mm512_test_epi64_mask(cnt4, ones) << 32;

    const __m512i rows5 = _mm512_loadu_si512((const void *)&Mat->M[40]);
    const __m512i cnt5 = _mm512_popcnt_epi64(_mm512_and_si512(rows5, x));
    raw |= (uint64_t)(uint8_t)_mm512_test_epi64_mask(cnt5, ones) << 40;

    const __m512i rows6 = _mm512_loadu_si512((const void *)&Mat->M[48]);
    const __m512i cnt6 = _mm512_popcnt_epi64(_mm512_and_si512(rows6, x));
    raw |= (uint64_t)(uint8_t)_mm512_test_epi64_mask(cnt6, ones) << 48;

    const __m512i rows7 = _mm512_loadu_si512((const void *)&Mat->M[56]);
    const __m512i cnt7 = _mm512_popcnt_epi64(_mm512_and_si512(rows7, x));
    raw |= (uint64_t)(uint8_t)_mm512_test_epi64_mask(cnt7, ones) << 56;

    return m64_reverse64(raw);
}

uint64_t MatMulNumM64(M64 Mat, uint64_t n)
{
    return m64_avx512_vpopcntdq_bitreverse_core(&Mat, n);
}

static inline uint64_t m64_avx512_vpopcntdq_movmask_core(const M64 *Mat, uint64_t n)
{
    const __m512i x = _mm512_set1_epi64((long long)n);
    uint64_t raw = 0;

    const __m512i rows0 = _mm512_loadu_si512((const void *)&Mat->M[0]);
    const __m512i cnt0 = _mm512_popcnt_epi64(_mm512_and_si512(rows0, x));
    raw |= (uint64_t)(uint8_t)_mm512_movepi64_mask(_mm512_slli_epi64(cnt0, 63)) << 0;

    const __m512i rows1 = _mm512_loadu_si512((const void *)&Mat->M[8]);
    const __m512i cnt1 = _mm512_popcnt_epi64(_mm512_and_si512(rows1, x));
    raw |= (uint64_t)(uint8_t)_mm512_movepi64_mask(_mm512_slli_epi64(cnt1, 63)) << 8;

    const __m512i rows2 = _mm512_loadu_si512((const void *)&Mat->M[16]);
    const __m512i cnt2 = _mm512_popcnt_epi64(_mm512_and_si512(rows2, x));
    raw |= (uint64_t)(uint8_t)_mm512_movepi64_mask(_mm512_slli_epi64(cnt2, 63)) << 16;

    const __m512i rows3 = _mm512_loadu_si512((const void *)&Mat->M[24]);
    const __m512i cnt3 = _mm512_popcnt_epi64(_mm512_and_si512(rows3, x));
    raw |= (uint64_t)(uint8_t)_mm512_movepi64_mask(_mm512_slli_epi64(cnt3, 63)) << 24;

    const __m512i rows4 = _mm512_loadu_si512((const void *)&Mat->M[32]);
    const __m512i cnt4 = _mm512_popcnt_epi64(_mm512_and_si512(rows4, x));
    raw |= (uint64_t)(uint8_t)_mm512_movepi64_mask(_mm512_slli_epi64(cnt4, 63)) << 32;

    const __m512i rows5 = _mm512_loadu_si512((const void *)&Mat->M[40]);
    const __m512i cnt5 = _mm512_popcnt_epi64(_mm512_and_si512(rows5, x));
    raw |= (uint64_t)(uint8_t)_mm512_movepi64_mask(_mm512_slli_epi64(cnt5, 63)) << 40;

    const __m512i rows6 = _mm512_loadu_si512((const void *)&Mat->M[48]);
    const __m512i cnt6 = _mm512_popcnt_epi64(_mm512_and_si512(rows6, x));
    raw |= (uint64_t)(uint8_t)_mm512_movepi64_mask(_mm512_slli_epi64(cnt6, 63)) << 48;

    const __m512i rows7 = _mm512_loadu_si512((const void *)&Mat->M[56]);
    const __m512i cnt7 = _mm512_popcnt_epi64(_mm512_and_si512(rows7, x));
    raw |= (uint64_t)(uint8_t)_mm512_movepi64_mask(_mm512_slli_epi64(cnt7, 63)) << 56;

    return m64_reverse64(raw);
}

void MatMulVecM64(M64 Mat, V64 Vec, V64 *ans)
{
    ans->V = m64_avx512_vpopcntdq_movmask_core(&Mat, Vec.V);
}

static inline uint8_t m8_gfni_u64load_core(const M8 *Mat, uint8_t n)
{
    uint64_t rows;
    memcpy(&rows, Mat->M, sizeof(rows));

    __m128i x = _mm_cvtsi32_si128((int)n);
    __m128i mat = _mm_cvtsi64_si128((long long)rows);
    __m128i out = _mm_gf2p8affine_epi64_epi8(x, mat, 0);

    return (uint8_t)_mm_cvtsi128_si32(out);
}

uint8_t MatMulNumM8(M8 Mat, uint8_t n)
{
    return m8_gfni_u64load_core(&Mat, n);
}
