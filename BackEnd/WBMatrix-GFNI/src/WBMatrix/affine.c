#include "internal.h"




static const uint8_t GPT_M128_PACK_VPERMB_MASK_ARR[64] __attribute__((aligned(64))) = {
    56, 48, 40, 32, 24, 16,  8, 0,
    57, 49, 41, 33, 25, 17,  9, 1,
    58, 50, 42, 34, 26, 18, 10, 2,
    59, 51, 43, 35, 27, 19, 11, 3,
    60, 52, 44, 36, 28, 20, 12, 4,
    61, 53, 45, 37, 29, 21, 13, 5,
    62, 54, 46, 38, 30, 22, 14, 6,
    63, 55, 47, 39, 31, 23, 15, 7
};
static const uint8_t GPT_M128_VPERMB_MASK_ARR[64] __attribute__((aligned(64))) = {
    7, 15, 23, 31, 39, 47, 55, 63,
    6, 14, 22, 30, 38, 46, 54, 62,
    5, 13, 21, 29, 37, 45, 53, 61,
    4, 12, 20, 28, 36, 44, 52, 60,
    3, 11, 19, 27, 35, 43, 51, 59,
    2, 10, 18, 26, 34, 42, 50, 58,
    1,  9, 17, 25, 33, 41, 49, 57,
    0,  8, 16, 24, 32, 40, 48, 56
};



static inline uint64_t affineu128_reverse64(uint64_t x);
static inline uint64_t affineu128_pairfold_half_testmask(const M128 *Mat, uint64_t v0, uint64_t v1, int row_base);
static inline void affineu128_pairfold_core(const M128 *Mat, uint64_t v0, uint64_t v1, uint64_t ans[2]);
void affineU128(Aff128 aff, uint64_t arr[], uint64_t ans[]);
static inline uint16_t affineu16_gfni_ymm_linear(const M16 *mat, uint16_t x);
uint16_t affineU16(Aff16 aff, uint16_t arr);
static inline uint32_t affineu32_reverse32(uint32_t x);
static inline uint32_t affineu32_avx512_vpopcntdq_bitreverse32_core(const M32 *Mat, uint32_t n);
uint32_t affineU32(Aff32 aff, uint32_t arr);
static inline uint8_t affineu4_gfni_masked_linear(const M4 *mat, uint8_t x);
uint8_t affineU4(Aff4 aff, uint8_t arr);
static inline uint64_t affineu64_reverse64(uint64_t x);
static inline uint64_t affineu64_avx512_vpopcntdq_bitreverse_core(const M64 *Mat, uint64_t n);
uint64_t affineU64(Aff64 aff, uint64_t arr);
static inline uint8_t affineu8_gfni_u64load_linear(const M8 *mat, uint8_t x);
uint8_t affineU8(Aff8 aff, uint8_t arr);
static inline void extract_m128_8x8tiles_gpt(
    const M128 *in, int row_start,
    __m512i *out_left, __m512i *out_right,
    __m512i idx_left, __m512i idx_right, __m512i vpermb_mask);
static inline void pack_m128_8x8tiles_gpt(
    __m512i in_left, __m512i in_right, M128 *out, int row_start,
    __m512i pack_mask, __m512i idx_pack0, __m512i idx_pack1);
static inline M128TileCtx_GPT make_m128_tile_ctx_gpt(void);
static inline __m512i broadcast_a_tile_lane_gpt(
    __m512i a_tiles, const M128TileCtx_GPT *ctx, int lane);
static void MatMulMatM128_GPT5_prepareB_8x8tiles(
    const M128 *Mat2, M128PreparedB_8x8Tiles_GPT *PreparedB);
static void MatMulMatM128_GPT11_preparedB_8x8tiles_dualrow_rowdirect(
    const M128 *Mat1, const M128PreparedB_8x8Tiles_GPT *PreparedB, M128 *Mat);
static void MatMulMatM128_GPT11_8x8tiles_dualrow_rowdirect_total(const M128 *Mat1, const M128 *Mat2, M128 *Mat);
void affinemixM128(Aff128 aff, Aff128 preaff_inv, Aff128 *mixaff);
void genaffinepairM128(Aff128 *aff, Aff128 *aff_inv);
static inline uint16_t affinemix_m16_lowbyte_mulnum(const M16 *Mat, uint16_t n);
static inline void affinemix_m16_matmul_ymm3gfni(const M16 *Mat1, const M16 *Mat2, M16 *Mat);
static inline void affinemix_m16_gfni_fused_core(const Aff16 *aff, const Aff16 *preaff_inv, Aff16 *mixaff);
void affinemixM16(Aff16 aff, Aff16 preaff_inv, Aff16 *mixaff);
static inline uint16_t m16_genaffinepair_lowbyte_mulnum(const M16 *Mat, uint16_t n);
void genaffinepairM16(Aff16 *aff, Aff16 *aff_inv);
static inline void MatMulMatM32_GPT4_rowdirect_core(
    __m256i A0, __m256i A1, __m256i A2, __m256i A3,
    __m256i B0, __m256i B1, __m256i B2, __m256i B3,
    __m256i *C0, __m256i *C1, __m256i *C2, __m256i *C3);
static void MatMulMatM32_GPT4_reg_ymms(
    __m256i A0, __m256i A1, __m256i A2, __m256i A3,
    __m256i B0, __m256i B1, __m256i B2, __m256i B3,
    __m256i *C0, __m256i *C1, __m256i *C2, __m256i *C3);
static inline uint32_t affinemix_m32_reverse32(uint32_t x);
static inline uint32_t affinemix_m32_bitreverse32_mulnum_from_rows(
    __m256i A0, __m256i A1, __m256i A2, __m256i A3, uint32_t n);
static inline void affinemix_m32_rowdirect_matmul_from_rows(
    __m256i A0, __m256i A1, __m256i A2, __m256i A3,
    const M32 *Mat2, M32 *Mat);
static inline void affinemix_m32_gfni_fused_core(const Aff32 *aff, const Aff32 *preaff_inv, Aff32 *mixaff);
void affinemixM32(Aff32 aff, Aff32 preaff_inv, Aff32 *mixaff);
static inline uint32_t m32_genaffinepair_reverse32(uint32_t x);
static inline uint32_t m32_genaffinepair_bitreverse32_mulnum(const M32 *Mat, uint32_t n);
void genaffinepairM32(Aff32 *aff, Aff32 *aff_inv);
static inline void affinemix_m4_gfni_fused_core(const Aff4 *aff, const Aff4 *preaff_inv, Aff4 *mixaff);
void affinemixM4(Aff4 aff, Aff4 preaff_inv, Aff4 *mixaff);
static inline uint8_t m4_genaffinepair_gfni_mulnum(const M4 *Mat, uint8_t n);
void genaffinepairM4(Aff4 *aff, Aff4 *aff_inv);
static inline __attribute__((always_inline)) __m512i pair_ymms_to_zmm_gpt12(
    __m256i lo, __m256i hi);
static inline __attribute__((always_inline)) void prepare_m32pairbt_from_m64_dualhalf_gpt18(
    const M64 *Mat2, M64PreparedB_M32PairBT_GPT *PreparedB);
static inline __attribute__((always_inline)) void m32_pair_prepared_product_xor2_inline_gpt14(
    const __m256i A0[4], const __m512i PairBT0[4],
    const __m256i A1[4], const __m512i PairBT1[4],
    __m512i Cpair[4]);
static inline __attribute__((always_inline)) void load_m64_half_as_m32_ymms_direct_gpt15(
    const uint64_t *rows, __m256i Aleft[4], __m256i Aright[4]);
static inline __attribute__((always_inline)) void store_m32_pair_ymms_as_m64_half_gpt15(
    const __m512i Cpair[4], uint64_t *rows);
static void MatMulMatM64_GPT15_preparedB_m32pairbt_rawaccum_streamA(
    const M64 *Mat1, const M64PreparedB_M32PairBT_GPT *PreparedB, M64 *Mat);
static void MatMulMatM64_GPT19_m32pairbt_rawaccum_streamA_dualprepare_total(
    const M64 *Mat1, const M64 *Mat2, M64 *Mat);
void affinemixM64(Aff64 aff, Aff64 preaff_inv, Aff64 *mixaff);
void genaffinepairM64(Aff64 *aff, Aff64 *aff_inv);
static inline void affinemix_m8_gfni_fused_core(const Aff8 *aff, const Aff8 *preaff_inv, Aff8 *mixaff);
void affinemixM8(Aff8 aff, Aff8 preaff_inv, Aff8 *mixaff);
static inline uint8_t m8_genaffinepair_gfni_u64load_mulnum(const M8 *Mat, uint8_t n);
void genaffinepairM8(Aff8 *aff, Aff8 *aff_inv);
static inline uint64_t m128_reverse64(uint64_t x);
static inline uint64_t m128_vpopcntdq_pairfold_half_testmask(const M128 *Mat, uint64_t v0, uint64_t v1, int row_base);
static inline void m128_vpopcntdq_pairfold_core(const M128 *Mat, const V128 *Vec, V128 *ans);
static void MatMulVecM128_GPT5_avx512_vpopcntdq_pairfold_ptr(const M128 *Mat, const V128 *Vec, V128 *ans);
static inline uint64_t m64_reverse64(uint64_t x);
static inline uint64_t m64_avx512_vpopcntdq_movmask_core(const M64 *Mat, uint64_t n);
static uint64_t MatMulNumM64_GPT6_avx512_vpopcntdq_movmask_ptr(const M64 *Mat, uint64_t n);

static inline uint64_t affineu128_reverse64(uint64_t x)
{
    x = ((x & 0x5555555555555555ULL) << 1) | ((x >> 1) & 0x5555555555555555ULL);
    x = ((x & 0x3333333333333333ULL) << 2) | ((x >> 2) & 0x3333333333333333ULL);
    x = ((x & 0x0f0f0f0f0f0f0f0fULL) << 4) | ((x >> 4) & 0x0f0f0f0f0f0f0f0fULL);
    x = ((x & 0x00ff00ff00ff00ffULL) << 8) | ((x >> 8) & 0x00ff00ff00ff00ffULL);
    x = ((x & 0x0000ffff0000ffffULL) << 16) | ((x >> 16) & 0x0000ffff0000ffffULL);
    return (x << 32) | (x >> 32);
}

static inline uint64_t affineu128_pairfold_half_testmask(const M128 *Mat, uint64_t v0, uint64_t v1, int row_base)
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

    return affineu128_reverse64(raw);
}

static inline void affineu128_pairfold_core(const M128 *Mat, uint64_t v0, uint64_t v1, uint64_t ans[2])
{
    ans[0] = affineu128_pairfold_half_testmask(Mat, v0, v1, 0);
    ans[1] = affineu128_pairfold_half_testmask(Mat, v0, v1, 64);
}

void affineU128(Aff128 aff, uint64_t arr[], uint64_t ans[])
{
    affineu128_pairfold_core(&aff.Mat, arr[0], arr[1], ans);
    ans[0] ^= aff.Vec.V[0];
    ans[1] ^= aff.Vec.V[1];
}

static inline uint16_t affineu16_gfni_ymm_linear(const M16 *mat, uint16_t x)
{
    const uint8_t x_low = (uint8_t)x;
    const uint8_t x_high = (uint8_t)(x >> 8);
    const __m128i planar_mask = _mm_setr_epi8(
        0, 2, 4, 6, 8, 10, 12, 14,
        1, 3, 5, 7, 9, 11, 13, 15);
    const __m128i top_raw = _mm_loadu_si128((const __m128i *)&mat->M[0]);
    const __m128i bot_raw = _mm_loadu_si128((const __m128i *)&mat->M[8]);
    const __m128i top_planar = _mm_shuffle_epi8(top_raw, planar_mask);
    const __m128i bot_planar = _mm_shuffle_epi8(bot_raw, planar_mask);
    const __m256i mats = _mm256_inserti128_si256(_mm256_castsi128_si256(top_planar), bot_planar, 1);
    const __m256i x_reg = _mm256_set_epi64x((long long)(uint64_t)x_high,
                                            (long long)(uint64_t)x_low,
                                            (long long)(uint64_t)x_high,
                                            (long long)(uint64_t)x_low);
    const __m256i parts = _mm256_gf2p8affine_epi64_epi8(x_reg, mats, 0);
    const __m256i pair_xor = _mm256_xor_si256(parts, _mm256_srli_si256(parts, 8));
    const uint8_t y_top = (uint8_t)_mm_cvtsi128_si32(_mm256_castsi256_si128(pair_xor));
    const uint8_t y_bot = (uint8_t)_mm_cvtsi128_si32(_mm256_extracti128_si256(pair_xor, 1));

    return (uint16_t)(((uint16_t)y_top << 8) | y_bot);
}

uint16_t affineU16(Aff16 aff, uint16_t arr)
{
    return (uint16_t)(affineu16_gfni_ymm_linear(&aff.Mat, arr) ^ aff.Vec.V);
}

static inline uint32_t affineu32_reverse32(uint32_t x)
{
    x = ((x & 0x55555555U) << 1) | ((x >> 1) & 0x55555555U);
    x = ((x & 0x33333333U) << 2) | ((x >> 2) & 0x33333333U);
    x = ((x & 0x0f0f0f0fU) << 4) | ((x >> 4) & 0x0f0f0f0fU);
    x = ((x & 0x00ff00ffU) << 8) | ((x >> 8) & 0x00ff00ffU);
    return (x << 16) | (x >> 16);
}

static inline uint32_t affineu32_avx512_vpopcntdq_bitreverse32_core(const M32 *Mat, uint32_t n)
{
    const __m512i x = _mm512_set1_epi32((int)n);
    const __m512i ones = _mm512_set1_epi32(1);
    const __m512i rows0 = _mm512_loadu_si512((const void *)&Mat->M[0]);
    const __m512i rows1 = _mm512_loadu_si512((const void *)&Mat->M[16]);
    const __m512i cnt0 = _mm512_popcnt_epi32(_mm512_and_si512(rows0, x));
    const __m512i cnt1 = _mm512_popcnt_epi32(_mm512_and_si512(rows1, x));
    const uint32_t raw = (uint32_t)_mm512_test_epi32_mask(cnt0, ones) |
                         ((uint32_t)_mm512_test_epi32_mask(cnt1, ones) << 16);

    return affineu32_reverse32(raw);
}

uint32_t affineU32(Aff32 aff, uint32_t arr)
{
    return affineu32_avx512_vpopcntdq_bitreverse32_core(&aff.Mat, arr) ^ aff.Vec.V;
}

static inline uint8_t affineu4_gfni_masked_linear(const M4 *mat, uint8_t x)
{
    uint32_t rows;
    memcpy(&rows, mat->M, sizeof(rows));
    rows &= 0x0f0f0f0fU;

    __m128i mat_reg = _mm_cvtsi32_si128((int)rows);
    __m128i x_reg = _mm_cvtsi32_si128((int)(x & 0x0fU));
    __m128i out = _mm_gf2p8affine_epi64_epi8(x_reg, mat_reg, 0);

    return (uint8_t)(((uint32_t)_mm_cvtsi128_si32(out) >> 4) & 0x0fU);
}

uint8_t affineU4(Aff4 aff, uint8_t arr)
{
    return (uint8_t)(affineu4_gfni_masked_linear(&aff.Mat, arr) ^ aff.Vec.V);
}

static inline uint64_t affineu64_reverse64(uint64_t x)
{
    x = ((x & 0x5555555555555555ULL) << 1) | ((x >> 1) & 0x5555555555555555ULL);
    x = ((x & 0x3333333333333333ULL) << 2) | ((x >> 2) & 0x3333333333333333ULL);
    x = ((x & 0x0f0f0f0f0f0f0f0fULL) << 4) | ((x >> 4) & 0x0f0f0f0f0f0f0f0fULL);
    x = ((x & 0x00ff00ff00ff00ffULL) << 8) | ((x >> 8) & 0x00ff00ff00ff00ffULL);
    x = ((x & 0x0000ffff0000ffffULL) << 16) | ((x >> 16) & 0x0000ffff0000ffffULL);
    return (x << 32) | (x >> 32);
}

static inline uint64_t affineu64_avx512_vpopcntdq_bitreverse_core(const M64 *Mat, uint64_t n)
{
    const __m512i x = _mm512_set1_epi64((long long)n);
    const __m512i ones = _mm512_set1_epi64(1);
    uint64_t raw = 0;

    for (int block = 0; block < 8; block++) {
        const __m512i rows = _mm512_loadu_si512((const void *)&Mat->M[block * 8]);
        const __m512i cnt = _mm512_popcnt_epi64(_mm512_and_si512(rows, x));
        raw |= (uint64_t)(uint8_t)_mm512_test_epi64_mask(cnt, ones) << (8 * block);
    }

    return affineu64_reverse64(raw);
}

uint64_t affineU64(Aff64 aff, uint64_t arr)
{
    return affineu64_avx512_vpopcntdq_bitreverse_core(&aff.Mat, arr) ^ aff.Vec.V;
}

static inline uint8_t affineu8_gfni_u64load_linear(const M8 *mat, uint8_t x)
{
    uint64_t rows;
    memcpy(&rows, mat->M, sizeof(rows));

    __m128i x_reg = _mm_cvtsi32_si128((int)x);
    __m128i mat_reg = _mm_cvtsi64_si128((long long)rows);
    __m128i out = _mm_gf2p8affine_epi64_epi8(x_reg, mat_reg, 0);

    return (uint8_t)_mm_cvtsi128_si32(out);
}

uint8_t affineU8(Aff8 aff, uint8_t arr)
{
    return (uint8_t)(affineu8_gfni_u64load_linear(&aff.Mat, arr) ^ aff.Vec.V);
}

static inline void extract_m128_8x8tiles_gpt(
    const M128 *in, int row_start,
    __m512i *out_left, __m512i *out_right,
    __m512i idx_left, __m512i idx_right, __m512i vpermb_mask)
{
    __m512i rows0 = _mm512_loadu_si512((const __m512i *)&in->M[row_start][0]);
    __m512i rows1 = _mm512_loadu_si512((const __m512i *)&in->M[row_start + 4][0]);
    __m512i left_rows = _mm512_permutex2var_epi64(rows0, idx_left, rows1);
    __m512i right_rows = _mm512_permutex2var_epi64(rows0, idx_right, rows1);
    *out_left = _mm512_permutexvar_epi8(vpermb_mask, left_rows);
    *out_right = _mm512_permutexvar_epi8(vpermb_mask, right_rows);
}

static inline void pack_m128_8x8tiles_gpt(
    __m512i in_left, __m512i in_right, M128 *out, int row_start,
    __m512i pack_mask, __m512i idx_pack0, __m512i idx_pack1)
{
    __m512i left_rows = _mm512_permutexvar_epi8(pack_mask, in_left);
    __m512i right_rows = _mm512_permutexvar_epi8(pack_mask, in_right);
    __m512i rows0 = _mm512_permutex2var_epi64(left_rows, idx_pack0, right_rows);
    __m512i rows1 = _mm512_permutex2var_epi64(left_rows, idx_pack1, right_rows);
    _mm512_storeu_si512((__m512i *)&out->M[row_start][0], rows0);
    _mm512_storeu_si512((__m512i *)&out->M[row_start + 4][0], rows1);
}

static inline M128TileCtx_GPT make_m128_tile_ctx_gpt(void) {
    M128TileCtx_GPT ctx;
    const uint64_t transpose_const = 0x0102040810204080ULL;
    ctx.transpose = _mm512_set1_epi64((long long)transpose_const);
    ctx.vpermb_mask = _mm512_load_si512((const __m512i *)GPT_M128_VPERMB_MASK_ARR);
    ctx.pack_mask = _mm512_load_si512((const __m512i *)GPT_M128_PACK_VPERMB_MASK_ARR);
    ctx.idx_left = _mm512_set_epi64(14, 12, 10, 8, 6, 4, 2, 0);
    ctx.idx_right = _mm512_set_epi64(15, 13, 11, 9, 7, 5, 3, 1);
    ctx.idx_pack0 = _mm512_set_epi64(11, 3, 10, 2, 9, 1, 8, 0);
    ctx.idx_pack1 = _mm512_set_epi64(15, 7, 14, 6, 13, 5, 12, 4);
    ctx.bcast_lane[0] = _mm512_set1_epi64(0);
    ctx.bcast_lane[1] = _mm512_set1_epi64(1);
    ctx.bcast_lane[2] = _mm512_set1_epi64(2);
    ctx.bcast_lane[3] = _mm512_set1_epi64(3);
    ctx.bcast_lane[4] = _mm512_set1_epi64(4);
    ctx.bcast_lane[5] = _mm512_set1_epi64(5);
    ctx.bcast_lane[6] = _mm512_set1_epi64(6);
    ctx.bcast_lane[7] = _mm512_set1_epi64(7);
    return ctx;
}

static inline __m512i broadcast_a_tile_lane_gpt(
    __m512i a_tiles, const M128TileCtx_GPT *ctx, int lane)
{
    return _mm512_permutexvar_epi64(ctx->bcast_lane[lane], a_tiles);
}

static void MatMulMatM128_GPT5_prepareB_8x8tiles(
    const M128 *Mat2, M128PreparedB_8x8Tiles_GPT *PreparedB)
{
    const uint64_t transpose_const = 0x0102040810204080ULL;
    __m512i transpose = _mm512_set1_epi64((long long)transpose_const);
    __m512i vpermb_mask = _mm512_load_si512((const __m512i *)GPT_M128_VPERMB_MASK_ARR);
    __m512i idx_left = _mm512_set_epi64(14, 12, 10, 8, 6, 4, 2, 0);
    __m512i idx_right = _mm512_set_epi64(15, 13, 11, 9, 7, 5, 3, 1);

    for (int k = 0; k < 16; k++) {
        __m512i b_left, b_right;
        extract_m128_8x8tiles_gpt(Mat2, k * 8, &b_left, &b_right,
                                  idx_left, idx_right, vpermb_mask);
        PreparedB->BT[k][0] = _mm512_gf2p8affine_epi64_epi8(transpose, b_left, 0);
        PreparedB->BT[k][1] = _mm512_gf2p8affine_epi64_epi8(transpose, b_right, 0);
    }
}

static void MatMulMatM128_GPT11_preparedB_8x8tiles_dualrow_rowdirect(
    const M128 *Mat1, const M128PreparedB_8x8Tiles_GPT *PreparedB, M128 *Mat)
{
    M128TileCtx_GPT ctx = make_m128_tile_ctx_gpt();

    for (int i = 0; i < 16; i += 2) {
        __m512i a0_left, a0_right;
        __m512i a1_left, a1_right;
        __m512i c0_left = _mm512_setzero_si512();
        __m512i c0_right = _mm512_setzero_si512();
        __m512i c1_left = _mm512_setzero_si512();
        __m512i c1_right = _mm512_setzero_si512();

        extract_m128_8x8tiles_gpt(Mat1, i * 8, &a0_left, &a0_right,
                                  ctx.idx_left, ctx.idx_right, ctx.vpermb_mask);
        extract_m128_8x8tiles_gpt(Mat1, (i + 1) * 8, &a1_left, &a1_right,
                                  ctx.idx_left, ctx.idx_right, ctx.vpermb_mask);

        for (int k = 0; k < 8; k++) {
            __m512i b_left = PreparedB->BT[k][0];
            __m512i b_right = PreparedB->BT[k][1];
            __m512i a0_broadcast = broadcast_a_tile_lane_gpt(a0_left, &ctx, k);
            __m512i a1_broadcast = broadcast_a_tile_lane_gpt(a1_left, &ctx, k);

            c0_left = _mm512_xor_si512(
                c0_left, _mm512_gf2p8affine_epi64_epi8(a0_broadcast, b_left, 0));
            c0_right = _mm512_xor_si512(
                c0_right, _mm512_gf2p8affine_epi64_epi8(a0_broadcast, b_right, 0));
            c1_left = _mm512_xor_si512(
                c1_left, _mm512_gf2p8affine_epi64_epi8(a1_broadcast, b_left, 0));
            c1_right = _mm512_xor_si512(
                c1_right, _mm512_gf2p8affine_epi64_epi8(a1_broadcast, b_right, 0));
        }
        for (int k = 0; k < 8; k++) {
            __m512i b_left = PreparedB->BT[k + 8][0];
            __m512i b_right = PreparedB->BT[k + 8][1];
            __m512i a0_broadcast = broadcast_a_tile_lane_gpt(a0_right, &ctx, k);
            __m512i a1_broadcast = broadcast_a_tile_lane_gpt(a1_right, &ctx, k);

            c0_left = _mm512_xor_si512(
                c0_left, _mm512_gf2p8affine_epi64_epi8(a0_broadcast, b_left, 0));
            c0_right = _mm512_xor_si512(
                c0_right, _mm512_gf2p8affine_epi64_epi8(a0_broadcast, b_right, 0));
            c1_left = _mm512_xor_si512(
                c1_left, _mm512_gf2p8affine_epi64_epi8(a1_broadcast, b_left, 0));
            c1_right = _mm512_xor_si512(
                c1_right, _mm512_gf2p8affine_epi64_epi8(a1_broadcast, b_right, 0));
        }

        pack_m128_8x8tiles_gpt(c0_left, c0_right, Mat, i * 8,
                               ctx.pack_mask, ctx.idx_pack0, ctx.idx_pack1);
        pack_m128_8x8tiles_gpt(c1_left, c1_right, Mat, (i + 1) * 8,
                               ctx.pack_mask, ctx.idx_pack0, ctx.idx_pack1);
    }
}

static void MatMulMatM128_GPT11_8x8tiles_dualrow_rowdirect_total(const M128 *Mat1, const M128 *Mat2, M128 *Mat) {
    M128PreparedB_8x8Tiles_GPT PreparedB;
    MatMulMatM128_GPT5_prepareB_8x8tiles(Mat2, &PreparedB);
    MatMulMatM128_GPT11_preparedB_8x8tiles_dualrow_rowdirect(Mat1, &PreparedB, Mat);
}

void affinemixM128(Aff128 aff, Aff128 preaff_inv, Aff128 *mixaff)
{
    MatMulMatM128_GPT11_8x8tiles_dualrow_rowdirect_total(&aff.Mat, &preaff_inv.Mat, &mixaff->Mat);
    MatMulVecM128_GPT5_avx512_vpopcntdq_pairfold_ptr(&aff.Mat, &preaff_inv.Vec, &mixaff->Vec);
    mixaff->Vec.V[0] ^= aff.Vec.V[0];
    mixaff->Vec.V[1] ^= aff.Vec.V[1];
}

void genaffinepairM128(Aff128 *aff, Aff128 *aff_inv)
{
    genMatpairM128(
        &aff->Mat, &aff_inv->Mat);
    randV128(&aff->Vec);
    MatMulVecM128(aff_inv->Mat, aff->Vec, &aff_inv->Vec);
}

static inline uint16_t affinemix_m16_lowbyte_mulnum(const M16 *Mat, uint16_t n)
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

static inline void affinemix_m16_matmul_ymm3gfni(const M16 *Mat1, const M16 *Mat2, M16 *Mat)
{
    const uint64_t i_val = 0x0102040810204080ULL;
    const __m256i i_trans = _mm256_set1_epi64x((long long)i_val);
    const __m256i a = _mm256_loadu_si256((const __m256i *)Mat1->M);
    const __m256i b = _mm256_loadu_si256((const __m256i *)Mat2->M);

    const __m256i mask_b = _mm256_setr_epi8(
        1, 3, 5, 7, 9, 11, 13, 15,
        0, 2, 4, 6, 8, 10, 12, 14,
        1, 3, 5, 7, 9, 11, 13, 15,
        0, 2, 4, 6, 8, 10, 12, 14);
    const __m256i b_planar = _mm256_shuffle_epi8(b, mask_b);
    const __m256i b_t = _mm256_gf2p8affine_epi64_epi8(i_trans, b_planar, 0);

    const __m256i b_top_t = _mm256_permute2x128_si256(b_t, b_t, 0x00);
    const __m256i b_bot_t = _mm256_permute2x128_si256(b_t, b_t, 0x11);

    const __m256i mask_a_h = _mm256_setr_epi8(
        1, 3, 5, 7, 9, 11, 13, 15,
        1, 3, 5, 7, 9, 11, 13, 15,
        1, 3, 5, 7, 9, 11, 13, 15,
        1, 3, 5, 7, 9, 11, 13, 15);
    const __m256i mask_a_l = _mm256_setr_epi8(
        0, 2, 4, 6, 8, 10, 12, 14,
        0, 2, 4, 6, 8, 10, 12, 14,
        0, 2, 4, 6, 8, 10, 12, 14,
        0, 2, 4, 6, 8, 10, 12, 14);

    const __m256i a_h = _mm256_shuffle_epi8(a, mask_a_h);
    const __m256i a_l = _mm256_shuffle_epi8(a, mask_a_l);

    const __m256i rows = _mm256_xor_si256(
        _mm256_gf2p8affine_epi64_epi8(a_h, b_top_t, 0),
        _mm256_gf2p8affine_epi64_epi8(a_l, b_bot_t, 0));

    const __m256i store_mask = _mm256_setr_epi8(
        8, 0, 9, 1, 10, 2, 11, 3,
        12, 4, 13, 5, 14, 6, 15, 7,
        8, 0, 9, 1, 10, 2, 11, 3,
        12, 4, 13, 5, 14, 6, 15, 7);
    const __m256i c = _mm256_shuffle_epi8(rows, store_mask);
    _mm256_storeu_si256((__m256i *)Mat->M, c);
}

static inline void affinemix_m16_gfni_fused_core(const Aff16 *aff, const Aff16 *preaff_inv, Aff16 *mixaff)
{
    affinemix_m16_matmul_ymm3gfni(&aff->Mat, &preaff_inv->Mat, &mixaff->Mat);
    mixaff->Vec.V = (uint16_t)(affinemix_m16_lowbyte_mulnum(&aff->Mat, preaff_inv->Vec.V) ^ aff->Vec.V);
}

void affinemixM16(Aff16 aff, Aff16 preaff_inv, Aff16 *mixaff)
{
    affinemix_m16_gfni_fused_core(&aff, &preaff_inv, mixaff);
}

static inline uint16_t m16_genaffinepair_lowbyte_mulnum(const M16 *Mat, uint16_t n)
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

void genaffinepairM16(Aff16 *aff, Aff16 *aff_inv)
{
    genMatpairM16(
        &aff->Mat, &aff_inv->Mat);
    aff->Vec.V = (uint16_t)cus_random();
    aff_inv->Vec.V = m16_genaffinepair_lowbyte_mulnum(&aff_inv->Mat, aff->Vec.V);
}

static inline void MatMulMatM32_GPT4_rowdirect_core(
    __m256i A0, __m256i A1, __m256i A2, __m256i A3,
    __m256i B0, __m256i B1, __m256i B2, __m256i B3,
    __m256i *C0, __m256i *C1, __m256i *C2, __m256i *C3)
{
    const __m256i I = _mm256_set1_epi64x(0x0102040810204080ULL);

    const __m256i shuf_B = _mm256_setr_epi8(
        0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
        0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    const __m256i perm_B = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

    __m256i BT0 = _mm256_gf2p8affine_epi64_epi8(
        I, _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(B0, shuf_B), perm_B), 0);
    __m256i BT1 = _mm256_gf2p8affine_epi64_epi8(
        I, _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(B1, shuf_B), perm_B), 0);
    __m256i BT2 = _mm256_gf2p8affine_epi64_epi8(
        I, _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(B2, shuf_B), perm_B), 0);
    __m256i BT3 = _mm256_gf2p8affine_epi64_epi8(
        I, _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(B3, shuf_B), perm_B), 0);

    const __m256i m0 = _mm256_setr_epi8(
        3, 7, 11, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        3, 7, 11, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i m1 = _mm256_setr_epi8(
        2, 6, 10, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        2, 6, 10, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i m2 = _mm256_setr_epi8(
        1, 5, 9, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        1, 5, 9, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i m3 = _mm256_setr_epi8(
        0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i perm_ctrl = _mm256_setr_epi32(0, 4, 0, 4, 0, 4, 0, 4);

#define GPT4_CALC_M32_ROW(A_ROW, C_OUT) do { \
        __m256i aik0 = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8((A_ROW), m0), perm_ctrl); \
        __m256i aik1 = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8((A_ROW), m1), perm_ctrl); \
        __m256i aik2 = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8((A_ROW), m2), perm_ctrl); \
        __m256i aik3 = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8((A_ROW), m3), perm_ctrl); \
        __m256i t0 = _mm256_gf2p8affine_epi64_epi8(aik0, BT0, 0); \
        __m256i t1 = _mm256_gf2p8affine_epi64_epi8(aik1, BT1, 0); \
        __m256i t2 = _mm256_gf2p8affine_epi64_epi8(aik2, BT2, 0); \
        __m256i t3 = _mm256_gf2p8affine_epi64_epi8(aik3, BT3, 0); \
        __m256i acc01 = _mm256_xor_si256(t0, t1); \
        __m256i acc23 = _mm256_xor_si256(t2, t3); \
        (C_OUT) = _mm256_xor_si256(acc01, acc23); \
    } while (0)

    GPT4_CALC_M32_ROW(A0, *C0);
    GPT4_CALC_M32_ROW(A1, *C1);
    GPT4_CALC_M32_ROW(A2, *C2);
    GPT4_CALC_M32_ROW(A3, *C3);

#undef GPT4_CALC_M32_ROW

    const __m256i inv_perm = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
    const __m256i inv_shuf = _mm256_setr_epi8(
        0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
        0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);

    *C0 = _mm256_shuffle_epi8(_mm256_permutevar8x32_epi32(*C0, inv_perm), inv_shuf);
    *C1 = _mm256_shuffle_epi8(_mm256_permutevar8x32_epi32(*C1, inv_perm), inv_shuf);
    *C2 = _mm256_shuffle_epi8(_mm256_permutevar8x32_epi32(*C2, inv_perm), inv_shuf);
    *C3 = _mm256_shuffle_epi8(_mm256_permutevar8x32_epi32(*C3, inv_perm), inv_shuf);
}

static void MatMulMatM32_GPT4_reg_ymms(
    __m256i A0, __m256i A1, __m256i A2, __m256i A3,
    __m256i B0, __m256i B1, __m256i B2, __m256i B3,
    __m256i *C0, __m256i *C1, __m256i *C2, __m256i *C3)
{
    MatMulMatM32_GPT4_rowdirect_core(A0, A1, A2, A3, B0, B1, B2, B3, C0, C1, C2, C3);
}

static inline uint32_t affinemix_m32_reverse32(uint32_t x)
{
    x = ((x & 0x55555555U) << 1) | ((x >> 1) & 0x55555555U);
    x = ((x & 0x33333333U) << 2) | ((x >> 2) & 0x33333333U);
    x = ((x & 0x0f0f0f0fU) << 4) | ((x >> 4) & 0x0f0f0f0fU);
    x = ((x & 0x00ff00ffU) << 8) | ((x >> 8) & 0x00ff00ffU);
    return (x << 16) | (x >> 16);
}

static inline uint32_t affinemix_m32_bitreverse32_mulnum_from_rows(
    __m256i A0, __m256i A1, __m256i A2, __m256i A3, uint32_t n)
{
    const __m512i x = _mm512_set1_epi32((int)n);
    const __m512i ones = _mm512_set1_epi32(1);
    const __m512i rows0 = _mm512_inserti64x4(_mm512_castsi256_si512(A0), A1, 1);
    const __m512i rows1 = _mm512_inserti64x4(_mm512_castsi256_si512(A2), A3, 1);
    const __m512i cnt0 = _mm512_popcnt_epi32(_mm512_and_si512(rows0, x));
    const __m512i cnt1 = _mm512_popcnt_epi32(_mm512_and_si512(rows1, x));
    const uint32_t raw = (uint32_t)_mm512_test_epi32_mask(cnt0, ones) |
                         ((uint32_t)_mm512_test_epi32_mask(cnt1, ones) << 16);

    return affinemix_m32_reverse32(raw);
}

static inline void affinemix_m32_rowdirect_matmul_from_rows(
    __m256i A0, __m256i A1, __m256i A2, __m256i A3,
    const M32 *Mat2, M32 *Mat)
{
    const __m256i B0 = _mm256_loadu_si256((const __m256i *)&Mat2->M[0]);
    const __m256i B1 = _mm256_loadu_si256((const __m256i *)&Mat2->M[8]);
    const __m256i B2 = _mm256_loadu_si256((const __m256i *)&Mat2->M[16]);
    const __m256i B3 = _mm256_loadu_si256((const __m256i *)&Mat2->M[24]);
    __m256i C0, C1, C2, C3;

    MatMulMatM32_GPT4_reg_ymms(A0, A1, A2, A3, B0, B1, B2, B3, &C0, &C1, &C2, &C3);

    _mm256_storeu_si256((__m256i *)&Mat->M[0], C0);
    _mm256_storeu_si256((__m256i *)&Mat->M[8], C1);
    _mm256_storeu_si256((__m256i *)&Mat->M[16], C2);
    _mm256_storeu_si256((__m256i *)&Mat->M[24], C3);
}

static inline void affinemix_m32_gfni_fused_core(const Aff32 *aff, const Aff32 *preaff_inv, Aff32 *mixaff)
{
    const __m256i A0 = _mm256_loadu_si256((const __m256i *)&aff->Mat.M[0]);
    const __m256i A1 = _mm256_loadu_si256((const __m256i *)&aff->Mat.M[8]);
    const __m256i A2 = _mm256_loadu_si256((const __m256i *)&aff->Mat.M[16]);
    const __m256i A3 = _mm256_loadu_si256((const __m256i *)&aff->Mat.M[24]);

    affinemix_m32_rowdirect_matmul_from_rows(A0, A1, A2, A3, &preaff_inv->Mat, &mixaff->Mat);
    mixaff->Vec.V = affinemix_m32_bitreverse32_mulnum_from_rows(A0, A1, A2, A3, preaff_inv->Vec.V) ^
                    aff->Vec.V;
}

void affinemixM32(Aff32 aff, Aff32 preaff_inv, Aff32 *mixaff)
{
    affinemix_m32_gfni_fused_core(&aff, &preaff_inv, mixaff);
}

static inline uint32_t m32_genaffinepair_reverse32(uint32_t x)
{
    x = ((x & 0x55555555U) << 1) | ((x >> 1) & 0x55555555U);
    x = ((x & 0x33333333U) << 2) | ((x >> 2) & 0x33333333U);
    x = ((x & 0x0f0f0f0fU) << 4) | ((x >> 4) & 0x0f0f0f0fU);
    x = ((x & 0x00ff00ffU) << 8) | ((x >> 8) & 0x00ff00ffU);
    return (x << 16) | (x >> 16);
}

static inline uint32_t m32_genaffinepair_bitreverse32_mulnum(const M32 *Mat, uint32_t n)
{
    const __m512i x = _mm512_set1_epi32((int)n);
    const __m512i ones = _mm512_set1_epi32(1);
    const __m512i rows0 = _mm512_loadu_si512((const void *)&Mat->M[0]);
    const __m512i rows1 = _mm512_loadu_si512((const void *)&Mat->M[16]);
    const __m512i cnt0 = _mm512_popcnt_epi32(_mm512_and_si512(rows0, x));
    const __m512i cnt1 = _mm512_popcnt_epi32(_mm512_and_si512(rows1, x));
    const uint32_t raw = (uint32_t)_mm512_test_epi32_mask(cnt0, ones) |
                         ((uint32_t)_mm512_test_epi32_mask(cnt1, ones) << 16);

    return m32_genaffinepair_reverse32(raw);
}

void genaffinepairM32(Aff32 *aff, Aff32 *aff_inv)
{
    genMatpairM32(
        &aff->Mat, &aff_inv->Mat);
    aff_inv->Vec.V = cus_random();
    aff->Vec.V = m32_genaffinepair_bitreverse32_mulnum(&aff->Mat, aff_inv->Vec.V);
}

static inline void affinemix_m4_gfni_fused_core(const Aff4 *aff, const Aff4 *preaff_inv, Aff4 *mixaff)
{
    uint32_t a_rows;
    uint32_t b_rows;
    memcpy(&a_rows, aff->Mat.M, sizeof(a_rows));
    memcpy(&b_rows, preaff_inv->Mat.M, sizeof(b_rows));
    a_rows &= 0x0f0f0f0fU;
    b_rows &= 0x0f0f0f0fU;

    __m128i a_exp = _mm_cvtsi32_si128((int)(a_rows << 4));
    __m128i b_exp = _mm_cvtsi32_si128((int)(b_rows << 4));
    __m128i i_trans = _mm_cvtsi64_si128((long long)0x0102040810204080ULL);

    __m128i b_trans = _mm_gf2p8affine_epi64_epi8(i_trans, b_exp, 0);
    __m128i c_exp = _mm_gf2p8affine_epi64_epi8(a_exp, b_trans, 0);
    uint32_t c_rows = ((uint32_t)_mm_cvtsi128_si32(c_exp) >> 4) & 0x0f0f0f0fU;
    memcpy(mixaff->Mat.M, &c_rows, sizeof(c_rows));

    __m128i vec_reg = _mm_cvtsi32_si128((int)(preaff_inv->Vec.V & 0x0fU));
    __m128i vec_out = _mm_gf2p8affine_epi64_epi8(vec_reg, _mm_cvtsi32_si128((int)a_rows), 0);
    mixaff->Vec.V = (uint8_t)((((uint32_t)_mm_cvtsi128_si32(vec_out) >> 4) & 0x0fU) ^ aff->Vec.V);
}

void affinemixM4(Aff4 aff, Aff4 preaff_inv, Aff4 *mixaff)
{
    Aff4 out;
    affinemix_m4_gfni_fused_core(&aff, &preaff_inv, &out);
    *mixaff = out;
}

static inline uint8_t m4_genaffinepair_gfni_mulnum(const M4 *Mat, uint8_t n)
{
    uint32_t rows;
    memcpy(&rows, Mat->M, sizeof(rows));

    const __m128i mat_reg = _mm_cvtsi32_si128((int)rows);
    const __m128i vec_reg = _mm_cvtsi32_si128((int)(n & 0x0fU));
    const __m128i out = _mm_gf2p8affine_epi64_epi8(vec_reg, mat_reg, 0);

    return (uint8_t)(((uint32_t)_mm_cvtsi128_si32(out) >> 4) & 0x0fU);
}

void genaffinepairM4(Aff4 *aff, Aff4 *aff_inv)
{
    genMatpairM4(
        &aff->Mat, &aff_inv->Mat);
    aff_inv->Vec.V = (uint8_t)(cus_random() & 0x0fU);
    aff->Vec.V = m4_genaffinepair_gfni_mulnum(&aff->Mat, aff_inv->Vec.V);
}

static inline __attribute__((always_inline)) __m512i pair_ymms_to_zmm_gpt12(
    __m256i lo, __m256i hi)
{
    return _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
}

static inline __attribute__((always_inline)) void prepare_m32pairbt_from_m64_dualhalf_gpt18(
    const M64 *Mat2, M64PreparedB_M32PairBT_GPT *PreparedB)
{
                                                                                                                                                                                                                                                                                              
    const __m512i I = _mm512_set1_epi64(0x0102040810204080ULL);
    const __m128i shuf_128 = _mm_setr_epi8(
        0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    const __m512i shuf_B = _mm512_broadcast_i32x4(shuf_128);
    const __m512i perm_B = _mm512_setr_epi32(
        0, 4, 1, 5, 2, 6, 3, 7,
        8, 12, 9, 13, 10, 14, 11, 15);

#define GPT18_PREPARE_ONE(B_ROWS, OUT_PAIRBT) do { \
        __m512i b_rows = _mm512_loadu_si512((const void *)(B_ROWS)); \
        __m256i b_left = _mm512_cvtepi64_epi32(_mm512_srli_epi64(b_rows, 32)); \
        __m256i b_right = _mm512_cvtepi64_epi32(b_rows); \
        __m512i pair = pair_ymms_to_zmm_gpt12(b_left, b_right); \
        (OUT_PAIRBT) = _mm512_gf2p8affine_epi64_epi8( \
            I, _mm512_permutexvar_epi32(perm_B, _mm512_shuffle_epi8(pair, shuf_B)), 0); \
    } while (0)

    for (int k = 0; k < 4; k++) {
        GPT18_PREPARE_ONE(&Mat2->M[k * 8], PreparedB->B00_B01[k]);
        GPT18_PREPARE_ONE(&Mat2->M[32 + k * 8], PreparedB->B10_B11[k]);
    }

#undef GPT18_PREPARE_ONE
}

static inline __attribute__((always_inline)) void m32_pair_prepared_product_xor2_inline_gpt14(
    const __m256i A0[4], const __m512i PairBT0[4],
    const __m256i A1[4], const __m512i PairBT1[4],
    __m512i Cpair[4])
{
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
    const __m512i I = _mm512_set1_epi64(0x0102040810204080ULL);
    const __m256i m0 = _mm256_setr_epi8(
        3, 7, 11, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        3, 7, 11, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i m1 = _mm256_setr_epi8(
        2, 6, 10, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        2, 6, 10, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i m2 = _mm256_setr_epi8(
        1, 5, 9, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        1, 5, 9, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i m3 = _mm256_setr_epi8(
        0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i perm_ctrl = _mm256_setr_epi32(0, 4, 0, 4, 0, 4, 0, 4);
    const __m512i inv_perm = _mm512_setr_epi32(
        0, 2, 4, 6, 1, 3, 5, 7,
        8, 10, 12, 14, 9, 11, 13, 15);
    const __m128i inv_shuf_128 = _mm_setr_epi8(
        0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    const __m512i inv_shuf = _mm512_broadcast_i32x4(inv_shuf_128);

#define GPT14_RAW_ACC(A_ROW, PAIRBT, OUT_RAW) do { \
        __m256i aik0_y = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8((A_ROW), m0), perm_ctrl); \
        __m256i aik1_y = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8((A_ROW), m1), perm_ctrl); \
        __m256i aik2_y = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8((A_ROW), m2), perm_ctrl); \
        __m256i aik3_y = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8((A_ROW), m3), perm_ctrl); \
        __m512i aik0 = pair_ymms_to_zmm_gpt12(aik0_y, aik0_y); \
        __m512i aik1 = pair_ymms_to_zmm_gpt12(aik1_y, aik1_y); \
        __m512i aik2 = pair_ymms_to_zmm_gpt12(aik2_y, aik2_y); \
        __m512i aik3 = pair_ymms_to_zmm_gpt12(aik3_y, aik3_y); \
        __m512i t0 = _mm512_gf2p8affine_epi64_epi8((PAIRBT)[0], aik0, 0); \
        __m512i t1 = _mm512_gf2p8affine_epi64_epi8((PAIRBT)[1], aik1, 0); \
        __m512i t2 = _mm512_gf2p8affine_epi64_epi8((PAIRBT)[2], aik2, 0); \
        __m512i t3 = _mm512_gf2p8affine_epi64_epi8((PAIRBT)[3], aik3, 0); \
        __m512i acc01 = _mm512_xor_si512(t0, t1); \
        __m512i acc23 = _mm512_xor_si512(t2, t3); \
        (OUT_RAW) = _mm512_xor_si512(acc01, acc23); \
    } while (0)

#define GPT14_CALC_XOR_ROW(A_LEFT, PB_LEFT, A_RIGHT, PB_RIGHT, C_OUT) do { \
        __m512i raw_left; \
        __m512i raw_right; \
        GPT14_RAW_ACC((A_LEFT), (PB_LEFT), raw_left); \
        GPT14_RAW_ACC((A_RIGHT), (PB_RIGHT), raw_right); \
        __m512i raw = _mm512_xor_si512(raw_left, raw_right); \
        __m512i c = _mm512_gf2p8affine_epi64_epi8(I, raw, 0); \
        (C_OUT) = _mm512_shuffle_epi8(_mm512_permutexvar_epi32(inv_perm, c), inv_shuf); \
    } while (0)

    GPT14_CALC_XOR_ROW(A0[0], PairBT0, A1[0], PairBT1, Cpair[0]);
    GPT14_CALC_XOR_ROW(A0[1], PairBT0, A1[1], PairBT1, Cpair[1]);
    GPT14_CALC_XOR_ROW(A0[2], PairBT0, A1[2], PairBT1, Cpair[2]);
    GPT14_CALC_XOR_ROW(A0[3], PairBT0, A1[3], PairBT1, Cpair[3]);

#undef GPT14_RAW_ACC
#undef GPT14_CALC_XOR_ROW
}

static inline __attribute__((always_inline)) void load_m64_half_as_m32_ymms_direct_gpt15(
    const uint64_t *rows, __m256i Aleft[4], __m256i Aright[4])
{
    for (int i = 0; i < 4; i++) {
        __m512i block = _mm512_loadu_si512((const void *)&rows[i * 8]);
        Aleft[i] = _mm512_cvtepi64_epi32(_mm512_srli_epi64(block, 32));
        Aright[i] = _mm512_cvtepi64_epi32(block);
    }
}

static inline __attribute__((always_inline)) void store_m32_pair_ymms_as_m64_half_gpt15(
    const __m512i Cpair[4], uint64_t *rows)
{
    for (int i = 0; i < 4; i++) {
        __m256i left = _mm512_castsi512_si256(Cpair[i]);
        __m256i right = _mm512_extracti64x4_epi64(Cpair[i], 1);
        __m512i out_rows = _mm512_or_si512(
            _mm512_slli_epi64(_mm512_cvtepu32_epi64(left), 32),
            _mm512_cvtepu32_epi64(right));
        _mm512_storeu_si512((void *)&rows[i * 8], out_rows);
    }
}

static void MatMulMatM64_GPT15_preparedB_m32pairbt_rawaccum_streamA(
    const M64 *Mat1, const M64PreparedB_M32PairBT_GPT *PreparedB, M64 *Mat)
{
    __m256i Aleft[4], Aright[4];
    __m512i Cpair[4];

    load_m64_half_as_m32_ymms_direct_gpt15(&Mat1->M[0], Aleft, Aright);
    m32_pair_prepared_product_xor2_inline_gpt14(
        Aleft, PreparedB->B00_B01, Aright, PreparedB->B10_B11, Cpair);
    store_m32_pair_ymms_as_m64_half_gpt15(Cpair, &Mat->M[0]);

    load_m64_half_as_m32_ymms_direct_gpt15(&Mat1->M[32], Aleft, Aright);
    m32_pair_prepared_product_xor2_inline_gpt14(
        Aleft, PreparedB->B00_B01, Aright, PreparedB->B10_B11, Cpair);
    store_m32_pair_ymms_as_m64_half_gpt15(Cpair, &Mat->M[32]);
}

static void MatMulMatM64_GPT19_m32pairbt_rawaccum_streamA_dualprepare_total(
    const M64 *Mat1, const M64 *Mat2, M64 *Mat)
{
    M64PreparedB_M32PairBT_GPT PreparedB __attribute__((aligned(64)));

    prepare_m32pairbt_from_m64_dualhalf_gpt18(Mat2, &PreparedB);
    MatMulMatM64_GPT15_preparedB_m32pairbt_rawaccum_streamA(Mat1, &PreparedB, Mat);
}

void affinemixM64(Aff64 aff, Aff64 preaff_inv, Aff64 *mixaff)
{
    MatMulMatM64_GPT19_m32pairbt_rawaccum_streamA_dualprepare_total(&aff.Mat, &preaff_inv.Mat,
                                                                    &mixaff->Mat);
    mixaff->Vec.V = MatMulNumM64_GPT6_avx512_vpopcntdq_movmask_ptr(&aff.Mat, preaff_inv.Vec.V) ^
                    aff.Vec.V;
}

void genaffinepairM64(Aff64 *aff, Aff64 *aff_inv)
{
    genMatpairM64(
        &aff->Mat, &aff_inv->Mat);
    uint32_t *v = (uint32_t *)&aff->Vec.V;
    v[1] = cus_random();
    v[0] = cus_random();
    aff_inv->Vec.V = MatMulNumM64_GPT6_avx512_vpopcntdq_movmask_ptr(&aff_inv->Mat, aff->Vec.V);
}

static inline void affinemix_m8_gfni_fused_core(const Aff8 *aff, const Aff8 *preaff_inv, Aff8 *mixaff)
{
    const uint64_t i_transpose = 0x0102040810204080ULL;
    uint64_t a_rows;
    uint64_t b_rows;
    memcpy(&a_rows, aff->Mat.M, sizeof(a_rows));
    memcpy(&b_rows, preaff_inv->Mat.M, sizeof(b_rows));

    __m128i a = _mm_cvtsi64_si128((long long)a_rows);
    __m128i b = _mm_cvtsi64_si128((long long)b_rows);
    __m128i i = _mm_cvtsi64_si128((long long)i_transpose);

    __m128i b_t = _mm_gf2p8affine_epi64_epi8(i, b, 0);
    __m128i c_rows = _mm_gf2p8affine_epi64_epi8(a, b_t, 0);
    _mm_storel_epi64((__m128i *)mixaff->Mat.M, c_rows);

    __m128i vec = _mm_cvtsi32_si128((int)preaff_inv->Vec.V);
    __m128i vec_out = _mm_gf2p8affine_epi64_epi8(vec, a, 0);
    mixaff->Vec.V = (uint8_t)((uint8_t)_mm_cvtsi128_si32(vec_out) ^ aff->Vec.V);
}

void affinemixM8(Aff8 aff, Aff8 preaff_inv, Aff8 *mixaff)
{
    affinemix_m8_gfni_fused_core(&aff, &preaff_inv, mixaff);
}

static inline uint8_t m8_genaffinepair_gfni_u64load_mulnum(const M8 *Mat, uint8_t n)
{
    uint64_t rows;
    memcpy(&rows, Mat->M, sizeof(rows));

    __m128i x = _mm_cvtsi32_si128((int)n);
    __m128i mat = _mm_cvtsi64_si128((long long)rows);
    __m128i out = _mm_gf2p8affine_epi64_epi8(x, mat, 0);

    return (uint8_t)_mm_cvtsi128_si32(out);
}

void genaffinepairM8(Aff8 *aff, Aff8 *aff_inv)
{
    genMatpairM8(
        &aff->Mat, &aff_inv->Mat);
    aff->Vec.V = (uint8_t)cus_random();
    aff_inv->Vec.V = m8_genaffinepair_gfni_u64load_mulnum(&aff_inv->Mat, aff->Vec.V);
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

static void MatMulVecM128_GPT5_avx512_vpopcntdq_pairfold_ptr(const M128 *Mat, const V128 *Vec, V128 *ans)
{
    m128_vpopcntdq_pairfold_core(Mat, Vec, ans);
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

static uint64_t MatMulNumM64_GPT6_avx512_vpopcntdq_movmask_ptr(const M64 *Mat, uint64_t n)
{
    return m64_avx512_vpopcntdq_movmask_core(Mat, n);
}

