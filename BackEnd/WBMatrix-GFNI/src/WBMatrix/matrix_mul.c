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
static const uint8_t pack_vpermb_mask_arr_256_opt2[64] __attribute__((aligned(64))) = {
    56, 48, 40, 32, 24, 16,  8, 0,
    57, 49, 41, 33, 25, 17,  9, 1,
    58, 50, 42, 34, 26, 18, 10, 2,
    59, 51, 43, 35, 27, 19, 11, 3,
    60, 52, 44, 36, 28, 20, 12, 4,
    61, 53, 45, 37, 29, 21, 13, 5,
    62, 54, 46, 38, 30, 22, 14, 6,
    63, 55, 47, 39, 31, 23, 15, 7
};
static const uint8_t vpermb_mask_arr_256_opt2[64] __attribute__((aligned(64))) = {
    7, 15, 23, 31, 39, 47, 55, 63,
    6, 14, 22, 30, 38, 46, 54, 62,
    5, 13, 21, 29, 37, 45, 53, 61,
    4, 12, 20, 28, 36, 44, 52, 60,
    3, 11, 19, 27, 35, 43, 51, 59,
    2, 10, 18, 26, 34, 42, 50, 58,
    1,  9, 17, 25, 33, 41, 49, 57,
    0,  8, 16, 24, 32, 40, 48, 56
};

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
static void MatMulMatM128_GPT10_preparedB_8x8tiles_rowdirect(
    const M128 *Mat1, const M128PreparedB_8x8Tiles_GPT *PreparedB, M128 *Mat);
void MatMulMatM128(M128 Mat1, M128 Mat2, M128 *Mat);
static inline __m128i load_and_duplicate_128(const M16 *Mat, int offset_rows, int use_low_bytes);
static inline __m128i load_B_pair_128(const M16 *Mat, int offset_rows);
static inline __m128i gfni_mul_128(__m128i A_dual, __m128i B_pair, __m128i I);
void MatMulMatM16(M16 Mat1, M16 Mat2, M16 *Mat);
static inline void extract_256_blocks_simd_opt2(const M256 *in, int r_start, 
                                           __m512i *out0, __m512i *out1, 
                                           __m512i *out2, __m512i *out3, 
                                           __m512i vpermb_mask);
static inline void pack_256_blocks_simd_opt2(__m512i in0, __m512i in1, __m512i in2, __m512i in3,
                                        M256 *out, int r_start, __m512i pack_mask);
static inline void matmul_m256_opt3_core(const M256 *Mat1, const M256 *Mat2, M256 *Mat);
void MatMulMatM256(M256 Mat1, M256 Mat2, M256 *Mat);
static inline void MatMulMatM32_GPT4_rowdirect_core(
    __m256i A0, __m256i A1, __m256i A2, __m256i A3,
    __m256i B0, __m256i B1, __m256i B2, __m256i B3,
    __m256i *C0, __m256i *C1, __m256i *C2, __m256i *C3);
static void MatMulMatM32_GPT4_rowdirect_ptr(const M32 *Mat1, const M32 *Mat2, M32 *Mat);
void MatMulMatM32(M32 Mat1, M32 Mat2, M32 *Mat);
static inline void m4_gpt5_gfni_lowqword_core(const M4 *Mat1, const M4 *Mat2, M4 *Mat);
void MatMulMatM4(M4 Mat1, M4 Mat2, M4 *Mat);
static inline __attribute__((always_inline)) __m512i pair_ymms_to_zmm_gpt12(
    __m256i lo, __m256i hi);
static inline __attribute__((always_inline)) void load_m64_as_m32_ymms_direct_gpt12(
    const M64 *A,
    __m256i A00[4], __m256i A01[4], __m256i A10[4], __m256i A11[4]);
static inline __attribute__((always_inline)) void store_m32_pair_ymms_as_m64_direct_gpt12(
    const __m512i Ctop[4], const __m512i Cbottom[4], M64 *C);
static inline __attribute__((always_inline)) void prepare_m32pairbt_from_m64_half_gpt12(
    const uint64_t *rows, __m512i PairBT[4]);
static void MatMulMatM64_GPT12_prepareB_m32pairbt(
    const M64 *Mat2, M64PreparedB_M32PairBT_GPT *PreparedB);
static inline __attribute__((always_inline)) void m32_pair_prepared_product_xor2_inline_gpt14(
    const __m256i A0[4], const __m512i PairBT0[4],
    const __m256i A1[4], const __m512i PairBT1[4],
    __m512i Cpair[4]);
static void MatMulMatM64_GPT14_preparedB_m32pairbt_rawaccum(
    const M64 *Mat1, const M64PreparedB_M32PairBT_GPT *PreparedB, M64 *Mat);
static void MatMulMatM64_GPT14_m32pairbt_rawaccum_total(
    const M64 *Mat1, const M64 *Mat2, M64 *Mat);
void MatMulMatM64(M64 Mat1, M64 Mat2, M64 *Mat);
void MatMulMatM8(M8 Mat1, M8 Mat2, M8 *Mat);
static inline void MatMulMatM8_GPT3_2gfni_core(const uint8_t *A_ptr, const uint8_t *B_ptr, uint8_t *C_ptr);
void wbmatrix2_matmul_m8_two_gfni(M8 Mat1, M8 Mat2, M8 *Mat);

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

static void MatMulMatM128_GPT10_preparedB_8x8tiles_rowdirect(
    const M128 *Mat1, const M128PreparedB_8x8Tiles_GPT *PreparedB, M128 *Mat)
{
    M128TileCtx_GPT ctx = make_m128_tile_ctx_gpt();

    for (int i = 0; i < 16; i++) {
        __m512i a_left, a_right;
        __m512i c_left = _mm512_setzero_si512();
        __m512i c_right = _mm512_setzero_si512();

        extract_m128_8x8tiles_gpt(Mat1, i * 8, &a_left, &a_right,
                                  ctx.idx_left, ctx.idx_right, ctx.vpermb_mask);

        for (int k = 0; k < 8; k++) {
            __m512i a_broadcast = broadcast_a_tile_lane_gpt(a_left, &ctx, k);
            c_left = _mm512_xor_si512(
                c_left,
                _mm512_gf2p8affine_epi64_epi8(a_broadcast, PreparedB->BT[k][0], 0));
            c_right = _mm512_xor_si512(
                c_right,
                _mm512_gf2p8affine_epi64_epi8(a_broadcast, PreparedB->BT[k][1], 0));
        }
        for (int k = 0; k < 8; k++) {
            __m512i a_broadcast = broadcast_a_tile_lane_gpt(a_right, &ctx, k);
            c_left = _mm512_xor_si512(
                c_left,
                _mm512_gf2p8affine_epi64_epi8(a_broadcast, PreparedB->BT[k + 8][0], 0));
            c_right = _mm512_xor_si512(
                c_right,
                _mm512_gf2p8affine_epi64_epi8(a_broadcast, PreparedB->BT[k + 8][1], 0));
        }

        pack_m128_8x8tiles_gpt(c_left, c_right, Mat, i * 8,
                               ctx.pack_mask, ctx.idx_pack0, ctx.idx_pack1);
    }
}

void MatMulMatM128(M128 Mat1, M128 Mat2, M128 *Mat) {
    M128PreparedB_8x8Tiles_GPT PreparedB;
    MatMulMatM128_GPT5_prepareB_8x8tiles(&Mat2, &PreparedB);
    MatMulMatM128_GPT10_preparedB_8x8tiles_rowdirect(&Mat1, &PreparedB, Mat);
}

static inline __m128i load_and_duplicate_128(const M16 *Mat, int offset_rows, int use_low_bytes) {
                             
                                                   
    __m128i raw = _mm_loadu_si128((const __m128i*)((const uint8_t*)Mat->M + offset_rows * 2));
    
    __m128i mask;
    if (use_low_bytes) {
                                     
                                     
        mask = _mm_setr_epi8(
            0, 2, 4, 6, 8, 10, 12, 14, 
            0, 2, 4, 6, 8, 10, 12, 14
        );
    } else {
                                     
                                     
        mask = _mm_setr_epi8(
            1, 3, 5, 7, 9, 11, 13, 15, 
            1, 3, 5, 7, 9, 11, 13, 15
        );
    }
    return _mm_shuffle_epi8(raw, mask);
}

static inline __m128i load_B_pair_128(const M16 *Mat, int offset_rows) {
    __m128i raw = _mm_loadu_si128((const __m128i*)((const uint8_t*)Mat->M + offset_rows * 2));
    
                                
                               
    __m128i mask = _mm_setr_epi8(
        1, 3, 5, 7, 9, 11, 13, 15,           
        0, 2, 4, 6, 8, 10, 12, 14            
    );
    return _mm_shuffle_epi8(raw, mask);
}

static inline __m128i gfni_mul_128(__m128i A_dual, __m128i B_pair, __m128i I) {
    __m128i B_t = _mm_gf2p8affine_epi64_epi8(I, B_pair, 0);
    __m128i C_c = _mm_gf2p8affine_epi64_epi8(B_t, A_dual, 0);
    return _mm_gf2p8affine_epi64_epi8(I, C_c, 0);
}

void MatMulMatM16(M16 Mat1, M16 Mat2, M16 *Mat)
{
    const uint64_t I_val = 0x0102040810204080ULL;
    __m128i I = _mm_set1_epi64x((long long)I_val);

                     
                            
                            
    __m128i B_top = load_B_pair_128(&Mat2, 0);
    __m128i B_bot = load_B_pair_128(&Mat2, 8);

                                                       
                                                    
    
                                                 
    __m128i A00_dual = load_and_duplicate_128(&Mat1, 0, 0); 
    __m128i Res_top_1 = gfni_mul_128(A00_dual, B_top, I);
    
                                                
    __m128i A01_dual = load_and_duplicate_128(&Mat1, 0, 1);
    __m128i Res_top_2 = gfni_mul_128(A01_dual, B_bot, I);
    
    __m128i C_top = _mm_xor_si128(Res_top_1, Res_top_2);

                                                        
                                                    
    
    __m128i A10_dual = load_and_duplicate_128(&Mat1, 8, 0);
    __m128i Res_bot_1 = gfni_mul_128(A10_dual, B_top, I);
    
    __m128i A11_dual = load_and_duplicate_128(&Mat1, 8, 1);
    __m128i Res_bot_2 = gfni_mul_128(A11_dual, B_bot, I);
    
    __m128i C_bot = _mm_xor_si128(Res_bot_1, Res_bot_2);

                                              
                                             
                                                            
                                              
                                                      
                                                       
                             
    
    __m128i final_store_mask = _mm_setr_epi8(
        8, 0, 9, 1, 10, 2, 11, 3, 12, 4, 13, 5, 14, 6, 15, 7
    );
    
    __m128i C_top_formatted = _mm_shuffle_epi8(C_top, final_store_mask);
    __m128i C_bot_formatted = _mm_shuffle_epi8(C_bot, final_store_mask);
    
    _mm_storeu_si128((__m128i*)Mat->M, C_top_formatted);
    _mm_storeu_si128((__m128i*)(Mat->M + 8), C_bot_formatted);
}

static inline void extract_256_blocks_simd_opt2(const M256 *in, int r_start, 
                                           __m512i *out0, __m512i *out1, 
                                           __m512i *out2, __m512i *out3, 
                                           __m512i vpermb_mask) {
    __m512i raw0 = _mm512_set_epi64(in->M[r_start+7][0], in->M[r_start+6][0], in->M[r_start+5][0], in->M[r_start+4][0],
                                    in->M[r_start+3][0], in->M[r_start+2][0], in->M[r_start+1][0], in->M[r_start][0]);
    __m512i raw1 = _mm512_set_epi64(in->M[r_start+7][1], in->M[r_start+6][1], in->M[r_start+5][1], in->M[r_start+4][1],
                                    in->M[r_start+3][1], in->M[r_start+2][1], in->M[r_start+1][1], in->M[r_start][1]);
    __m512i raw2 = _mm512_set_epi64(in->M[r_start+7][2], in->M[r_start+6][2], in->M[r_start+5][2], in->M[r_start+4][2],
                                    in->M[r_start+3][2], in->M[r_start+2][2], in->M[r_start+1][2], in->M[r_start][2]);
    __m512i raw3 = _mm512_set_epi64(in->M[r_start+7][3], in->M[r_start+6][3], in->M[r_start+5][3], in->M[r_start+4][3],
                                    in->M[r_start+3][3], in->M[r_start+2][3], in->M[r_start+1][3], in->M[r_start][3]);

    *out0 = _mm512_permutexvar_epi8(vpermb_mask, raw0);
    *out1 = _mm512_permutexvar_epi8(vpermb_mask, raw1);
    *out2 = _mm512_permutexvar_epi8(vpermb_mask, raw2);
    *out3 = _mm512_permutexvar_epi8(vpermb_mask, raw3);
}

static inline void pack_256_blocks_simd_opt2(__m512i in0, __m512i in1, __m512i in2, __m512i in3,
                                        M256 *out, int r_start, __m512i pack_mask) {
    __m512i raw0 = _mm512_permutexvar_epi8(pack_mask, in0);
    __m512i raw1 = _mm512_permutexvar_epi8(pack_mask, in1);
    __m512i raw2 = _mm512_permutexvar_epi8(pack_mask, in2);
    __m512i raw3 = _mm512_permutexvar_epi8(pack_mask, in3);

    uint64_t v0[8] __attribute__((aligned(64)));
    uint64_t v1[8] __attribute__((aligned(64)));
    uint64_t v2[8] __attribute__((aligned(64)));
    uint64_t v3[8] __attribute__((aligned(64)));
    
    _mm512_store_si512((__m512i*)v0, raw0);
    _mm512_store_si512((__m512i*)v1, raw1);
    _mm512_store_si512((__m512i*)v2, raw2);
    _mm512_store_si512((__m512i*)v3, raw3);

    for (int i = 0; i < 8; i++) {
        out->M[r_start + i][0] = v0[i];
        out->M[r_start + i][1] = v1[i];
        out->M[r_start + i][2] = v2[i];
        out->M[r_start + i][3] = v3[i];
    }
}

static inline void matmul_m256_opt3_core(const M256 *Mat1, const M256 *Mat2, M256 *Mat) {
    const uint64_t I_transpose = 0x0102040810204080ULL;
    __m512i I_trans_512 = _mm512_set1_epi64((long long)I_transpose);

    __m512i vpermb_mask = _mm512_load_si512((__m512i*)vpermb_mask_arr_256_opt2);
    __m512i pack_mask   = _mm512_load_si512((__m512i*)pack_vpermb_mask_arr_256_opt2);

                                           
    uint64_t B_trans[32][32] __attribute__((aligned(64)));
    
                                                                      
    for(int K = 0; K < 32; K++) {
        __m512i b0, b1, b2, b3;
        extract_256_blocks_simd_opt2(Mat2, K * 8, &b0, &b1, &b2, &b3, vpermb_mask);
        
        _mm512_store_si512((__m512i*)&B_trans[K][0],  _mm512_gf2p8affine_epi64_epi8(I_trans_512, b0, 0));
        _mm512_store_si512((__m512i*)&B_trans[K][8],  _mm512_gf2p8affine_epi64_epi8(I_trans_512, b1, 0));
        _mm512_store_si512((__m512i*)&B_trans[K][16], _mm512_gf2p8affine_epi64_epi8(I_trans_512, b2, 0));
        _mm512_store_si512((__m512i*)&B_trans[K][24], _mm512_gf2p8affine_epi64_epi8(I_trans_512, b3, 0));
    }

                                                                                                

                                                                                            
    for(int I = 0; I < 32; I += 2) {
        
                                                                                    
        uint64_t A_row0[32] __attribute__((aligned(64)));
        uint64_t A_row1[32] __attribute__((aligned(64)));
        __m512i a0_0, a0_1, a0_2, a0_3;
        __m512i a1_0, a1_1, a1_2, a1_3;

        extract_256_blocks_simd_opt2(Mat1, I * 8, &a0_0, &a0_1, &a0_2, &a0_3, vpermb_mask);
        _mm512_store_si512((__m512i*)&A_row0[0], a0_0);
        _mm512_store_si512((__m512i*)&A_row0[8], a0_1);
        _mm512_store_si512((__m512i*)&A_row0[16], a0_2);
        _mm512_store_si512((__m512i*)&A_row0[24], a0_3);

        extract_256_blocks_simd_opt2(Mat1, (I + 1) * 8, &a1_0, &a1_1, &a1_2, &a1_3, vpermb_mask);
        _mm512_store_si512((__m512i*)&A_row1[0], a1_0);
        _mm512_store_si512((__m512i*)&A_row1[8], a1_1);
        _mm512_store_si512((__m512i*)&A_row1[16], a1_2);
        _mm512_store_si512((__m512i*)&A_row1[24], a1_3);

                                                          
        __m512i C0_0 = _mm512_setzero_si512();
        __m512i C0_1 = _mm512_setzero_si512();
        __m512i C0_2 = _mm512_setzero_si512();
        __m512i C0_3 = _mm512_setzero_si512();

                                                                  
        __m512i C1_0 = _mm512_setzero_si512();
        __m512i C1_1 = _mm512_setzero_si512();
        __m512i C1_2 = _mm512_setzero_si512();
        __m512i C1_3 = _mm512_setzero_si512();

        #pragma GCC unroll 4
        for(int K = 0; K < 32; K++) {
                                                                                                                    
                                                                          
            __m512i b0 = _mm512_load_si512((__m512i*)&B_trans[K][0]);
            __m512i b1 = _mm512_load_si512((__m512i*)&B_trans[K][8]);
            __m512i b2 = _mm512_load_si512((__m512i*)&B_trans[K][16]);
            __m512i b3 = _mm512_load_si512((__m512i*)&B_trans[K][24]);

                              
            __m512i A_brd0 = _mm512_set1_epi64((long long)A_row0[K]);
            C0_0 = _mm512_xor_si512(C0_0, _mm512_gf2p8affine_epi64_epi8(A_brd0, b0, 0));
            C0_1 = _mm512_xor_si512(C0_1, _mm512_gf2p8affine_epi64_epi8(A_brd0, b1, 0));
            C0_2 = _mm512_xor_si512(C0_2, _mm512_gf2p8affine_epi64_epi8(A_brd0, b2, 0));
            C0_3 = _mm512_xor_si512(C0_3, _mm512_gf2p8affine_epi64_epi8(A_brd0, b3, 0));

                                                                                                               
            __m512i A_brd1 = _mm512_set1_epi64((long long)A_row1[K]);
            C1_0 = _mm512_xor_si512(C1_0, _mm512_gf2p8affine_epi64_epi8(A_brd1, b0, 0));
            C1_1 = _mm512_xor_si512(C1_1, _mm512_gf2p8affine_epi64_epi8(A_brd1, b1, 0));
            C1_2 = _mm512_xor_si512(C1_2, _mm512_gf2p8affine_epi64_epi8(A_brd1, b2, 0));
            C1_3 = _mm512_xor_si512(C1_3, _mm512_gf2p8affine_epi64_epi8(A_brd1, b3, 0));
        }

                                   
        pack_256_blocks_simd_opt2(C0_0, C0_1, C0_2, C0_3, Mat, I * 8, pack_mask);
        pack_256_blocks_simd_opt2(C1_0, C1_1, C1_2, C1_3, Mat, (I + 1) * 8, pack_mask);
    }
}

void MatMulMatM256(M256 Mat1, M256 Mat2, M256 *Mat) {
    matmul_m256_opt3_core(&Mat1, &Mat2, Mat);
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

static void MatMulMatM32_GPT4_rowdirect_ptr(const M32 *Mat1, const M32 *Mat2, M32 *Mat)
{
    __m256i A0 = _mm256_loadu_si256((const __m256i *)&Mat1->M[0]);
    __m256i A1 = _mm256_loadu_si256((const __m256i *)&Mat1->M[8]);
    __m256i A2 = _mm256_loadu_si256((const __m256i *)&Mat1->M[16]);
    __m256i A3 = _mm256_loadu_si256((const __m256i *)&Mat1->M[24]);
    __m256i B0 = _mm256_loadu_si256((const __m256i *)&Mat2->M[0]);
    __m256i B1 = _mm256_loadu_si256((const __m256i *)&Mat2->M[8]);
    __m256i B2 = _mm256_loadu_si256((const __m256i *)&Mat2->M[16]);
    __m256i B3 = _mm256_loadu_si256((const __m256i *)&Mat2->M[24]);
    __m256i C0, C1, C2, C3;

    MatMulMatM32_GPT4_rowdirect_core(A0, A1, A2, A3, B0, B1, B2, B3, &C0, &C1, &C2, &C3);

    _mm256_storeu_si256((__m256i *)&Mat->M[0], C0);
    _mm256_storeu_si256((__m256i *)&Mat->M[8], C1);
    _mm256_storeu_si256((__m256i *)&Mat->M[16], C2);
    _mm256_storeu_si256((__m256i *)&Mat->M[24], C3);
}

void MatMulMatM32(M32 Mat1, M32 Mat2, M32 *Mat)
{
    MatMulMatM32_GPT4_rowdirect_ptr(&Mat1, &Mat2, Mat);
}

static inline void m4_gpt5_gfni_lowqword_core(const M4 *Mat1, const M4 *Mat2, M4 *Mat)
{
    uint32_t a_32;
    uint32_t b_32;
    memcpy(&a_32, Mat1->M, sizeof(a_32));
    memcpy(&b_32, Mat2->M, sizeof(b_32));
    a_32 &= 0x0f0f0f0fU;
    b_32 &= 0x0f0f0f0fU;

    __m128i A_exp = _mm_cvtsi32_si128((int)(a_32 << 4));
    __m128i B_exp = _mm_cvtsi32_si128((int)(b_32 << 4));
    __m128i I_trans_reg = _mm_cvtsi64_si128((long long)0x0102040810204080ULL);

    __m128i B_trans_reg = _mm_gf2p8affine_epi64_epi8(I_trans_reg, B_exp, 0);
    __m128i C_exp = _mm_gf2p8affine_epi64_epi8(A_exp, B_trans_reg, 0);
    uint32_t c_rows = ((uint32_t)_mm_cvtsi128_si32(C_exp) >> 4) & 0x0f0f0f0fU;
    memcpy(Mat->M, &c_rows, sizeof(c_rows));
}

void MatMulMatM4(M4 Mat1, M4 Mat2, M4 *Mat)
{
    m4_gpt5_gfni_lowqword_core(&Mat1, &Mat2, Mat);
}

static inline __attribute__((always_inline)) __m512i pair_ymms_to_zmm_gpt12(
    __m256i lo, __m256i hi)
{
    return _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
}

static inline __attribute__((always_inline)) void load_m64_as_m32_ymms_direct_gpt12(
    const M64 *A,
    __m256i A00[4], __m256i A01[4], __m256i A10[4], __m256i A11[4])
{
                                                                                                                                                                                                                                                                                                                                                                                                                                                        
    for (int i = 0; i < 4; i++) {
        __m512i top = _mm512_loadu_si512((const void *)&A->M[i * 8]);
        __m512i bottom = _mm512_loadu_si512((const void *)&A->M[32 + i * 8]);

        A00[i] = _mm512_cvtepi64_epi32(_mm512_srli_epi64(top, 32));
        A01[i] = _mm512_cvtepi64_epi32(top);
        A10[i] = _mm512_cvtepi64_epi32(_mm512_srli_epi64(bottom, 32));
        A11[i] = _mm512_cvtepi64_epi32(bottom);
    }
}

static inline __attribute__((always_inline)) void store_m32_pair_ymms_as_m64_direct_gpt12(
    const __m512i Ctop[4], const __m512i Cbottom[4], M64 *C)
{
                                                                                                                                                                                                                                                                                                                                                                                        
    for (int i = 0; i < 4; i++) {
        __m256i top_l = _mm512_castsi512_si256(Ctop[i]);
        __m256i top_r = _mm512_extracti64x4_epi64(Ctop[i], 1);
        __m512i top_rows = _mm512_or_si512(
            _mm512_slli_epi64(_mm512_cvtepu32_epi64(top_l), 32),
            _mm512_cvtepu32_epi64(top_r));

        __m256i bottom_l = _mm512_castsi512_si256(Cbottom[i]);
        __m256i bottom_r = _mm512_extracti64x4_epi64(Cbottom[i], 1);
        __m512i bottom_rows = _mm512_or_si512(
            _mm512_slli_epi64(_mm512_cvtepu32_epi64(bottom_l), 32),
            _mm512_cvtepu32_epi64(bottom_r));

        _mm512_storeu_si512((void *)&C->M[i * 8], top_rows);
        _mm512_storeu_si512((void *)&C->M[32 + i * 8], bottom_rows);
    }
}

static inline __attribute__((always_inline)) void prepare_m32pairbt_from_m64_half_gpt12(
    const uint64_t *rows, __m512i PairBT[4])
{
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
    const __m512i I = _mm512_set1_epi64(0x0102040810204080ULL);
    const __m128i shuf_128 = _mm_setr_epi8(
        0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    const __m512i shuf_B = _mm512_broadcast_i32x4(shuf_128);
    const __m512i perm_B = _mm512_setr_epi32(
        0, 4, 1, 5, 2, 6, 3, 7,
        8, 12, 9, 13, 10, 14, 11, 15);

    for (int k = 0; k < 4; k++) {
        __m512i b_rows = _mm512_loadu_si512((const void *)&rows[k * 8]);
        __m256i b_left = _mm512_cvtepi64_epi32(_mm512_srli_epi64(b_rows, 32));
        __m256i b_right = _mm512_cvtepi64_epi32(b_rows);
        __m512i pair = pair_ymms_to_zmm_gpt12(b_left, b_right);

        PairBT[k] = _mm512_gf2p8affine_epi64_epi8(
            I, _mm512_permutexvar_epi32(perm_B, _mm512_shuffle_epi8(pair, shuf_B)), 0);
    }
}

static void MatMulMatM64_GPT12_prepareB_m32pairbt(
    const M64 *Mat2, M64PreparedB_M32PairBT_GPT *PreparedB)
{
    prepare_m32pairbt_from_m64_half_gpt12(&Mat2->M[0], PreparedB->B00_B01);
    prepare_m32pairbt_from_m64_half_gpt12(&Mat2->M[32], PreparedB->B10_B11);
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

static void MatMulMatM64_GPT14_preparedB_m32pairbt_rawaccum(
    const M64 *Mat1, const M64PreparedB_M32PairBT_GPT *PreparedB, M64 *Mat)
{
    __m256i A00[4], A01[4], A10[4], A11[4];
    __m512i Ctop[4], Cbottom[4];

    load_m64_as_m32_ymms_direct_gpt12(Mat1, A00, A01, A10, A11);

    m32_pair_prepared_product_xor2_inline_gpt14(
        A00, PreparedB->B00_B01, A01, PreparedB->B10_B11, Ctop);
    m32_pair_prepared_product_xor2_inline_gpt14(
        A10, PreparedB->B00_B01, A11, PreparedB->B10_B11, Cbottom);

    store_m32_pair_ymms_as_m64_direct_gpt12(Ctop, Cbottom, Mat);
}

static void MatMulMatM64_GPT14_m32pairbt_rawaccum_total(
    const M64 *Mat1, const M64 *Mat2, M64 *Mat)
{
    M64PreparedB_M32PairBT_GPT PreparedB;

    MatMulMatM64_GPT12_prepareB_m32pairbt(Mat2, &PreparedB);
    MatMulMatM64_GPT14_preparedB_m32pairbt_rawaccum(Mat1, &PreparedB, Mat);
}

void MatMulMatM64(M64 Mat1, M64 Mat2, M64 *Mat)
{
    MatMulMatM64_GPT14_m32pairbt_rawaccum_total(&Mat1, &Mat2, Mat);
}

void MatMulMatM8(M8 Mat1, M8 Mat2, M8 *Mat)
{
                                                       
    const uint64_t I_transpose = 0x0102040810204080ULL;
    __m128i I_transpose_xmm = _mm_set1_epi64x((long long)I_transpose);
    
                                    
    __m128i B_xmm = _mm_loadl_epi64((__m128i*)&Mat2.M[0]);
    __m128i B_trans_xmm = _mm_gf2p8affine_epi64_epi8(I_transpose_xmm, B_xmm, 0);
    
                                                        
    __m128i A_xmm = _mm_loadl_epi64((__m128i*)&Mat1.M[0]);
    __m128i C_cols_xmm = _mm_gf2p8affine_epi64_epi8(B_trans_xmm, A_xmm, 0);
    
                                                                                      
                                                           
                                                           
    __m128i C_rows_xmm = _mm_gf2p8affine_epi64_epi8(I_transpose_xmm, C_cols_xmm, 0);
    
                      
    _mm_storel_epi64((__m128i*)&(*Mat).M[0], C_rows_xmm);
}

static inline void MatMulMatM8_GPT3_2gfni_core(const uint8_t *A_ptr, const uint8_t *B_ptr, uint8_t *C_ptr)
{
    const uint64_t I_transpose = 0x0102040810204080ULL;
    __m128i I = _mm_set1_epi64x((long long)I_transpose);

    __m128i A = _mm_loadl_epi64((const __m128i *)A_ptr);
    __m128i B = _mm_loadl_epi64((const __m128i *)B_ptr);

    __m128i B_t = _mm_gf2p8affine_epi64_epi8(I, B, 0);
    __m128i C_rows = _mm_gf2p8affine_epi64_epi8(A, B_t, 0);

    _mm_storel_epi64((__m128i *)C_ptr, C_rows);
}

void wbmatrix2_matmul_m8_two_gfni(M8 Mat1, M8 Mat2, M8 *Mat)
{
    MatMulMatM8_GPT3_2gfni_core(Mat1.M, Mat2.M, Mat->M);
}


void wbmatrix2_matmul_m8_three_gfni(M8 left, M8 right, M8 *out)
{
    MatMulMatM8(left, right, out);
}
