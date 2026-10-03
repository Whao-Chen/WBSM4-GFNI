#include "internal.h"


static void wbmatrix2_rand_m32_no_reseed(M32 *matrix)
{
    unsigned int row;
    for (row = 0; row < 32; ++row) {
        matrix->M[row] = cus_random();
    }
}

static void wbmatrix2_rand_m64_no_reseed(M64 *matrix)
{
    unsigned int row;
    for (row = 0; row < 64; ++row) {
        uint32_t *words = (uint32_t *)&matrix->M[row];
        words[1] = cus_random();
        words[0] = cus_random();
    }
}

static void wbmatrix2_rand_m128_no_reseed(M128 *matrix)
{
    unsigned int row;
    for (row = 0; row < 128; ++row) {
        uint32_t *words = (uint32_t *)&matrix->M[row][0];
        words[1] = cus_random();
        words[0] = cus_random();
        words[3] = cus_random();
        words[2] = cus_random();
    }
}

typedef struct Gpt128Bitstream {
    unsigned __int128 pool;
    unsigned int bits;
} Gpt128Bitstream;
typedef void (*Gpt128PinvMulFn)(const M128 *, const M128 *, M128 *);
typedef void (*Gpt128M64MulFn)(const M64 *, const M64 *, M64 *);
typedef struct Gpt128M16RegPair {
    __m128i top;
    __m128i bot;
} Gpt128M16RegPair;
typedef struct {
    uint64_t pool;
    unsigned int bits;
} gpt_m16_bitstream;
typedef struct Gpt256ByteStream {
    uint32_t word;
    unsigned int avail;
} Gpt256ByteStream;
typedef void (*Gpt256PinvMulFn)(const M256 *, const M256 *, M256 *);
typedef void (*Gpt256M128MulFn)(const M128 *, const M128 *, M128 *);
typedef void (*Gpt256M128SubgenFn)(M128 *, M128 *);
typedef struct Gpt256Bitstream {
    uint64_t bits;
    unsigned int avail;
} Gpt256Bitstream;
typedef struct Gpt256M16RegPair {
    __m128i top;
    __m128i bot;
} Gpt256M16RegPair;
typedef struct M16RegPair_GPT {
    __m128i top;
    __m128i bot;
} M16RegPair_GPT;
typedef struct Gpt32BitstreamP {
    uint64_t bits;
    int available;
} Gpt32BitstreamP;
typedef struct Gpt64Bitstream {
    uint64_t pool;
    unsigned int bits;
} Gpt64Bitstream;
typedef struct Gpt64M16RegPair {
    __m128i top;
    __m128i bot;
} Gpt64M16RegPair;

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

extern unsigned int m_index;
extern unsigned int m_intermediateOffset;

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
static void MatMulMatM128_GPT12_8x8tiles_rowdirect_aligned_total(
    const M128 *Mat1, const M128 *Mat2, M128 *Mat);
static inline void xorM64_m128gpt(const M64 *a, const M64 *b, M64 *out);
static void pack_M64_to_M128_m128gpt(
    const M64 *top_left, const M64 *top_right,
    const M64 *bot_left, const M64 *bot_right,
    M128 *out);
static inline uint64_t gpt128_take_bits(Gpt128Bitstream *stream, unsigned int n);
static inline uint32_t gpt128_bounded_rand_bitstream(Gpt128Bitstream *stream,
                                                     uint32_t bound);
static void gpt128_make_perm_bitstream(uint8_t p[128]);
static inline void gpt128_apply_perm_rows_and_pinv_nozero(
    const M128 *base,
    const uint8_t p[128],
    M128 *mat,
    M128 *pinv);
static void gpt128_apply_dynamic_p_fastp_with(
    const M128 *base,
    const M128 *base_inv,
    void (*make_perm)(uint8_t p[128]),
    Gpt128PinvMulFn pinv_mul,
    M128 *mat,
    M128 *mat_inv);
static inline void mulM64_total_gpt17_m128gpt(const M64 *a, const M64 *b, M64 *out);
static inline uint64_t gpt128_m64_col_mask(int col);
static void gpt128_m64_make_lower_unit_pair_direct(M64 *L, M64 *Linv);
static void gpt128_m64_make_upper_unit_pair_direct(M64 *U, M64 *Uinv);
static inline uint64_t gpt128_random64(void);
static inline uint64_t gpt128_m8_mul_u64(uint64_t a, uint64_t b);
static inline void gpt128_m8_make_lower_unit_pair_u64(uint64_t *L, uint64_t *Linv);
static inline void gpt128_m8_make_upper_unit_pair_u64(uint64_t *U, uint64_t *Uinv);
static inline Gpt128M16RegPair gpt128_pack_m8_blocks_to_m16(
    uint64_t lt, uint64_t rt, uint64_t lb, uint64_t rb);
static inline Gpt128M16RegPair gpt128_zero_m16_reg_pair(void);
static inline Gpt128M16RegPair gpt128_rand_m16_reg_pair(void);
static inline Gpt128M16RegPair gpt128_mul_m16_reg_pair(
    Gpt128M16RegPair a, Gpt128M16RegPair b);
static inline void gpt128_store_m32_row8(uint32_t *dst, Gpt128M16RegPair left,
                                         Gpt128M16RegPair right, int half);
static inline void gpt128_pack_m16_blocks_to_m32(
    Gpt128M16RegPair lt, Gpt128M16RegPair rt,
    Gpt128M16RegPair lb, Gpt128M16RegPair rb,
    M32 *out);
static void gpt128_m32_make_lower_unit_pair_m16reg(M32 *L, M32 *Linv);
static void gpt128_m32_make_upper_unit_pair_m16reg(M32 *U, M32 *Uinv);
static void gpt128_m64_make_lower_unit_pair_m32units(M64 *L, M64 *Linv);
static void gpt128_m64_make_upper_unit_pair_m32units(M64 *U, M64 *Uinv);
static inline void zeroM64_m128gpt(M64 *out);
static inline void muladdM64_mode_m128gpt(
    const M64 *a, const M64 *b, const M64 *add,
    Gpt128M64MulFn mul, M64 *out);
static void muladd_two_left_two_right_m64_m128gpt(
    const M64 *left0, const M64 *left1,
    const M64 *right0, const M64 *right1,
    const M64 *add00, const M64 *add01,
    const M64 *add10, const M64 *add11,
    int streamA_core,
    M64 *out00, M64 *out01,
    M64 *out10, M64 *out11);
static void genMatpairM128_radix2_unitriangular_plu_recursive_gpt(
    int two_left_two_right,
    int streamA_core,
    int m32unit_leaf,
    Gpt128M64MulFn mul,
    void (*make_perm)(uint8_t p[128]),
    Gpt128PinvMulFn pinv_mul,
    M128 *mat,
    M128 *mat_inv);
void wbmatrix2_genmatpair_m128_hbup(
    M128 *Mat, M128 *Mat_inv);
static void MatMulMatM16_reg(__m128i A0, __m128i A1, __m128i B0, __m128i B1, __m128i *C0, __m128i *C1);
static inline uint64_t gpt1_random64(void);
static inline void gpt1_store_left_perm(__m128i base_top, __m128i base_bot,
                                        const uint8_t p[16], M16 *out);
static inline __m128i gpt_m8_mul(__m128i A, __m128i B);
static inline void gpt_m8_pair_muladd(
    __m128i A, __m128i B0, __m128i B1,
    __m128i Add0, __m128i Add1,
    __m128i *Out0, __m128i *Out1);
static inline __m128i gpt_m8_rand_matrix(void);
static inline void gpt_pack_m16_blocks(
    __m128i LT, __m128i RT, __m128i LB, __m128i RB,
    __m128i *top, __m128i *bot);
static inline void gpt_m16_make_perm16_fast_bitsliced(uint8_t p[16]);
static inline void gpt_m16_transpose_reg(__m128i *top, __m128i *bot);
static inline void gpt_m16_apply_dynamic_p_transpose_shuffle(
    __m128i base_top, __m128i base_bot,
    __m128i base_inv_top, __m128i base_inv_bot,
    M16 *Mat, M16 *Mat_inv);
static inline void gpt_m8_make_lower_unit_pair(__m128i *L, __m128i *Linv);
static inline void gpt_m8_make_upper_unit_pair(__m128i *U, __m128i *Uinv);
void wbmatrix2_genmatpair_m16_hbup(M16 *Mat, M16 *Mat_inv);
static inline void extract_256_blocks_simd_opt2(const M256 *in, int r_start, 
                                           __m512i *out0, __m512i *out1, 
                                           __m512i *out2, __m512i *out3, 
                                           __m512i vpermb_mask);
static inline void pack_256_blocks_simd_opt2(__m512i in0, __m512i in1, __m512i in2, __m512i in3,
                                        M256 *out, int r_start, __m512i pack_mask);
static inline void matmul_m256_opt3_core(const M256 *Mat1, const M256 *Mat2, M256 *Mat);
static void MatMulMatM256_Opt3_ptr(const M256 *Mat1, const M256 *Mat2, M256 *Mat);
static void xorM128_m256gpt(const M128 *a, const M128 *b, M128 *out);
static void pack_M128_to_M256_m256gpt(
    const M128 *top_left, const M128 *top_right,
    const M128 *bot_left, const M128 *bot_right,
    M256 *out);
static inline uint8_t gpt256_take_random_byte(Gpt256ByteStream *stream);
static void gpt256_make_perm_bytestream(uint8_t p[256]);
static void gpt256_apply_dynamic_p_fastp_with(
    const M256 *base, const M256 *base_inv, Gpt256PinvMulFn pinv_mul,
    M256 *mat, M256 *mat_inv);
static inline void mulM128_gpt12_m256gpt(const M128 *a, const M128 *b, M128 *out);
static uint64_t gpt256_take_bits(Gpt256Bitstream *stream, unsigned int n);
static inline void gpt256_m128_zero(M128 *out);
static inline void gpt256_m64_zero(M64 *out);
static inline void gpt256_m128_col_mask(int col, uint64_t row[2]);
static inline uint64_t gpt256_m128_test_col(const uint64_t row[2], int col);
static void gpt256_m128_make_lower_unit_pair_direct(M128 *L, M128 *Linv);
static void gpt256_m128_make_upper_unit_pair_direct(M128 *U, M128 *Uinv);
static inline uint64_t gpt256_m64_col_mask(int col);
static void gpt256_m64_make_lower_unit_pair_direct(M64 *L, M64 *Linv);
static void gpt256_m64_make_upper_unit_pair_direct(M64 *U, M64 *Uinv);
static inline uint64_t gpt256_random64(void);
static inline uint64_t gpt256_m8_mul_u64(uint64_t a, uint64_t b);
static inline void gpt256_m8_make_lower_unit_pair_u64(uint64_t *L, uint64_t *Linv);
static inline void gpt256_m8_make_upper_unit_pair_u64(uint64_t *U, uint64_t *Uinv);
static inline Gpt256M16RegPair gpt256_pack_m8_blocks_to_m16(
    uint64_t lt, uint64_t rt, uint64_t lb, uint64_t rb);
static inline Gpt256M16RegPair gpt256_zero_m16_reg_pair(void);
static inline Gpt256M16RegPair gpt256_rand_m16_reg_pair(void);
static inline Gpt256M16RegPair gpt256_mul_m16_reg_pair(
    Gpt256M16RegPair a, Gpt256M16RegPair b);
static inline void gpt256_store_m32_row8(uint32_t *dst, Gpt256M16RegPair left,
                                         Gpt256M16RegPair right, int half);
static inline void gpt256_pack_m16_blocks_to_m32(
    Gpt256M16RegPair lt, Gpt256M16RegPair rt,
    Gpt256M16RegPair lb, Gpt256M16RegPair rb,
    M32 *out);
static inline void gpt256_zeroM32(M32 *out);
static inline void pack_M32_to_M64_m256gpt(
    const M32 *top_left, const M32 *top_right,
    const M32 *bot_left, const M32 *bot_right,
    M64 *out);
static void gpt256_m32_make_lower_unit_pair_m16reg(M32 *L, M32 *Linv);
static void gpt256_m32_make_upper_unit_pair_m16reg(M32 *U, M32 *Uinv);
static void gpt256_m64_make_lower_unit_pair_m32units(M64 *L, M64 *Linv);
static void gpt256_m64_make_upper_unit_pair_m32units(M64 *U, M64 *Uinv);
static inline void pack_M64_to_M128_m256gpt(
    const M64 *top_left, const M64 *top_right,
    const M64 *bot_left, const M64 *bot_right,
    M128 *out);
static void gpt256_m128_make_lower_unit_pair_m64units(M128 *L, M128 *Linv);
static void gpt256_m128_make_upper_unit_pair_m64units(M128 *U, M128 *Uinv);
static void gpt256_m128_make_lower_unit_pair_m32units(M128 *L, M128 *Linv);
static void gpt256_m128_make_upper_unit_pair_m32units(M128 *U, M128 *Uinv);
static inline void muladdM128_mode_m256gpt(
    const M128 *a, const M128 *b, const M128 *add,
    Gpt256M128MulFn mul, M128 *out);
static void muladd_two_left_two_right_m128_m256gpt(
    const M128 *left0, const M128 *left1,
    const M128 *right0, const M128 *right1,
    const M128 *add00, const M128 *add01,
    const M128 *add10, const M128 *add11,
    int rowdirect_core,
    M128 *out00, M128 *out01,
    M128 *out10, M128 *out11);
static void genMatpairM256_radix2_unitriangular_plu_recursive_gpt(
    int unit_leaf_mode,
    int two_left_two_right,
    int rowdirect_core,
    Gpt256M128MulFn mul,
    Gpt256PinvMulFn pinv_mul,
    M256 *mat,
    M256 *mat_inv);
void wbmatrix2_genmatpair_m256_hbup(
    M256 *Mat, M256 *Mat_inv);
static inline void MatMulMatM32_GPT1_core(
    __m256i A0, __m256i A1, __m256i A2, __m256i A3,
    __m256i B0, __m256i B1, __m256i B2, __m256i B3,
    __m256i *C0, __m256i *C1, __m256i *C2, __m256i *C3);
static inline void MatMulMatM32_GPT2_prepareB_core(
    __m256i B0, __m256i B1, __m256i B2, __m256i B3,
    M32PreparedB_GPT *PreparedB);
static inline void MatMulMatM32_GPT2_preparedB_core(
    __m256i A0, __m256i A1, __m256i A2, __m256i A3,
    const M32PreparedB_GPT *PreparedB,
    __m256i *C0, __m256i *C1, __m256i *C2, __m256i *C3);
static void MatMulMatM32_GPT2_prepareB(const M32 *Mat2, M32PreparedB_GPT *PreparedB);
static void MatMulMatM32_GPT2_preparedB(const M32 *Mat1, const M32PreparedB_GPT *PreparedB, M32 *Mat);
static inline void MatMulMatM32_GPT4_rowdirect_core(
    __m256i A0, __m256i A1, __m256i A2, __m256i A3,
    __m256i B0, __m256i B1, __m256i B2, __m256i B3,
    __m256i *C0, __m256i *C1, __m256i *C2, __m256i *C3);
static void MatMulMatM32_GPT4_rowdirect_ptr(const M32 *Mat1, const M32 *Mat2, M32 *Mat);
static inline void MatMulMatM32_GPT4_preparedB_rowdirect_core(
    __m256i A0, __m256i A1, __m256i A2, __m256i A3,
    const M32PreparedB_GPT *PreparedB,
    __m256i *C0, __m256i *C1, __m256i *C2, __m256i *C3);
static void MatMulMatM32_GPT4_preparedB_rowdirect(const M32 *Mat1, const M32PreparedB_GPT *PreparedB, M32 *Mat);
static void MatMulMatM32_GPT1_ptr(const M32 *Mat1, const M32 *Mat2, M32 *Mat);
static inline uint64_t gpt32_random64(void);
static inline M16RegPair_GPT gpt32_rand_m16_reg(void);
static inline M16RegPair_GPT gpt32_zero_m16_reg(void);
static inline M16RegPair_GPT gpt32_mul_m16(M16RegPair_GPT a, M16RegPair_GPT b);
static inline void gpt32_prepare_m16_right(M16RegPair_GPT right,
                                           M16PreparedB_GPT *prepared);
static inline void gpt32_muladd_two_left_two_right_m16(
    M16RegPair_GPT left0, M16RegPair_GPT left1,
    M16RegPair_GPT right0, M16RegPair_GPT right1,
    M16RegPair_GPT add00, M16RegPair_GPT add01,
    M16RegPair_GPT add10, M16RegPair_GPT add11,
    M16RegPair_GPT *out00, M16RegPair_GPT *out01,
    M16RegPair_GPT *out10, M16RegPair_GPT *out11);
static inline void gpt32_store_m32_row8(uint32_t *dst, M16RegPair_GPT left,
                                        M16RegPair_GPT right, int half);
static inline void gpt32_pack_m16_blocks(M16RegPair_GPT lt, M16RegPair_GPT rt,
                                         M16RegPair_GPT lb, M16RegPair_GPT rb,
                                         M32 *out);
static inline uint32_t gpt32_bounded_rand(uint32_t bound);
static inline uint32_t gpt32_bitstream_take(Gpt32BitstreamP *stream, int count);
static inline uint32_t gpt32_bounded_rand_bitstream(Gpt32BitstreamP *stream,
                                                    uint32_t bound);
static void gpt32_apply_dynamic_p_matmul(const M32 *base, const M32 *base_inv,
                                         M32 *mat, M32 *mat_inv);
static void gpt32_apply_dynamic_p_matmul_rowdirect(const M32 *base, const M32 *base_inv,
                                                   M32 *mat, M32 *mat_inv);
static void gpt32_apply_dynamic_p_matmul_bitstream(const M32 *base, const M32 *base_inv,
                                                   int rowdirect_pinv,
                                                   M32 *mat, M32 *mat_inv);
static inline uint64_t gpt32_m8_mul_u64(uint64_t a, uint64_t b);
static inline void gpt32_m8_make_lower_unit_pair_u64(uint64_t *L, uint64_t *Linv);
static inline void gpt32_m8_make_upper_unit_pair_u64(uint64_t *U, uint64_t *Uinv);
static inline M16RegPair_GPT gpt32_pack_m8_blocks_to_m16(uint64_t lt, uint64_t rt,
                                                         uint64_t lb, uint64_t rb);
static inline void gpt32_make_m16_lower_unit_pair(M16RegPair_GPT *L,
                                                  M16RegPair_GPT *Linv);
static inline void gpt32_make_m16_upper_unit_pair(M16RegPair_GPT *U,
                                                  M16RegPair_GPT *Uinv);
static void gpt32_radix2_unitriangular_plu_recursive_twoleft_tworight_core(
    int rowdirect_pinv, int bitstream_p, M32 *mat, M32 *mat_inv);
void wbmatrix2_genmatpair_m32_hbup(
    M32 *Mat, M32 *Mat_inv);
static inline uint32_t m4_gpt1_gen_pack_rows(uint8_t r0, uint8_t r1, uint8_t r2, uint8_t r3);
static inline uint8_t m4_gpt1_gen_bit(uint32_t r, unsigned int bit);
static inline uint32_t m4_gpt7_gen_swap_bytes_down(uint32_t rows, uint32_t high, uint32_t low);
static inline uint32_t m4_gpt7_gen_swap_cols_down(uint32_t rows, uint32_t high, uint32_t low);
static inline void m4_gpt7_gen_store_rows_nomask(uint32_t rows, M4 *Mat);
static inline uint32_t m4_gpt10_gen_fast_j2_nocmp(uint32_t j2);
static inline void m4_gpt12_gen_dynamic_p_apply_interleaved_swaps_nocmp_from_bits(uint32_t r,
                                                                                  uint32_t base_rows,
                                                                                  uint32_t base_inv_rows,
                                                                                  uint32_t *mat_rows,
                                                                                  uint32_t *mat_inv_rows);
static inline uint32_t m4_gpt11_permuteQPR_inline(uint32_t x);
static inline uint32_t m4_gpt11_cus_random_inline(void);
static inline uint8_t m4_gpt4_gen_select_row(uint8_t row, uint8_t bit);
static inline void m4_gpt4_gen_radix2_base_full_direct_from_bits(uint32_t r, uint32_t *base_rows, uint32_t *base_inv_rows);
void wbmatrix2_genmatpair_m4_hbup(M4 *Mat, M4 *Mat_inv);
static inline __attribute__((always_inline)) __m512i pair_ymms_to_zmm_gpt12(
    __m256i lo, __m256i hi);
static inline __attribute__((always_inline)) void load_m64_as_m32_ymms_direct_gpt12(
    const M64 *A,
    __m256i A00[4], __m256i A01[4], __m256i A10[4], __m256i A11[4]);
static inline __attribute__((always_inline)) void store_m32_pair_ymms_as_m64_direct_gpt12(
    const __m512i Ctop[4], const __m512i Cbottom[4], M64 *C);
static inline __attribute__((always_inline)) void prepare_m32pairbt_from_m64_half_gpt12(
    const uint64_t *rows, __m512i PairBT[4]);
static inline __attribute__((always_inline)) void prepare_m32pairbt_from_m64_dualhalf_gpt18(
    const M64 *Mat2, M64PreparedB_M32PairBT_GPT *PreparedB);
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
static inline __attribute__((always_inline)) void load_m64_half_as_m32_ymms_direct_gpt15(
    const uint64_t *rows, __m256i Aleft[4], __m256i Aright[4]);
static inline __attribute__((always_inline)) void store_m32_pair_ymms_as_m64_half_gpt15(
    const __m512i Cpair[4], uint64_t *rows);
static void MatMulMatM64_GPT15_preparedB_m32pairbt_rawaccum_streamA(
    const M64 *Mat1, const M64PreparedB_M32PairBT_GPT *PreparedB, M64 *Mat);
static void MatMulMatM64_GPT17_m32pairbt_rawaccum_aligned_total(
    const M64 *Mat1, const M64 *Mat2, M64 *Mat);
static void MatMulMatM64_GPT19_m32pairbt_rawaccum_streamA_dualprepare_total(
    const M64 *Mat1, const M64 *Mat2, M64 *Mat);
static inline void xorM32_gpt(const M32 *a, const M32 *b, M32 *out);
static inline void pack_M32_to_M64_gpt(
    const M32 *top_left, const M32 *top_right,
    const M32 *bot_left, const M32 *bot_right,
    M64 *out);
static inline uint32_t gpt64_bounded_rand(uint32_t bound);
static void gpt64_make_perm(uint8_t p[64]);
static void gpt64_apply_perm_rows_only(const M64 *base, const uint8_t p[64], M64 *mat);
static void gpt64_build_pinv_dense_from_perm(const uint8_t p[64], M64 *pinv);
static void gpt64_apply_dynamic_p_matmul_with(
    const M64 *base,
    const M64 *base_inv,
    void (*pinv_mul)(const M64 *, const M64 *, M64 *),
    void (*make_perm)(uint8_t p[64]),
    M64 *mat,
    M64 *mat_inv);
static inline uint32_t gpt64_take_bits(Gpt64Bitstream *stream, unsigned int n);
static inline uint32_t gpt64_bounded_rand_bitstream(Gpt64Bitstream *stream,
                                                    uint32_t bound);
static void gpt64_make_perm_bitstream(uint8_t p[64]);
static inline void gpt64_apply_perm_rows_and_pinv_nozero(
    const M64 *base,
    const uint8_t p[64],
    M64 *mat,
    M64 *pinv);
static inline void gpt64_apply_dynamic_p_fastp_gpt14(
    const M64 *base,
    const M64 *base_inv,
    void (*make_perm)(uint8_t p[64]),
    M64 *mat,
    M64 *mat_inv);
static inline void gpt64_apply_dynamic_p_fastp_gpt17(
    const M64 *base,
    const M64 *base_inv,
    void (*make_perm)(uint8_t p[64]),
    M64 *mat,
    M64 *mat_inv);
static inline void gpt64_apply_dynamic_p_fastp_gpt19(
    const M64 *base,
    const M64 *base_inv,
    void (*make_perm)(uint8_t p[64]),
    M64 *mat,
    M64 *mat_inv);
static inline uint64_t gpt64_random64(void);
static inline uint64_t gpt64_m8_mul_u64(uint64_t a, uint64_t b);
static inline void gpt64_m8_make_lower_unit_pair_u64(uint64_t *L, uint64_t *Linv);
static inline void gpt64_m8_make_upper_unit_pair_u64(uint64_t *U, uint64_t *Uinv);
static inline Gpt64M16RegPair gpt64_pack_m8_blocks_to_m16(uint64_t lt, uint64_t rt,
                                                          uint64_t lb, uint64_t rb);
static inline Gpt64M16RegPair gpt64_zero_m16_reg_pair(void);
static inline Gpt64M16RegPair gpt64_rand_m16_reg_pair(void);
static inline Gpt64M16RegPair gpt64_mul_m16_reg_pair(Gpt64M16RegPair a,
                                                     Gpt64M16RegPair b);
static inline void gpt64_store_m32_row8(uint32_t *dst, Gpt64M16RegPair left,
                                        Gpt64M16RegPair right, int half);
static inline void gpt64_pack_m16_blocks_to_m32(Gpt64M16RegPair lt,
                                                Gpt64M16RegPair rt,
                                                Gpt64M16RegPair lb,
                                                Gpt64M16RegPair rb,
                                                M32 *out);
static void gpt64_m32_make_lower_unit_pair_m16reg(M32 *L, M32 *Linv);
static void gpt64_m32_make_upper_unit_pair_m16reg(M32 *U, M32 *Uinv);
static inline void gpt64_mulM32_mode(const M32 *a, const M32 *b, int rowdirect,
                                     M32 *out);
static inline void gpt64_muladdM32_mode(const M32 *a, const M32 *b, const M32 *add,
                                        int rowdirect, M32 *out);
static inline void zeroM32_gpt(M32 *out);
static inline void gpt64_prepared_muladd_m32(const M32 *left,
                                             const M32PreparedB_GPT *prepared_right,
                                             const M32 *add,
                                             int rowdirect_prepared,
                                             M32 *out);
static void gpt64_muladd_two_left_two_right_m32(
    const M32 *left0, const M32 *left1,
    const M32 *right0, const M32 *right1,
    const M32 *add00, const M32 *add01,
    const M32 *add10, const M32 *add11,
    int rowdirect_prepared,
    M32 *out00, M32 *out01,
    M32 *out10, M32 *out11);
static void genMatpairM64_radix2_unitriangular_plu_recursive_m16units_gpt(
    int two_left_two_right,
    int rowdirect_prepared,
    int rowdirect_m32,
    int bitstream_p,
    int fastp_pinv,
    void (*pinv_mul)(const M64 *, const M64 *, M64 *),
    M64 *mat,
    M64 *mat_inv);
void wbmatrix2_genmatpair_m64_hbup(
    M64 *Mat, M64 *Mat_inv);
static inline uint64_t gpt8_random64(void);
static inline __m128i gpt8_make_dynamic_p_mask_fast_bitsliced_low8(void);
static inline void gpt8_gen_plu_base_no_p_invcols_chain6_sharedI(
    __m128i *out_base,
    __m128i *out_base_inv_cols,
    __m128i I);
void wbmatrix2_genmatpair_m8_hbup(M8 *Mat, M8 *Mat_inv);

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

static void MatMulMatM128_GPT12_8x8tiles_rowdirect_aligned_total(
    const M128 *Mat1, const M128 *Mat2, M128 *Mat)
{
    M128PreparedB_8x8Tiles_GPT PreparedB __attribute__((aligned(64)));
    MatMulMatM128_GPT5_prepareB_8x8tiles(Mat2, &PreparedB);
    MatMulMatM128_GPT10_preparedB_8x8tiles_rowdirect(Mat1, &PreparedB, Mat);
}

static inline void xorM64_m128gpt(const M64 *a, const M64 *b, M64 *out) {
    for (int i = 0; i < 8; i++) {
        __m512i va = _mm512_loadu_si512((const __m512i *)&a->M[i * 8]);
        __m512i vb = _mm512_loadu_si512((const __m512i *)&b->M[i * 8]);
        _mm512_storeu_si512((__m512i *)&out->M[i * 8], _mm512_xor_si512(va, vb));
    }
}

static void pack_M64_to_M128_m128gpt(
    const M64 *top_left, const M64 *top_right,
    const M64 *bot_left, const M64 *bot_right,
    M128 *out)
{
    for (int i = 0; i < 64; i++) {
        out->M[i][0] = top_left->M[i];
        out->M[i][1] = top_right->M[i];
        out->M[i + 64][0] = bot_left->M[i];
        out->M[i + 64][1] = bot_right->M[i];
    }
}

static inline uint64_t gpt128_take_bits(Gpt128Bitstream *stream, unsigned int n) {
    while (stream->bits < n) {
        stream->pool |= ((unsigned __int128)cus_random()) << stream->bits;
        stream->bits += 32U;
    }

    const uint64_t mask = (n == 64U) ? UINT64_MAX : ((1ULL << n) - 1ULL);
    const uint64_t out = (uint64_t)(stream->pool & mask);
    stream->pool >>= n;
    stream->bits -= n;
    return out;
}

static inline uint32_t gpt128_bounded_rand_bitstream(Gpt128Bitstream *stream,
                                                     uint32_t bound) {
    uint32_t x = bound - 1u;
    unsigned int bits = 0;
    do {
        bits++;
        x >>= 1;
    } while (x != 0);

    const uint32_t raw = (uint32_t)gpt128_take_bits(stream, bits);
    return (uint32_t)(((uint64_t)raw * (uint64_t)bound) >> bits);
}

static void gpt128_make_perm_bitstream(uint8_t p[128]) {
    for (int i = 0; i < 128; i++) {
        p[i] = (uint8_t)i;
    }

    Gpt128Bitstream stream = {0, 0};
    for (int i = 127; i > 0; i--) {
        int j = (int)gpt128_bounded_rand_bitstream(&stream, (uint32_t)(i + 1));
        uint8_t tmp = p[i];
        p[i] = p[j];
        p[j] = tmp;
    }
}

static inline void gpt128_apply_perm_rows_and_pinv_nozero(
    const M128 *base,
    const uint8_t p[128],
    M128 *mat,
    M128 *pinv)
{
    for (int i = 0; i < 128; i++) {
        const uint8_t pi = p[i];
        mat->M[i][0] = base->M[pi][0];
        mat->M[i][1] = base->M[pi][1];
        pinv->M[pi][0] = (i < 64) ? (1ULL << (63 - i)) : 0;
        pinv->M[pi][1] = (i >= 64) ? (1ULL << (127 - i)) : 0;
    }
}

static void gpt128_apply_dynamic_p_fastp_with(
    const M128 *base,
    const M128 *base_inv,
    void (*make_perm)(uint8_t p[128]),
    Gpt128PinvMulFn pinv_mul,
    M128 *mat,
    M128 *mat_inv)
{
    uint8_t p[128];
    M128 pinv;

    make_perm(p);
    gpt128_apply_perm_rows_and_pinv_nozero(base, p, mat, &pinv);
    pinv_mul(base_inv, &pinv, mat_inv);
}

static inline void mulM64_total_gpt17_m128gpt(const M64 *a, const M64 *b, M64 *out) {
    MatMulMatM64_GPT17_m32pairbt_rawaccum_aligned_total(a, b, out);
}

static inline uint64_t gpt128_m64_col_mask(int col) {
    return 1ULL << (63 - col);
}

static void gpt128_m64_make_lower_unit_pair_direct(M64 *L, M64 *Linv) {
    Gpt128Bitstream stream = {0, 0};

    for (int i = 0; i < 64; i++) {
        uint64_t row = gpt128_m64_col_mask(i);
        const uint64_t bits = i == 0 ? 0 : gpt128_take_bits(&stream, (unsigned int)i);
        for (int j = 0; j < i; j++) {
            const uint64_t take = 0ULL - ((bits >> j) & 1ULL);
            row ^= take & gpt128_m64_col_mask(j);
        }
        L->M[i] = row;
    }

    for (int i = 0; i < 64; i++) {
        uint64_t acc = L->M[i];
        uint64_t inv = gpt128_m64_col_mask(i);
        for (int k = i - 1; k >= 0; k--) {
            if (acc & gpt128_m64_col_mask(k)) {
                inv ^= gpt128_m64_col_mask(k);
                acc ^= L->M[k];
            }
        }
        Linv->M[i] = inv;
    }
}

static void gpt128_m64_make_upper_unit_pair_direct(M64 *U, M64 *Uinv) {
    Gpt128Bitstream stream = {0, 0};

    for (int i = 0; i < 64; i++) {
        uint64_t row = gpt128_m64_col_mask(i);
        const int width = 63 - i;
        const uint64_t bits = width == 0 ? 0 : gpt128_take_bits(&stream, (unsigned int)width);
        for (int j = i + 1; j < 64; j++) {
            const uint64_t take = 0ULL - ((bits >> (j - i - 1)) & 1ULL);
            row ^= take & gpt128_m64_col_mask(j);
        }
        U->M[i] = row;
    }

    for (int i = 63; i >= 0; i--) {
        uint64_t acc = U->M[i];
        uint64_t inv = gpt128_m64_col_mask(i);
        for (int k = i + 1; k < 64; k++) {
            if (acc & gpt128_m64_col_mask(k)) {
                inv ^= gpt128_m64_col_mask(k);
                acc ^= U->M[k];
            }
        }
        Uinv->M[i] = inv;
    }
}

static inline uint64_t gpt128_random64(void) {
    uint64_t high = (uint64_t)cus_random();
    uint64_t low = (uint64_t)cus_random();
    return (high << 32) | low;
}

static inline uint64_t gpt128_m8_mul_u64(uint64_t a, uint64_t b) {
    const uint64_t I_val = 0x0102040810204080ULL;
    __m128i I = _mm_set1_epi64x((long long)I_val);
    __m128i b_x = _mm_cvtsi64_si128((long long)b);
    __m128i a_x = _mm_cvtsi64_si128((long long)a);
    __m128i b_t = _mm_gf2p8affine_epi64_epi8(I, b_x, 0);
    __m128i c_cols = _mm_gf2p8affine_epi64_epi8(b_t, a_x, 0);
    __m128i c_rows = _mm_gf2p8affine_epi64_epi8(I, c_cols, 0);
    return (uint64_t)_mm_cvtsi128_si64(c_rows);
}

static inline void gpt128_m8_make_lower_unit_pair_u64(uint64_t *L, uint64_t *Linv) {
    const uint64_t r = gpt128_random64();
    const uint64_t I_val = 0x0102040810204080ULL;

    const uint64_t m1_val = I_val
        ^ ((r & 0xFULL) << 36)
        ^ (((r >> 4) & 0xFULL) << 44)
        ^ (((r >> 8) & 0xFULL) << 52)
        ^ (((r >> 12) & 0xFULL) << 60);
    const uint64_t m2_val = I_val
        ^ (((r >> 16) & 0x3ULL) << 22)
        ^ (((r >> 18) & 0x3ULL) << 30)
        ^ (((r >> 20) & 0x3ULL) << 50)
        ^ (((r >> 22) & 0x3ULL) << 58);
    const uint64_t m3_val = I_val
        ^ (((r >> 24) & 1ULL) << 15)
        ^ (((r >> 25) & 1ULL) << 29)
        ^ (((r >> 26) & 1ULL) << 43)
        ^ (((r >> 27) & 1ULL) << 57);

    *L = gpt128_m8_mul_u64(gpt128_m8_mul_u64(m1_val, m2_val), m3_val);
    *Linv = gpt128_m8_mul_u64(gpt128_m8_mul_u64(m3_val, m2_val), m1_val);
}

static inline void gpt128_m8_make_upper_unit_pair_u64(uint64_t *U, uint64_t *Uinv) {
    const uint64_t r = gpt128_random64();
    const uint64_t I_val = 0x0102040810204080ULL;

    const uint64_t n1_val = I_val
        ^ (r & 0xFULL)
        ^ (((r >> 4) & 0xFULL) << 8)
        ^ (((r >> 8) & 0xFULL) << 16)
        ^ (((r >> 12) & 0xFULL) << 24);
    const uint64_t n2_val = I_val
        ^ (((r >> 16) & 0x3ULL) << 4)
        ^ (((r >> 18) & 0x3ULL) << 12)
        ^ (((r >> 20) & 0x3ULL) << 32)
        ^ (((r >> 22) & 0x3ULL) << 40);
    const uint64_t n3_val = I_val
        ^ (((r >> 24) & 1ULL) << 6)
        ^ (((r >> 25) & 1ULL) << 20)
        ^ (((r >> 26) & 1ULL) << 34)
        ^ (((r >> 27) & 1ULL) << 48);

    *U = gpt128_m8_mul_u64(gpt128_m8_mul_u64(n1_val, n2_val), n3_val);
    *Uinv = gpt128_m8_mul_u64(gpt128_m8_mul_u64(n3_val, n2_val), n1_val);
}

static inline Gpt128M16RegPair gpt128_pack_m8_blocks_to_m16(
    uint64_t lt, uint64_t rt, uint64_t lb, uint64_t rb)
{
    Gpt128M16RegPair out;
    __m128i lt_x = _mm_cvtsi64_si128((long long)lt);
    __m128i rt_x = _mm_cvtsi64_si128((long long)rt);
    __m128i lb_x = _mm_cvtsi64_si128((long long)lb);
    __m128i rb_x = _mm_cvtsi64_si128((long long)rb);
    out.top = _mm_unpacklo_epi8(rt_x, lt_x);
    out.bot = _mm_unpacklo_epi8(rb_x, lb_x);
    return out;
}

static inline Gpt128M16RegPair gpt128_zero_m16_reg_pair(void) {
    Gpt128M16RegPair out;
    out.top = _mm_setzero_si128();
    out.bot = _mm_setzero_si128();
    return out;
}

static inline Gpt128M16RegPair gpt128_rand_m16_reg_pair(void) {
    Gpt128M16RegPair out;
    out.top = _mm_set_epi64x((long long)gpt128_random64(), (long long)gpt128_random64());
    out.bot = _mm_set_epi64x((long long)gpt128_random64(), (long long)gpt128_random64());
    return out;
}

static inline Gpt128M16RegPair gpt128_mul_m16_reg_pair(
    Gpt128M16RegPair a, Gpt128M16RegPair b)
{
    Gpt128M16RegPair out;
    MatMulMatM16_reg(a.top, a.bot, b.top, b.bot, &out.top, &out.bot);
    return out;
}

static inline void gpt128_store_m32_row8(uint32_t *dst, Gpt128M16RegPair left,
                                         Gpt128M16RegPair right, int half) {
    __m128i l = half == 0 ? left.top : left.bot;
    __m128i r = half == 0 ? right.top : right.bot;
    __m128i lo = _mm_unpacklo_epi16(r, l);
    __m128i hi = _mm_unpackhi_epi16(r, l);
    _mm_storeu_si128((__m128i *)dst, lo);
    _mm_storeu_si128((__m128i *)(dst + 4), hi);
}

static inline void gpt128_pack_m16_blocks_to_m32(
    Gpt128M16RegPair lt, Gpt128M16RegPair rt,
    Gpt128M16RegPair lb, Gpt128M16RegPair rb,
    M32 *out)
{
    gpt128_store_m32_row8(out->M, lt, rt, 0);
    gpt128_store_m32_row8(out->M + 8, lt, rt, 1);
    gpt128_store_m32_row8(out->M + 16, lb, rb, 0);
    gpt128_store_m32_row8(out->M + 24, lb, rb, 1);
}

static void gpt128_m32_make_lower_unit_pair_m16reg(M32 *L, M32 *Linv) {
    uint64_t La8, La8_inv, Lb8, Lb8_inv;
    uint64_t Lc8, Lc8_inv, Ld8, Ld8_inv;
    gpt128_m8_make_lower_unit_pair_u64(&La8, &La8_inv);
    gpt128_m8_make_lower_unit_pair_u64(&Lb8, &Lb8_inv);
    gpt128_m8_make_lower_unit_pair_u64(&Lc8, &Lc8_inv);
    gpt128_m8_make_lower_unit_pair_u64(&Ld8, &Ld8_inv);

    uint64_t X0 = gpt128_random64();
    uint64_t X0La = gpt128_m8_mul_u64(X0, La8);
    uint64_t LbInvX0 = gpt128_m8_mul_u64(Lb8_inv, X0);
    Gpt128M16RegPair La = gpt128_pack_m8_blocks_to_m16(La8, 0, X0La, Lb8);
    Gpt128M16RegPair La_inv = gpt128_pack_m8_blocks_to_m16(La8_inv, 0, LbInvX0, Lb8_inv);

    uint64_t X1 = gpt128_random64();
    uint64_t X1Lc = gpt128_m8_mul_u64(X1, Lc8);
    uint64_t LdInvX1 = gpt128_m8_mul_u64(Ld8_inv, X1);
    Gpt128M16RegPair Lb = gpt128_pack_m8_blocks_to_m16(Lc8, 0, X1Lc, Ld8);
    Gpt128M16RegPair Lb_inv = gpt128_pack_m8_blocks_to_m16(Lc8_inv, 0, LdInvX1, Ld8_inv);

    Gpt128M16RegPair zero = gpt128_zero_m16_reg_pair();
    Gpt128M16RegPair X = gpt128_rand_m16_reg_pair();
    Gpt128M16RegPair XLa = gpt128_mul_m16_reg_pair(X, La);
    Gpt128M16RegPair LbInvX = gpt128_mul_m16_reg_pair(Lb_inv, X);

    gpt128_pack_m16_blocks_to_m32(La, zero, XLa, Lb, L);
    gpt128_pack_m16_blocks_to_m32(La_inv, zero, LbInvX, Lb_inv, Linv);
}

static void gpt128_m32_make_upper_unit_pair_m16reg(M32 *U, M32 *Uinv) {
    uint64_t Ua8, Ua8_inv, Ub8, Ub8_inv;
    uint64_t Uc8, Uc8_inv, Ud8, Ud8_inv;
    gpt128_m8_make_upper_unit_pair_u64(&Ua8, &Ua8_inv);
    gpt128_m8_make_upper_unit_pair_u64(&Ub8, &Ub8_inv);
    gpt128_m8_make_upper_unit_pair_u64(&Uc8, &Uc8_inv);
    gpt128_m8_make_upper_unit_pair_u64(&Ud8, &Ud8_inv);

    uint64_t Y0 = gpt128_random64();
    uint64_t Y0Ub = gpt128_m8_mul_u64(Y0, Ub8);
    uint64_t UaInvY0 = gpt128_m8_mul_u64(Ua8_inv, Y0);
    Gpt128M16RegPair Ua = gpt128_pack_m8_blocks_to_m16(Ua8, Y0Ub, 0, Ub8);
    Gpt128M16RegPair Ua_inv = gpt128_pack_m8_blocks_to_m16(Ua8_inv, UaInvY0, 0, Ub8_inv);

    uint64_t Y1 = gpt128_random64();
    uint64_t Y1Ud = gpt128_m8_mul_u64(Y1, Ud8);
    uint64_t UcInvY1 = gpt128_m8_mul_u64(Uc8_inv, Y1);
    Gpt128M16RegPair Ub = gpt128_pack_m8_blocks_to_m16(Uc8, Y1Ud, 0, Ud8);
    Gpt128M16RegPair Ub_inv = gpt128_pack_m8_blocks_to_m16(Uc8_inv, UcInvY1, 0, Ud8_inv);

    Gpt128M16RegPair zero = gpt128_zero_m16_reg_pair();
    Gpt128M16RegPair Y = gpt128_rand_m16_reg_pair();
    Gpt128M16RegPair YUb = gpt128_mul_m16_reg_pair(Y, Ub);
    Gpt128M16RegPair UaInvY = gpt128_mul_m16_reg_pair(Ua_inv, Y);

    gpt128_pack_m16_blocks_to_m32(Ua, YUb, zero, Ub, U);
    gpt128_pack_m16_blocks_to_m32(Ua_inv, UaInvY, zero, Ub_inv, Uinv);
}

static void gpt128_m64_make_lower_unit_pair_m32units(M64 *L, M64 *Linv) {
    M32 La, La_inv, Lb, Lb_inv;
    M32 X, XLa, LbInvX, zero;

    gpt128_m32_make_lower_unit_pair_m16reg(&La, &La_inv);
    gpt128_m32_make_lower_unit_pair_m16reg(&Lb, &Lb_inv);
    wbmatrix2_rand_m32_no_reseed(&X);
    MatMulMatM32_GPT1_ptr(&X, &La, &XLa);
    MatMulMatM32_GPT1_ptr(&Lb_inv, &X, &LbInvX);
    zeroM32_gpt(&zero);

    pack_M32_to_M64_gpt(&La, &zero, &XLa, &Lb, L);
    pack_M32_to_M64_gpt(&La_inv, &zero, &LbInvX, &Lb_inv, Linv);
}

static void gpt128_m64_make_upper_unit_pair_m32units(M64 *U, M64 *Uinv) {
    M32 Ua, Ua_inv, Ub, Ub_inv;
    M32 Y, YUb, UaInvY, zero;

    gpt128_m32_make_upper_unit_pair_m16reg(&Ua, &Ua_inv);
    gpt128_m32_make_upper_unit_pair_m16reg(&Ub, &Ub_inv);
    wbmatrix2_rand_m32_no_reseed(&Y);
    MatMulMatM32_GPT1_ptr(&Y, &Ub, &YUb);
    MatMulMatM32_GPT1_ptr(&Ua_inv, &Y, &UaInvY);
    zeroM32_gpt(&zero);

    pack_M32_to_M64_gpt(&Ua, &YUb, &zero, &Ub, U);
    pack_M32_to_M64_gpt(&Ua_inv, &UaInvY, &zero, &Ub_inv, Uinv);
}

static inline void zeroM64_m128gpt(M64 *out) {
    for (int i = 0; i < 8; i++) {
        _mm512_storeu_si512((__m512i *)&out->M[i * 8], _mm512_setzero_si512());
    }
}

static inline void muladdM64_mode_m128gpt(
    const M64 *a, const M64 *b, const M64 *add,
    Gpt128M64MulFn mul, M64 *out)
{
    M64 prod;
    mul(a, b, &prod);
    xorM64_m128gpt(&prod, add, out);
}

static void muladd_two_left_two_right_m64_m128gpt(
    const M64 *left0, const M64 *left1,
    const M64 *right0, const M64 *right1,
    const M64 *add00, const M64 *add01,
    const M64 *add10, const M64 *add11,
    int streamA_core,
    M64 *out00, M64 *out01,
    M64 *out10, M64 *out11)
{
    M64 prod;
    M64PreparedB_M32PairBT_GPT prepared0, prepared1;

    MatMulMatM64_GPT12_prepareB_m32pairbt(right0, &prepared0);
    MatMulMatM64_GPT12_prepareB_m32pairbt(right1, &prepared1);

#define GPT128_M64_PREPARED_MULADD(LEFT, PREPARED, ADD, OUT) do { \
        if (streamA_core) { \
            MatMulMatM64_GPT15_preparedB_m32pairbt_rawaccum_streamA((LEFT), (PREPARED), &prod); \
        } else { \
            MatMulMatM64_GPT14_preparedB_m32pairbt_rawaccum((LEFT), (PREPARED), &prod); \
        } \
        xorM64_m128gpt(&prod, (ADD), (OUT)); \
    } while (0)

    GPT128_M64_PREPARED_MULADD(left0, &prepared0, add00, out00);
    GPT128_M64_PREPARED_MULADD(left0, &prepared1, add01, out01);
    GPT128_M64_PREPARED_MULADD(left1, &prepared0, add10, out10);
    GPT128_M64_PREPARED_MULADD(left1, &prepared1, add11, out11);

#undef GPT128_M64_PREPARED_MULADD
}

static void genMatpairM128_radix2_unitriangular_plu_recursive_gpt(
    int two_left_two_right,
    int streamA_core,
    int m32unit_leaf,
    Gpt128M64MulFn mul,
    void (*make_perm)(uint8_t p[128]),
    Gpt128PinvMulFn pinv_mul,
    M128 *mat,
    M128 *mat_inv)
{
    M64 La, La_inv, Lb, Lb_inv;
    M64 Ua, Ua_inv, Ub, Ub_inv;
    M64 X, Y;
    M64 XLa, YUb, LbUb;
    M64 base_lt, base_rt, base_lb, base_rb;
    M64 UaInvY, LbInvX, UaInvLaInv;
    M64 inv_lt, inv_rt, inv_lb, inv_rb;
    M64 zero;
    M128 base, base_inv;

    if (m32unit_leaf) {
        gpt128_m64_make_lower_unit_pair_m32units(&La, &La_inv);
        gpt128_m64_make_lower_unit_pair_m32units(&Lb, &Lb_inv);
        gpt128_m64_make_upper_unit_pair_m32units(&Ua, &Ua_inv);
        gpt128_m64_make_upper_unit_pair_m32units(&Ub, &Ub_inv);
    } else {
        gpt128_m64_make_lower_unit_pair_direct(&La, &La_inv);
        gpt128_m64_make_lower_unit_pair_direct(&Lb, &Lb_inv);
        gpt128_m64_make_upper_unit_pair_direct(&Ua, &Ua_inv);
        gpt128_m64_make_upper_unit_pair_direct(&Ub, &Ub_inv);
    }
    wbmatrix2_rand_m64_no_reseed(&X);
    wbmatrix2_rand_m64_no_reseed(&Y);

    mul(&X, &La, &XLa);
    mul(&Y, &Ub, &YUb);
    mul(&Lb, &Ub, &LbUb);

    if (two_left_two_right) {
        zeroM64_m128gpt(&zero);
        muladd_two_left_two_right_m64_m128gpt(
            &La, &XLa, &Ua, &YUb,
            &zero, &zero, &zero, &LbUb,
            streamA_core,
            &base_lt, &base_rt, &base_lb, &base_rb);
    } else {
        mul(&La, &Ua, &base_lt);
        mul(&La, &YUb, &base_rt);
        mul(&XLa, &Ua, &base_lb);
        muladdM64_mode_m128gpt(&XLa, &YUb, &LbUb, mul, &base_rb);
    }

    mul(&Ua_inv, &Y, &UaInvY);
    mul(&Lb_inv, &X, &LbInvX);
    mul(&Ua_inv, &La_inv, &UaInvLaInv);

    if (two_left_two_right) {
        zeroM64_m128gpt(&zero);
        muladd_two_left_two_right_m64_m128gpt(
            &UaInvY, &Ub_inv, &LbInvX, &Lb_inv,
            &UaInvLaInv, &zero, &zero, &zero,
            streamA_core,
            &inv_lt, &inv_rt, &inv_lb, &inv_rb);
    } else {
        muladdM64_mode_m128gpt(&UaInvY, &LbInvX, &UaInvLaInv, mul, &inv_lt);
        mul(&UaInvY, &Lb_inv, &inv_rt);
        mul(&Ub_inv, &LbInvX, &inv_lb);
        mul(&Ub_inv, &Lb_inv, &inv_rb);
    }

    pack_M64_to_M128_m128gpt(&base_lt, &base_rt, &base_lb, &base_rb, &base);
    pack_M64_to_M128_m128gpt(&inv_lt, &inv_rt, &inv_lb, &inv_rb, &base_inv);
    gpt128_apply_dynamic_p_fastp_with(&base, &base_inv, make_perm, pinv_mul,
                                      mat, mat_inv);
}

void wbmatrix2_genmatpair_m128_hbup(
    M128 *Mat, M128 *Mat_inv)
{
    genMatpairM128_radix2_unitriangular_plu_recursive_gpt(
        0, 0, 1, mulM64_total_gpt17_m128gpt,
        gpt128_make_perm_bitstream,
        MatMulMatM128_GPT12_8x8tiles_rowdirect_aligned_total,
        Mat, Mat_inv);
}

static void MatMulMatM16_reg(__m128i A0, __m128i A1, __m128i B0, __m128i B1, __m128i *C0, __m128i *C1) {
    const uint64_t I_val = 0x0102040810204080ULL;
    __m128i I = _mm_set1_epi64x((long long)I_val);

    __m128i mask_B = _mm_setr_epi8(1, 3, 5, 7, 9, 11, 13, 15,  0, 2, 4, 6, 8, 10, 12, 14);
    __m128i B_top = _mm_shuffle_epi8(B0, mask_B);
    __m128i B_bot = _mm_shuffle_epi8(B1, mask_B);

    __m128i B_top_trans = _mm_gf2p8affine_epi64_epi8(I, B_top, 0);
    __m128i B_bot_trans = _mm_gf2p8affine_epi64_epi8(I, B_bot, 0);

    __m128i mask_A_H = _mm_setr_epi8(1, 3, 5, 7, 9, 11, 13, 15, 1, 3, 5, 7, 9, 11, 13, 15);
    __m128i mask_A_L = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 0, 2, 4, 6, 8, 10, 12, 14);

    __m128i A00_dual = _mm_shuffle_epi8(A0, mask_A_H);
    __m128i A01_dual = _mm_shuffle_epi8(A0, mask_A_L);
    __m128i A10_dual = _mm_shuffle_epi8(A1, mask_A_H);
    __m128i A11_dual = _mm_shuffle_epi8(A1, mask_A_L);

    __m128i C_top = _mm_xor_si128(
        _mm_gf2p8affine_epi64_epi8(A00_dual, B_top_trans, 0),
        _mm_gf2p8affine_epi64_epi8(A01_dual, B_bot_trans, 0));

    __m128i C_bot = _mm_xor_si128(
        _mm_gf2p8affine_epi64_epi8(A10_dual, B_top_trans, 0),
        _mm_gf2p8affine_epi64_epi8(A11_dual, B_bot_trans, 0));

    __m128i store_mask = _mm_setr_epi8(
        8, 0, 9, 1, 10, 2, 11, 3, 12, 4, 13, 5, 14, 6, 15, 7
    );

    *C0 = _mm_shuffle_epi8(C_top, store_mask);
    *C1 = _mm_shuffle_epi8(C_bot, store_mask);
}

static inline uint64_t gpt1_random64(void) {
    uint64_t high = (uint64_t)cus_random();
    uint64_t low = (uint64_t)cus_random();
    return (high << 32) | low;
}

static inline void gpt1_store_left_perm(__m128i base_top, __m128i base_bot,
                                        const uint8_t p[16], M16 *out) {
    uint16_t rows[16];
    _mm_storeu_si128((__m128i *)rows, base_top);
    _mm_storeu_si128((__m128i *)(rows + 8), base_bot);

    for (int i = 0; i < 16; i++) {
        out->M[i] = rows[p[i]];
    }
}

static inline __m128i gpt_m8_mul(__m128i A, __m128i B) {
    const uint64_t I_val = 0x0102040810204080ULL;
    __m128i I = _mm_set1_epi64x((long long)I_val);
    __m128i B_t = _mm_gf2p8affine_epi64_epi8(I, B, 0);
    __m128i C_cols = _mm_gf2p8affine_epi64_epi8(B_t, A, 0);
    return _mm_gf2p8affine_epi64_epi8(I, C_cols, 0);
}

static inline void gpt_m8_pair_muladd(
    __m128i A, __m128i B0, __m128i B1,
    __m128i Add0, __m128i Add1,
    __m128i *Out0, __m128i *Out1)
{
    const uint64_t I_val = 0x0102040810204080ULL;
    __m128i I = _mm_set1_epi64x((long long)I_val);

    __m128i B_pair = _mm_unpacklo_epi64(B0, B1);
    __m128i A_pair = _mm_unpacklo_epi64(A, A);
    __m128i B_pair_t = _mm_gf2p8affine_epi64_epi8(I, B_pair, 0);
    __m128i C_cols = _mm_gf2p8affine_epi64_epi8(B_pair_t, A_pair, 0);
    __m128i C_rows = _mm_gf2p8affine_epi64_epi8(I, C_cols, 0);
    __m128i Add_pair = _mm_unpacklo_epi64(Add0, Add1);
    C_rows = _mm_xor_si128(C_rows, Add_pair);

    *Out0 = C_rows;
    *Out1 = _mm_srli_si128(C_rows, 8);
}

static inline __m128i gpt_m8_rand_matrix(void) {
    return _mm_cvtsi64_si128((long long)gpt1_random64());
}

static inline void gpt_pack_m16_blocks(
    __m128i LT, __m128i RT, __m128i LB, __m128i RB,
    __m128i *top, __m128i *bot)
{
    *top = _mm_unpacklo_epi8(RT, LT);
    *bot = _mm_unpacklo_epi8(RB, LB);
}

static inline void gpt_m16_make_perm16_fast_bitsliced(uint8_t p[16]) {
    for (int i = 0; i < 16; ++i) {
        p[i] = (uint8_t)i;
    }

    uint64_t r = gpt1_random64();
    for (int i = 15; i >= 8; --i) {
        const uint32_t bound = (uint32_t)i + 1U;
        uint32_t j = (uint32_t)(r & 0x0fU);
        r >>= 4;
        j -= (uint32_t)(j >= bound) * bound;
        uint8_t t = p[i];
        p[i] = p[j];
        p[j] = t;
    }
    for (int i = 7; i >= 4; --i) {
        const uint32_t bound = (uint32_t)i + 1U;
        uint32_t j = (uint32_t)(r & 0x07U);
        r >>= 3;
        j -= (uint32_t)(j >= bound) * bound;
        uint8_t t = p[i];
        p[i] = p[j];
        p[j] = t;
    }
    for (int i = 3; i >= 2; --i) {
        const uint32_t bound = (uint32_t)i + 1U;
        uint32_t j = (uint32_t)(r & 0x03U);
        r >>= 2;
        j -= (uint32_t)(j >= bound) * bound;
        uint8_t t = p[i];
        p[i] = p[j];
        p[j] = t;
    }

    const uint32_t j1 = (uint32_t)(r & 1U);
    uint8_t t = p[1];
    p[1] = p[j1];
    p[j1] = t;
}

static inline void gpt_m16_transpose_reg(__m128i *top, __m128i *bot) {
    const uint64_t I_val = 0x0102040810204080ULL;
    const __m128i I = _mm_set1_epi64x((long long)I_val);
    const __m128i mask_R = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m128i mask_L = _mm_setr_epi8(1, 3, 5, 7, 9, 11, 13, 15, -1, -1, -1, -1, -1, -1, -1, -1);

    const __m128i RT = _mm_shuffle_epi8(*top, mask_R);
    const __m128i LT = _mm_shuffle_epi8(*top, mask_L);
    const __m128i RB = _mm_shuffle_epi8(*bot, mask_R);
    const __m128i LB = _mm_shuffle_epi8(*bot, mask_L);

    const __m128i RT_t = _mm_gf2p8affine_epi64_epi8(I, RT, 0);
    const __m128i LT_t = _mm_gf2p8affine_epi64_epi8(I, LT, 0);
    const __m128i RB_t = _mm_gf2p8affine_epi64_epi8(I, RB, 0);
    const __m128i LB_t = _mm_gf2p8affine_epi64_epi8(I, LB, 0);

    *top = _mm_unpacklo_epi8(LB_t, LT_t);
    *bot = _mm_unpacklo_epi8(RB_t, RT_t);
}

static inline void gpt_m16_apply_dynamic_p_transpose_shuffle(
    __m128i base_top, __m128i base_bot,
    __m128i base_inv_top, __m128i base_inv_bot,
    M16 *Mat, M16 *Mat_inv)
{
    uint8_t p[16];
    gpt_m16_make_perm16_fast_bitsliced(p);

    gpt1_store_left_perm(base_top, base_bot, p, Mat);

    gpt_m16_transpose_reg(&base_inv_top, &base_inv_bot);

    uint16_t inv_rows[16];
    uint16_t permuted_inv_rows[16];
    _mm_storeu_si128((__m128i *)inv_rows, base_inv_top);
    _mm_storeu_si128((__m128i *)(inv_rows + 8), base_inv_bot);

    for (int i = 0; i < 16; ++i) {
        permuted_inv_rows[i] = inv_rows[p[i]];
    }

    base_inv_top = _mm_loadu_si128((const __m128i *)permuted_inv_rows);
    base_inv_bot = _mm_loadu_si128((const __m128i *)(permuted_inv_rows + 8));
    gpt_m16_transpose_reg(&base_inv_top, &base_inv_bot);

    _mm_storeu_si128((__m128i *)Mat_inv->M, base_inv_top);
    _mm_storeu_si128((__m128i *)(Mat_inv->M + 8), base_inv_bot);
}

static inline void gpt_m8_make_lower_unit_pair(__m128i *L, __m128i *Linv) {
    const uint64_t r = gpt1_random64();
    const uint64_t I_val = 0x0102040810204080ULL;

    const uint64_t m1_val = I_val
        ^ ((r & 0xFULL) << 36)
        ^ (((r >> 4) & 0xFULL) << 44)
        ^ (((r >> 8) & 0xFULL) << 52)
        ^ (((r >> 12) & 0xFULL) << 60);
    const uint64_t m2_val = I_val
        ^ (((r >> 16) & 0x3ULL) << 22)
        ^ (((r >> 18) & 0x3ULL) << 30)
        ^ (((r >> 20) & 0x3ULL) << 50)
        ^ (((r >> 22) & 0x3ULL) << 58);
    const uint64_t m3_val = I_val
        ^ (((r >> 24) & 1ULL) << 15)
        ^ (((r >> 25) & 1ULL) << 29)
        ^ (((r >> 26) & 1ULL) << 43)
        ^ (((r >> 27) & 1ULL) << 57);

    const __m128i M1 = _mm_cvtsi64_si128((long long)m1_val);
    const __m128i M2 = _mm_cvtsi64_si128((long long)m2_val);
    const __m128i M3 = _mm_cvtsi64_si128((long long)m3_val);

    *L = gpt_m8_mul(gpt_m8_mul(M1, M2), M3);
    *Linv = gpt_m8_mul(gpt_m8_mul(M3, M2), M1);
}

static inline void gpt_m8_make_upper_unit_pair(__m128i *U, __m128i *Uinv) {
    const uint64_t r = gpt1_random64();
    const uint64_t I_val = 0x0102040810204080ULL;

    const uint64_t n1_val = I_val
        ^ (r & 0xFULL)
        ^ (((r >> 4) & 0xFULL) << 8)
        ^ (((r >> 8) & 0xFULL) << 16)
        ^ (((r >> 12) & 0xFULL) << 24);
    const uint64_t n2_val = I_val
        ^ (((r >> 16) & 0x3ULL) << 4)
        ^ (((r >> 18) & 0x3ULL) << 12)
        ^ (((r >> 20) & 0x3ULL) << 32)
        ^ (((r >> 22) & 0x3ULL) << 40);
    const uint64_t n3_val = I_val
        ^ (((r >> 24) & 1ULL) << 6)
        ^ (((r >> 25) & 1ULL) << 20)
        ^ (((r >> 26) & 1ULL) << 34)
        ^ (((r >> 27) & 1ULL) << 48);

    const __m128i N1 = _mm_cvtsi64_si128((long long)n1_val);
    const __m128i N2 = _mm_cvtsi64_si128((long long)n2_val);
    const __m128i N3 = _mm_cvtsi64_si128((long long)n3_val);

    *U = gpt_m8_mul(gpt_m8_mul(N1, N2), N3);
    *Uinv = gpt_m8_mul(gpt_m8_mul(N3, N2), N1);
}

void wbmatrix2_genmatpair_m16_hbup(M16 *Mat, M16 *Mat_inv) {
    __m128i La, La_inv, Lb, Lb_inv;
    __m128i Ua, Ua_inv, Ub, Ub_inv;
    __m128i zero = _mm_setzero_si128();

    gpt_m8_make_lower_unit_pair(&La, &La_inv);
    gpt_m8_make_lower_unit_pair(&Lb, &Lb_inv);
    gpt_m8_make_upper_unit_pair(&Ua, &Ua_inv);
    gpt_m8_make_upper_unit_pair(&Ub, &Ub_inv);

    __m128i X = gpt_m8_rand_matrix();
    __m128i Y = gpt_m8_rand_matrix();

    __m128i XLa = gpt_m8_mul(X, La);
    __m128i YUb = gpt_m8_mul(Y, Ub);
    __m128i LbUb = gpt_m8_mul(Lb, Ub);

    __m128i base_LT, base_RT;
    gpt_m8_pair_muladd(La, Ua, YUb, zero, zero, &base_LT, &base_RT);

    __m128i base_LB, base_RB;
    gpt_m8_pair_muladd(XLa, Ua, YUb, zero, LbUb, &base_LB, &base_RB);

    __m128i UaInvY = gpt_m8_mul(Ua_inv, Y);
    __m128i LbInvX = gpt_m8_mul(Lb_inv, X);
    __m128i UaInvLaInv = gpt_m8_mul(Ua_inv, La_inv);

    __m128i inv_LT, inv_RT;
    gpt_m8_pair_muladd(UaInvY, LbInvX, Lb_inv, UaInvLaInv, zero, &inv_LT, &inv_RT);

    __m128i inv_LB, inv_RB;
    gpt_m8_pair_muladd(Ub_inv, LbInvX, Lb_inv, zero, zero, &inv_LB, &inv_RB);

    __m128i base_top, base_bot, base_inv_top, base_inv_bot;
    gpt_pack_m16_blocks(base_LT, base_RT, base_LB, base_RB, &base_top, &base_bot);
    gpt_pack_m16_blocks(inv_LT, inv_RT, inv_LB, inv_RB, &base_inv_top, &base_inv_bot);

    gpt_m16_apply_dynamic_p_transpose_shuffle(base_top, base_bot, base_inv_top, base_inv_bot, Mat, Mat_inv);
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

static void MatMulMatM256_Opt3_ptr(const M256 *Mat1, const M256 *Mat2, M256 *Mat) {
    if (Mat == Mat1 || Mat == Mat2) {
        M256 Mat1_copy = *Mat1;
        M256 Mat2_copy = *Mat2;
        matmul_m256_opt3_core(&Mat1_copy, &Mat2_copy, Mat);
        return;
    }
    matmul_m256_opt3_core(Mat1, Mat2, Mat);
}

static void xorM128_m256gpt(const M128 *a, const M128 *b, M128 *out) {
    for (int i = 0; i < 128; i++) {
        out->M[i][0] = a->M[i][0] ^ b->M[i][0];
        out->M[i][1] = a->M[i][1] ^ b->M[i][1];
    }
}

static void pack_M128_to_M256_m256gpt(
    const M128 *top_left, const M128 *top_right,
    const M128 *bot_left, const M128 *bot_right,
    M256 *out)
{
    for (int i = 0; i < 128; i++) {
        out->M[i][0] = top_left->M[i][0];
        out->M[i][1] = top_left->M[i][1];
        out->M[i][2] = top_right->M[i][0];
        out->M[i][3] = top_right->M[i][1];
        out->M[i + 128][0] = bot_left->M[i][0];
        out->M[i + 128][1] = bot_left->M[i][1];
        out->M[i + 128][2] = bot_right->M[i][0];
        out->M[i + 128][3] = bot_right->M[i][1];
    }
}

static inline uint8_t gpt256_take_random_byte(Gpt256ByteStream *stream) {
    if (stream->avail == 0) {
        stream->word = cus_random();
        stream->avail = 4;
    }
    uint8_t out = (uint8_t)(stream->word & 0xffU);
    stream->word >>= 8;
    stream->avail--;
    return out;
}

static void gpt256_make_perm_bytestream(uint8_t p[256]) {
    Gpt256ByteStream stream = {0, 0};
    for (int i = 0; i < 256; i++) {
        p[i] = (uint8_t)i;
    }
    for (int i = 255; i > 0; i--) {
        uint8_t r = gpt256_take_random_byte(&stream);
        int j = (int)(((uint32_t)r * (uint32_t)(i + 1)) >> 8);
        uint8_t tmp = p[i];
        p[i] = p[j];
        p[j] = tmp;
    }
}

static void gpt256_apply_dynamic_p_fastp_with(
    const M256 *base, const M256 *base_inv, Gpt256PinvMulFn pinv_mul,
    M256 *mat, M256 *mat_inv)
{
    uint8_t p[256];
    M256 pinv;

    gpt256_make_perm_bytestream(p);
    for (int i = 0; i < 256; i++) {
        const int src = p[i];
        const int word = i >> 6;
        const int bit = 63 - (i & 63);

        mat->M[i][0] = base->M[src][0];
        mat->M[i][1] = base->M[src][1];
        mat->M[i][2] = base->M[src][2];
        mat->M[i][3] = base->M[src][3];

        pinv.M[src][0] = 0;
        pinv.M[src][1] = 0;
        pinv.M[src][2] = 0;
        pinv.M[src][3] = 0;
        pinv.M[src][word] = 1ULL << bit;
    }

    pinv_mul(base_inv, &pinv, mat_inv);
}

static inline void mulM128_gpt12_m256gpt(const M128 *a, const M128 *b, M128 *out) {
    MatMulMatM128_GPT12_8x8tiles_rowdirect_aligned_total(a, b, out);
}

static uint64_t gpt256_take_bits(Gpt256Bitstream *stream, unsigned int n) {
    uint64_t out = 0;
    unsigned int have = 0;

    while (have < n) {
        if (stream->avail == 0) {
            const uint64_t high = (uint64_t)cus_random();
            const uint64_t low = (uint64_t)cus_random();
            stream->bits = (high << 32) | low;
            stream->avail = 64;
        }
        const unsigned int take = (n - have) < stream->avail ? (n - have) : stream->avail;
        const uint64_t mask = take == 64 ? ~0ULL : ((1ULL << take) - 1ULL);
        out |= (stream->bits & mask) << have;
        stream->bits >>= take;
        stream->avail -= take;
        have += take;
    }
    return out;
}

static inline void gpt256_m128_zero(M128 *out) {
    for (int i = 0; i < 128; i++) {
        out->M[i][0] = 0;
        out->M[i][1] = 0;
    }
}

static inline void gpt256_m64_zero(M64 *out) {
    for (int i = 0; i < 64; i++) {
        out->M[i] = 0;
    }
}

static inline void gpt256_m128_col_mask(int col, uint64_t row[2]) {
    row[0] = 0;
    row[1] = 0;
    row[col >> 6] = 1ULL << (63 - (col & 63));
}

static inline uint64_t gpt256_m128_test_col(const uint64_t row[2], int col) {
    return (row[col >> 6] >> (63 - (col & 63))) & 1ULL;
}

static void gpt256_m128_make_lower_unit_pair_direct(M128 *L, M128 *Linv) {
    Gpt256Bitstream stream = {0, 0};

    for (int i = 0; i < 128; i++) {
        uint64_t row[2];
        gpt256_m128_col_mask(i, row);
        for (int base = 0; base < i; base += 64) {
            int width = i - base;
            if (width > 64) {
                width = 64;
            }
            const uint64_t bits = gpt256_take_bits(&stream, (unsigned int)width);
            for (int j = 0; j < width; j++) {
                const int col = base + j;
                const uint64_t take = 0ULL - ((bits >> j) & 1ULL);
                row[col >> 6] ^= take & (1ULL << (63 - (col & 63)));
            }
        }
        L->M[i][0] = row[0];
        L->M[i][1] = row[1];
    }

    for (int i = 0; i < 128; i++) {
        uint64_t acc[2] = {L->M[i][0], L->M[i][1]};
        uint64_t inv[2];
        gpt256_m128_col_mask(i, inv);
        for (int k = i - 1; k >= 0; k--) {
            if (gpt256_m128_test_col(acc, k)) {
                uint64_t pivot[2];
                gpt256_m128_col_mask(k, pivot);
                inv[0] ^= pivot[0];
                inv[1] ^= pivot[1];
                acc[0] ^= L->M[k][0];
                acc[1] ^= L->M[k][1];
            }
        }
        Linv->M[i][0] = inv[0];
        Linv->M[i][1] = inv[1];
    }
}

static void gpt256_m128_make_upper_unit_pair_direct(M128 *U, M128 *Uinv) {
    Gpt256Bitstream stream = {0, 0};

    for (int i = 0; i < 128; i++) {
        uint64_t row[2];
        gpt256_m128_col_mask(i, row);
        for (int base = i + 1; base < 128; base += 64) {
            int width = 128 - base;
            if (width > 64) {
                width = 64;
            }
            const uint64_t bits = gpt256_take_bits(&stream, (unsigned int)width);
            for (int j = 0; j < width; j++) {
                const int col = base + j;
                const uint64_t take = 0ULL - ((bits >> j) & 1ULL);
                row[col >> 6] ^= take & (1ULL << (63 - (col & 63)));
            }
        }
        U->M[i][0] = row[0];
        U->M[i][1] = row[1];
    }

    for (int i = 127; i >= 0; i--) {
        uint64_t acc[2] = {U->M[i][0], U->M[i][1]};
        uint64_t inv[2];
        gpt256_m128_col_mask(i, inv);
        for (int k = i + 1; k < 128; k++) {
            if (gpt256_m128_test_col(acc, k)) {
                uint64_t pivot[2];
                gpt256_m128_col_mask(k, pivot);
                inv[0] ^= pivot[0];
                inv[1] ^= pivot[1];
                acc[0] ^= U->M[k][0];
                acc[1] ^= U->M[k][1];
            }
        }
        Uinv->M[i][0] = inv[0];
        Uinv->M[i][1] = inv[1];
    }
}

static inline uint64_t gpt256_m64_col_mask(int col) {
    return 1ULL << (63 - col);
}

static void gpt256_m64_make_lower_unit_pair_direct(M64 *L, M64 *Linv) {
    Gpt256Bitstream stream = {0, 0};

    for (int i = 0; i < 64; i++) {
        uint64_t row = gpt256_m64_col_mask(i);
        const uint64_t bits = i == 0 ? 0 : gpt256_take_bits(&stream, (unsigned int)i);
        for (int j = 0; j < i; j++) {
            const uint64_t take = 0ULL - ((bits >> j) & 1ULL);
            row ^= take & gpt256_m64_col_mask(j);
        }
        L->M[i] = row;
    }

    for (int i = 0; i < 64; i++) {
        uint64_t acc = L->M[i];
        uint64_t inv = gpt256_m64_col_mask(i);
        for (int k = i - 1; k >= 0; k--) {
            if (acc & gpt256_m64_col_mask(k)) {
                inv ^= gpt256_m64_col_mask(k);
                acc ^= L->M[k];
            }
        }
        Linv->M[i] = inv;
    }
}

static void gpt256_m64_make_upper_unit_pair_direct(M64 *U, M64 *Uinv) {
    Gpt256Bitstream stream = {0, 0};

    for (int i = 0; i < 64; i++) {
        uint64_t row = gpt256_m64_col_mask(i);
        const int width = 63 - i;
        const uint64_t bits = width == 0 ? 0 : gpt256_take_bits(&stream, (unsigned int)width);
        for (int j = i + 1; j < 64; j++) {
            const uint64_t take = 0ULL - ((bits >> (j - i - 1)) & 1ULL);
            row ^= take & gpt256_m64_col_mask(j);
        }
        U->M[i] = row;
    }

    for (int i = 63; i >= 0; i--) {
        uint64_t acc = U->M[i];
        uint64_t inv = gpt256_m64_col_mask(i);
        for (int k = i + 1; k < 64; k++) {
            if (acc & gpt256_m64_col_mask(k)) {
                inv ^= gpt256_m64_col_mask(k);
                acc ^= U->M[k];
            }
        }
        Uinv->M[i] = inv;
    }
}

static inline uint64_t gpt256_random64(void) {
    uint64_t high = (uint64_t)cus_random();
    uint64_t low = (uint64_t)cus_random();
    return (high << 32) | low;
}

static inline uint64_t gpt256_m8_mul_u64(uint64_t a, uint64_t b) {
    const uint64_t I_val = 0x0102040810204080ULL;
    __m128i I = _mm_set1_epi64x((long long)I_val);
    __m128i b_x = _mm_cvtsi64_si128((long long)b);
    __m128i a_x = _mm_cvtsi64_si128((long long)a);
    __m128i b_t = _mm_gf2p8affine_epi64_epi8(I, b_x, 0);
    __m128i c_cols = _mm_gf2p8affine_epi64_epi8(b_t, a_x, 0);
    __m128i c_rows = _mm_gf2p8affine_epi64_epi8(I, c_cols, 0);
    return (uint64_t)_mm_cvtsi128_si64(c_rows);
}

static inline void gpt256_m8_make_lower_unit_pair_u64(uint64_t *L, uint64_t *Linv) {
    const uint64_t r = gpt256_random64();
    const uint64_t I_val = 0x0102040810204080ULL;

    const uint64_t m1_val = I_val
        ^ ((r & 0xFULL) << 36)
        ^ (((r >> 4) & 0xFULL) << 44)
        ^ (((r >> 8) & 0xFULL) << 52)
        ^ (((r >> 12) & 0xFULL) << 60);
    const uint64_t m2_val = I_val
        ^ (((r >> 16) & 0x3ULL) << 22)
        ^ (((r >> 18) & 0x3ULL) << 30)
        ^ (((r >> 20) & 0x3ULL) << 50)
        ^ (((r >> 22) & 0x3ULL) << 58);
    const uint64_t m3_val = I_val
        ^ (((r >> 24) & 1ULL) << 15)
        ^ (((r >> 25) & 1ULL) << 29)
        ^ (((r >> 26) & 1ULL) << 43)
        ^ (((r >> 27) & 1ULL) << 57);

    *L = gpt256_m8_mul_u64(gpt256_m8_mul_u64(m1_val, m2_val), m3_val);
    *Linv = gpt256_m8_mul_u64(gpt256_m8_mul_u64(m3_val, m2_val), m1_val);
}

static inline void gpt256_m8_make_upper_unit_pair_u64(uint64_t *U, uint64_t *Uinv) {
    const uint64_t r = gpt256_random64();
    const uint64_t I_val = 0x0102040810204080ULL;

    const uint64_t n1_val = I_val
        ^ (r & 0xFULL)
        ^ (((r >> 4) & 0xFULL) << 8)
        ^ (((r >> 8) & 0xFULL) << 16)
        ^ (((r >> 12) & 0xFULL) << 24);
    const uint64_t n2_val = I_val
        ^ (((r >> 16) & 0x3ULL) << 4)
        ^ (((r >> 18) & 0x3ULL) << 12)
        ^ (((r >> 20) & 0x3ULL) << 32)
        ^ (((r >> 22) & 0x3ULL) << 40);
    const uint64_t n3_val = I_val
        ^ (((r >> 24) & 1ULL) << 6)
        ^ (((r >> 25) & 1ULL) << 20)
        ^ (((r >> 26) & 1ULL) << 34)
        ^ (((r >> 27) & 1ULL) << 48);

    *U = gpt256_m8_mul_u64(gpt256_m8_mul_u64(n1_val, n2_val), n3_val);
    *Uinv = gpt256_m8_mul_u64(gpt256_m8_mul_u64(n3_val, n2_val), n1_val);
}

static inline Gpt256M16RegPair gpt256_pack_m8_blocks_to_m16(
    uint64_t lt, uint64_t rt, uint64_t lb, uint64_t rb)
{
    Gpt256M16RegPair out;
    __m128i lt_x = _mm_cvtsi64_si128((long long)lt);
    __m128i rt_x = _mm_cvtsi64_si128((long long)rt);
    __m128i lb_x = _mm_cvtsi64_si128((long long)lb);
    __m128i rb_x = _mm_cvtsi64_si128((long long)rb);
    out.top = _mm_unpacklo_epi8(rt_x, lt_x);
    out.bot = _mm_unpacklo_epi8(rb_x, lb_x);
    return out;
}

static inline Gpt256M16RegPair gpt256_zero_m16_reg_pair(void) {
    Gpt256M16RegPair out;
    out.top = _mm_setzero_si128();
    out.bot = _mm_setzero_si128();
    return out;
}

static inline Gpt256M16RegPair gpt256_rand_m16_reg_pair(void) {
    Gpt256M16RegPair out;
    out.top = _mm_set_epi64x((long long)gpt256_random64(), (long long)gpt256_random64());
    out.bot = _mm_set_epi64x((long long)gpt256_random64(), (long long)gpt256_random64());
    return out;
}

static inline Gpt256M16RegPair gpt256_mul_m16_reg_pair(
    Gpt256M16RegPair a, Gpt256M16RegPair b)
{
    Gpt256M16RegPair out;
    MatMulMatM16_reg(a.top, a.bot, b.top, b.bot, &out.top, &out.bot);
    return out;
}

static inline void gpt256_store_m32_row8(uint32_t *dst, Gpt256M16RegPair left,
                                         Gpt256M16RegPair right, int half) {
    __m128i l = half == 0 ? left.top : left.bot;
    __m128i r = half == 0 ? right.top : right.bot;
    __m128i lo = _mm_unpacklo_epi16(r, l);
    __m128i hi = _mm_unpackhi_epi16(r, l);
    _mm_storeu_si128((__m128i *)dst, lo);
    _mm_storeu_si128((__m128i *)(dst + 4), hi);
}

static inline void gpt256_pack_m16_blocks_to_m32(
    Gpt256M16RegPair lt, Gpt256M16RegPair rt,
    Gpt256M16RegPair lb, Gpt256M16RegPair rb,
    M32 *out)
{
    gpt256_store_m32_row8(out->M, lt, rt, 0);
    gpt256_store_m32_row8(out->M + 8, lt, rt, 1);
    gpt256_store_m32_row8(out->M + 16, lb, rb, 0);
    gpt256_store_m32_row8(out->M + 24, lb, rb, 1);
}

static inline void gpt256_zeroM32(M32 *out) {
    for (int i = 0; i < 4; i++) {
        _mm256_storeu_si256((__m256i *)&out->M[i * 8], _mm256_setzero_si256());
    }
}

static inline void pack_M32_to_M64_m256gpt(
    const M32 *top_left, const M32 *top_right,
    const M32 *bot_left, const M32 *bot_right,
    M64 *out)
{
    for (int i = 0; i < 32; i++) {
        out->M[i] = ((uint64_t)top_left->M[i] << 32) | top_right->M[i];
        out->M[i + 32] = ((uint64_t)bot_left->M[i] << 32) | bot_right->M[i];
    }
}

static void gpt256_m32_make_lower_unit_pair_m16reg(M32 *L, M32 *Linv) {
    uint64_t La8, La8_inv, Lb8, Lb8_inv;
    uint64_t Lc8, Lc8_inv, Ld8, Ld8_inv;
    gpt256_m8_make_lower_unit_pair_u64(&La8, &La8_inv);
    gpt256_m8_make_lower_unit_pair_u64(&Lb8, &Lb8_inv);
    gpt256_m8_make_lower_unit_pair_u64(&Lc8, &Lc8_inv);
    gpt256_m8_make_lower_unit_pair_u64(&Ld8, &Ld8_inv);

    uint64_t X0 = gpt256_random64();
    uint64_t X0La = gpt256_m8_mul_u64(X0, La8);
    uint64_t LbInvX0 = gpt256_m8_mul_u64(Lb8_inv, X0);
    Gpt256M16RegPair La = gpt256_pack_m8_blocks_to_m16(La8, 0, X0La, Lb8);
    Gpt256M16RegPair La_inv = gpt256_pack_m8_blocks_to_m16(La8_inv, 0, LbInvX0, Lb8_inv);

    uint64_t X1 = gpt256_random64();
    uint64_t X1Lc = gpt256_m8_mul_u64(X1, Lc8);
    uint64_t LdInvX1 = gpt256_m8_mul_u64(Ld8_inv, X1);
    Gpt256M16RegPair Lb = gpt256_pack_m8_blocks_to_m16(Lc8, 0, X1Lc, Ld8);
    Gpt256M16RegPair Lb_inv = gpt256_pack_m8_blocks_to_m16(Lc8_inv, 0, LdInvX1, Ld8_inv);

    Gpt256M16RegPair zero = gpt256_zero_m16_reg_pair();
    Gpt256M16RegPair X = gpt256_rand_m16_reg_pair();
    Gpt256M16RegPair XLa = gpt256_mul_m16_reg_pair(X, La);
    Gpt256M16RegPair LbInvX = gpt256_mul_m16_reg_pair(Lb_inv, X);

    gpt256_pack_m16_blocks_to_m32(La, zero, XLa, Lb, L);
    gpt256_pack_m16_blocks_to_m32(La_inv, zero, LbInvX, Lb_inv, Linv);
}

static void gpt256_m32_make_upper_unit_pair_m16reg(M32 *U, M32 *Uinv) {
    uint64_t Ua8, Ua8_inv, Ub8, Ub8_inv;
    uint64_t Uc8, Uc8_inv, Ud8, Ud8_inv;
    gpt256_m8_make_upper_unit_pair_u64(&Ua8, &Ua8_inv);
    gpt256_m8_make_upper_unit_pair_u64(&Ub8, &Ub8_inv);
    gpt256_m8_make_upper_unit_pair_u64(&Uc8, &Uc8_inv);
    gpt256_m8_make_upper_unit_pair_u64(&Ud8, &Ud8_inv);

    uint64_t Y0 = gpt256_random64();
    uint64_t Y0Ub = gpt256_m8_mul_u64(Y0, Ub8);
    uint64_t UaInvY0 = gpt256_m8_mul_u64(Ua8_inv, Y0);
    Gpt256M16RegPair Ua = gpt256_pack_m8_blocks_to_m16(Ua8, Y0Ub, 0, Ub8);
    Gpt256M16RegPair Ua_inv = gpt256_pack_m8_blocks_to_m16(Ua8_inv, UaInvY0, 0, Ub8_inv);

    uint64_t Y1 = gpt256_random64();
    uint64_t Y1Ud = gpt256_m8_mul_u64(Y1, Ud8);
    uint64_t UcInvY1 = gpt256_m8_mul_u64(Uc8_inv, Y1);
    Gpt256M16RegPair Ub = gpt256_pack_m8_blocks_to_m16(Uc8, Y1Ud, 0, Ud8);
    Gpt256M16RegPair Ub_inv = gpt256_pack_m8_blocks_to_m16(Uc8_inv, UcInvY1, 0, Ud8_inv);

    Gpt256M16RegPair zero = gpt256_zero_m16_reg_pair();
    Gpt256M16RegPair Y = gpt256_rand_m16_reg_pair();
    Gpt256M16RegPair YUb = gpt256_mul_m16_reg_pair(Y, Ub);
    Gpt256M16RegPair UaInvY = gpt256_mul_m16_reg_pair(Ua_inv, Y);

    gpt256_pack_m16_blocks_to_m32(Ua, YUb, zero, Ub, U);
    gpt256_pack_m16_blocks_to_m32(Ua_inv, UaInvY, zero, Ub_inv, Uinv);
}

static void gpt256_m64_make_lower_unit_pair_m32units(M64 *L, M64 *Linv) {
    M32 La, La_inv, Lb, Lb_inv;
    M32 X, XLa, LbInvX, zero;

    gpt256_m32_make_lower_unit_pair_m16reg(&La, &La_inv);
    gpt256_m32_make_lower_unit_pair_m16reg(&Lb, &Lb_inv);
    wbmatrix2_rand_m32_no_reseed(&X);
    MatMulMatM32_GPT1_ptr(&X, &La, &XLa);
    MatMulMatM32_GPT1_ptr(&Lb_inv, &X, &LbInvX);
    gpt256_zeroM32(&zero);

    pack_M32_to_M64_m256gpt(&La, &zero, &XLa, &Lb, L);
    pack_M32_to_M64_m256gpt(&La_inv, &zero, &LbInvX, &Lb_inv, Linv);
}

static void gpt256_m64_make_upper_unit_pair_m32units(M64 *U, M64 *Uinv) {
    M32 Ua, Ua_inv, Ub, Ub_inv;
    M32 Y, YUb, UaInvY, zero;

    gpt256_m32_make_upper_unit_pair_m16reg(&Ua, &Ua_inv);
    gpt256_m32_make_upper_unit_pair_m16reg(&Ub, &Ub_inv);
    wbmatrix2_rand_m32_no_reseed(&Y);
    MatMulMatM32_GPT1_ptr(&Y, &Ub, &YUb);
    MatMulMatM32_GPT1_ptr(&Ua_inv, &Y, &UaInvY);
    gpt256_zeroM32(&zero);

    pack_M32_to_M64_m256gpt(&Ua, &YUb, &zero, &Ub, U);
    pack_M32_to_M64_m256gpt(&Ua_inv, &UaInvY, &zero, &Ub_inv, Uinv);
}

static inline void pack_M64_to_M128_m256gpt(
    const M64 *top_left, const M64 *top_right,
    const M64 *bot_left, const M64 *bot_right,
    M128 *out)
{
    for (int i = 0; i < 64; i++) {
        out->M[i][0] = top_left->M[i];
        out->M[i][1] = top_right->M[i];
        out->M[i + 64][0] = bot_left->M[i];
        out->M[i + 64][1] = bot_right->M[i];
    }
}

static void gpt256_m128_make_lower_unit_pair_m64units(M128 *L, M128 *Linv) {
    M64 La, La_inv, Lb, Lb_inv;
    M64 X, XLa, LbInvX, zero;

    gpt256_m64_make_lower_unit_pair_direct(&La, &La_inv);
    gpt256_m64_make_lower_unit_pair_direct(&Lb, &Lb_inv);
    wbmatrix2_rand_m64_no_reseed(&X);
    MatMulMatM64_GPT17_m32pairbt_rawaccum_aligned_total(&X, &La, &XLa);
    MatMulMatM64_GPT17_m32pairbt_rawaccum_aligned_total(&Lb_inv, &X, &LbInvX);
    gpt256_m64_zero(&zero);

    pack_M64_to_M128_m256gpt(&La, &zero, &XLa, &Lb, L);
    pack_M64_to_M128_m256gpt(&La_inv, &zero, &LbInvX, &Lb_inv, Linv);
}

static void gpt256_m128_make_upper_unit_pair_m64units(M128 *U, M128 *Uinv) {
    M64 Ua, Ua_inv, Ub, Ub_inv;
    M64 Y, YUb, UaInvY, zero;

    gpt256_m64_make_upper_unit_pair_direct(&Ua, &Ua_inv);
    gpt256_m64_make_upper_unit_pair_direct(&Ub, &Ub_inv);
    wbmatrix2_rand_m64_no_reseed(&Y);
    MatMulMatM64_GPT17_m32pairbt_rawaccum_aligned_total(&Y, &Ub, &YUb);
    MatMulMatM64_GPT17_m32pairbt_rawaccum_aligned_total(&Ua_inv, &Y, &UaInvY);
    gpt256_m64_zero(&zero);

    pack_M64_to_M128_m256gpt(&Ua, &YUb, &zero, &Ub, U);
    pack_M64_to_M128_m256gpt(&Ua_inv, &UaInvY, &zero, &Ub_inv, Uinv);
}

static void gpt256_m128_make_lower_unit_pair_m32units(M128 *L, M128 *Linv) {
    M64 La, La_inv, Lb, Lb_inv;
    M64 X, XLa, LbInvX, zero;

    gpt256_m64_make_lower_unit_pair_m32units(&La, &La_inv);
    gpt256_m64_make_lower_unit_pair_m32units(&Lb, &Lb_inv);
    wbmatrix2_rand_m64_no_reseed(&X);
    MatMulMatM64_GPT17_m32pairbt_rawaccum_aligned_total(&X, &La, &XLa);
    MatMulMatM64_GPT17_m32pairbt_rawaccum_aligned_total(&Lb_inv, &X, &LbInvX);
    gpt256_m64_zero(&zero);

    pack_M64_to_M128_m256gpt(&La, &zero, &XLa, &Lb, L);
    pack_M64_to_M128_m256gpt(&La_inv, &zero, &LbInvX, &Lb_inv, Linv);
}

static void gpt256_m128_make_upper_unit_pair_m32units(M128 *U, M128 *Uinv) {
    M64 Ua, Ua_inv, Ub, Ub_inv;
    M64 Y, YUb, UaInvY, zero;

    gpt256_m64_make_upper_unit_pair_m32units(&Ua, &Ua_inv);
    gpt256_m64_make_upper_unit_pair_m32units(&Ub, &Ub_inv);
    wbmatrix2_rand_m64_no_reseed(&Y);
    MatMulMatM64_GPT17_m32pairbt_rawaccum_aligned_total(&Y, &Ub, &YUb);
    MatMulMatM64_GPT17_m32pairbt_rawaccum_aligned_total(&Ua_inv, &Y, &UaInvY);
    gpt256_m64_zero(&zero);

    pack_M64_to_M128_m256gpt(&Ua, &YUb, &zero, &Ub, U);
    pack_M64_to_M128_m256gpt(&Ua_inv, &UaInvY, &zero, &Ub_inv, Uinv);
}

static inline void muladdM128_mode_m256gpt(
    const M128 *a, const M128 *b, const M128 *add,
    Gpt256M128MulFn mul, M128 *out)
{
    M128 prod;
    mul(a, b, &prod);
    xorM128_m256gpt(&prod, add, out);
}

static void muladd_two_left_two_right_m128_m256gpt(
    const M128 *left0, const M128 *left1,
    const M128 *right0, const M128 *right1,
    const M128 *add00, const M128 *add01,
    const M128 *add10, const M128 *add11,
    int rowdirect_core,
    M128 *out00, M128 *out01,
    M128 *out10, M128 *out11)
{
    M128 prod;
    M128PreparedB_8x8Tiles_GPT prepared0, prepared1;

    MatMulMatM128_GPT5_prepareB_8x8tiles(right0, &prepared0);
    MatMulMatM128_GPT5_prepareB_8x8tiles(right1, &prepared1);

#define GPT256_M128_PREPARED_MULADD(LEFT, PREPARED, ADD, OUT) do { \
        MatMulMatM128_GPT10_preparedB_8x8tiles_rowdirect((LEFT), (PREPARED), &prod); \
        xorM128_m256gpt(&prod, (ADD), (OUT)); \
    } while (0)

    (void)rowdirect_core;

    GPT256_M128_PREPARED_MULADD(left0, &prepared0, add00, out00);
    GPT256_M128_PREPARED_MULADD(left0, &prepared1, add01, out01);
    GPT256_M128_PREPARED_MULADD(left1, &prepared0, add10, out10);
    GPT256_M128_PREPARED_MULADD(left1, &prepared1, add11, out11);

#undef GPT256_M128_PREPARED_MULADD
}

static void genMatpairM256_radix2_unitriangular_plu_recursive_gpt(
    int unit_leaf_mode,
    int two_left_two_right,
    int rowdirect_core,
    Gpt256M128MulFn mul,
    Gpt256PinvMulFn pinv_mul,
    M256 *mat,
    M256 *mat_inv)
{
    M128 La, La_inv, Lb, Lb_inv;
    M128 Ua, Ua_inv, Ub, Ub_inv;
    M128 X, Y;
    M128 XLa, YUb, LbUb;
    M128 base_lt, base_rt, base_lb, base_rb;
    M128 UaInvY, LbInvX, UaInvLaInv;
    M128 inv_lt, inv_rt, inv_lb, inv_rb;
    M128 zero;
    M256 base, base_inv;

    if (unit_leaf_mode == 2) {
        gpt256_m128_make_lower_unit_pair_m32units(&La, &La_inv);
        gpt256_m128_make_lower_unit_pair_m32units(&Lb, &Lb_inv);
        gpt256_m128_make_upper_unit_pair_m32units(&Ua, &Ua_inv);
        gpt256_m128_make_upper_unit_pair_m32units(&Ub, &Ub_inv);
    } else if (unit_leaf_mode == 1) {
        gpt256_m128_make_lower_unit_pair_m64units(&La, &La_inv);
        gpt256_m128_make_lower_unit_pair_m64units(&Lb, &Lb_inv);
        gpt256_m128_make_upper_unit_pair_m64units(&Ua, &Ua_inv);
        gpt256_m128_make_upper_unit_pair_m64units(&Ub, &Ub_inv);
    } else {
        gpt256_m128_make_lower_unit_pair_direct(&La, &La_inv);
        gpt256_m128_make_lower_unit_pair_direct(&Lb, &Lb_inv);
        gpt256_m128_make_upper_unit_pair_direct(&Ua, &Ua_inv);
        gpt256_m128_make_upper_unit_pair_direct(&Ub, &Ub_inv);
    }
    wbmatrix2_rand_m128_no_reseed(&X);
    wbmatrix2_rand_m128_no_reseed(&Y);

    mul(&X, &La, &XLa);
    mul(&Y, &Ub, &YUb);
    mul(&Lb, &Ub, &LbUb);

    if (two_left_two_right) {
        gpt256_m128_zero(&zero);
        muladd_two_left_two_right_m128_m256gpt(
            &La, &XLa, &Ua, &YUb,
            &zero, &zero, &zero, &LbUb,
            rowdirect_core,
            &base_lt, &base_rt, &base_lb, &base_rb);
    } else {
        mul(&La, &Ua, &base_lt);
        mul(&La, &YUb, &base_rt);
        mul(&XLa, &Ua, &base_lb);
        muladdM128_mode_m256gpt(&XLa, &YUb, &LbUb, mul, &base_rb);
    }

    mul(&Ua_inv, &Y, &UaInvY);
    mul(&Lb_inv, &X, &LbInvX);
    mul(&Ua_inv, &La_inv, &UaInvLaInv);

    if (two_left_two_right) {
        gpt256_m128_zero(&zero);
        muladd_two_left_two_right_m128_m256gpt(
            &UaInvY, &Ub_inv, &LbInvX, &Lb_inv,
            &UaInvLaInv, &zero, &zero, &zero,
            rowdirect_core,
            &inv_lt, &inv_rt, &inv_lb, &inv_rb);
    } else {
        muladdM128_mode_m256gpt(&UaInvY, &LbInvX, &UaInvLaInv, mul, &inv_lt);
        mul(&UaInvY, &Lb_inv, &inv_rt);
        mul(&Ub_inv, &LbInvX, &inv_lb);
        mul(&Ub_inv, &Lb_inv, &inv_rb);
    }

    pack_M128_to_M256_m256gpt(&base_lt, &base_rt, &base_lb, &base_rb, &base);
    pack_M128_to_M256_m256gpt(&inv_lt, &inv_rt, &inv_lb, &inv_rb, &base_inv);
    gpt256_apply_dynamic_p_fastp_with(&base, &base_inv, pinv_mul, mat, mat_inv);
}

void wbmatrix2_genmatpair_m256_hbup(
    M256 *Mat, M256 *Mat_inv)
{
    genMatpairM256_radix2_unitriangular_plu_recursive_gpt(
        2, 1, 1, mulM128_gpt12_m256gpt,
        MatMulMatM256_Opt3_ptr, Mat, Mat_inv);
}

static inline void MatMulMatM32_GPT1_core(
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

#define GPT1_CALC_M32_ROW(A_ROW, C_OUT) do { \
        __m256i aik0 = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8((A_ROW), m0), perm_ctrl); \
        __m256i aik1 = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8((A_ROW), m1), perm_ctrl); \
        __m256i aik2 = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8((A_ROW), m2), perm_ctrl); \
        __m256i aik3 = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8((A_ROW), m3), perm_ctrl); \
        __m256i t0 = _mm256_gf2p8affine_epi64_epi8(BT0, aik0, 0); \
        __m256i t1 = _mm256_gf2p8affine_epi64_epi8(BT1, aik1, 0); \
        __m256i t2 = _mm256_gf2p8affine_epi64_epi8(BT2, aik2, 0); \
        __m256i t3 = _mm256_gf2p8affine_epi64_epi8(BT3, aik3, 0); \
        __m256i acc01 = _mm256_xor_si256(t0, t1); \
        __m256i acc23 = _mm256_xor_si256(t2, t3); \
        __m256i acc = _mm256_xor_si256(acc01, acc23); \
        (C_OUT) = _mm256_gf2p8affine_epi64_epi8(I, acc, 0); \
    } while (0)

    GPT1_CALC_M32_ROW(A0, *C0);
    GPT1_CALC_M32_ROW(A1, *C1);
    GPT1_CALC_M32_ROW(A2, *C2);
    GPT1_CALC_M32_ROW(A3, *C3);

#undef GPT1_CALC_M32_ROW

    const __m256i inv_perm = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
    const __m256i inv_shuf = _mm256_setr_epi8(
        0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
        0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);

    *C0 = _mm256_shuffle_epi8(_mm256_permutevar8x32_epi32(*C0, inv_perm), inv_shuf);
    *C1 = _mm256_shuffle_epi8(_mm256_permutevar8x32_epi32(*C1, inv_perm), inv_shuf);
    *C2 = _mm256_shuffle_epi8(_mm256_permutevar8x32_epi32(*C2, inv_perm), inv_shuf);
    *C3 = _mm256_shuffle_epi8(_mm256_permutevar8x32_epi32(*C3, inv_perm), inv_shuf);
}

static inline void MatMulMatM32_GPT2_prepareB_core(
    __m256i B0, __m256i B1, __m256i B2, __m256i B3,
    M32PreparedB_GPT *PreparedB)
{
    const __m256i I = _mm256_set1_epi64x(0x0102040810204080ULL);

    const __m256i shuf_B = _mm256_setr_epi8(
        0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
        0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    const __m256i perm_B = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

    PreparedB->BT[0] = _mm256_gf2p8affine_epi64_epi8(
        I, _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(B0, shuf_B), perm_B), 0);
    PreparedB->BT[1] = _mm256_gf2p8affine_epi64_epi8(
        I, _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(B1, shuf_B), perm_B), 0);
    PreparedB->BT[2] = _mm256_gf2p8affine_epi64_epi8(
        I, _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(B2, shuf_B), perm_B), 0);
    PreparedB->BT[3] = _mm256_gf2p8affine_epi64_epi8(
        I, _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(B3, shuf_B), perm_B), 0);
}

static inline void MatMulMatM32_GPT2_preparedB_core(
    __m256i A0, __m256i A1, __m256i A2, __m256i A3,
    const M32PreparedB_GPT *PreparedB,
    __m256i *C0, __m256i *C1, __m256i *C2, __m256i *C3)
{
    const __m256i I = _mm256_set1_epi64x(0x0102040810204080ULL);

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

#define GPT2_CALC_M32_ROW(A_ROW, C_OUT) do { \
        __m256i aik0 = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8((A_ROW), m0), perm_ctrl); \
        __m256i aik1 = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8((A_ROW), m1), perm_ctrl); \
        __m256i aik2 = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8((A_ROW), m2), perm_ctrl); \
        __m256i aik3 = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8((A_ROW), m3), perm_ctrl); \
        __m256i t0 = _mm256_gf2p8affine_epi64_epi8(PreparedB->BT[0], aik0, 0); \
        __m256i t1 = _mm256_gf2p8affine_epi64_epi8(PreparedB->BT[1], aik1, 0); \
        __m256i t2 = _mm256_gf2p8affine_epi64_epi8(PreparedB->BT[2], aik2, 0); \
        __m256i t3 = _mm256_gf2p8affine_epi64_epi8(PreparedB->BT[3], aik3, 0); \
        __m256i acc01 = _mm256_xor_si256(t0, t1); \
        __m256i acc23 = _mm256_xor_si256(t2, t3); \
        __m256i acc = _mm256_xor_si256(acc01, acc23); \
        (C_OUT) = _mm256_gf2p8affine_epi64_epi8(I, acc, 0); \
    } while (0)

    GPT2_CALC_M32_ROW(A0, *C0);
    GPT2_CALC_M32_ROW(A1, *C1);
    GPT2_CALC_M32_ROW(A2, *C2);
    GPT2_CALC_M32_ROW(A3, *C3);

#undef GPT2_CALC_M32_ROW

    const __m256i inv_perm = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
    const __m256i inv_shuf = _mm256_setr_epi8(
        0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
        0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);

    *C0 = _mm256_shuffle_epi8(_mm256_permutevar8x32_epi32(*C0, inv_perm), inv_shuf);
    *C1 = _mm256_shuffle_epi8(_mm256_permutevar8x32_epi32(*C1, inv_perm), inv_shuf);
    *C2 = _mm256_shuffle_epi8(_mm256_permutevar8x32_epi32(*C2, inv_perm), inv_shuf);
    *C3 = _mm256_shuffle_epi8(_mm256_permutevar8x32_epi32(*C3, inv_perm), inv_shuf);
}

static void MatMulMatM32_GPT2_prepareB(const M32 *Mat2, M32PreparedB_GPT *PreparedB)
{
    __m256i B0 = _mm256_loadu_si256((const __m256i *)&Mat2->M[0]);
    __m256i B1 = _mm256_loadu_si256((const __m256i *)&Mat2->M[8]);
    __m256i B2 = _mm256_loadu_si256((const __m256i *)&Mat2->M[16]);
    __m256i B3 = _mm256_loadu_si256((const __m256i *)&Mat2->M[24]);

    MatMulMatM32_GPT2_prepareB_core(B0, B1, B2, B3, PreparedB);
}

static void MatMulMatM32_GPT2_preparedB(const M32 *Mat1, const M32PreparedB_GPT *PreparedB, M32 *Mat)
{
    __m256i A0 = _mm256_loadu_si256((const __m256i *)&Mat1->M[0]);
    __m256i A1 = _mm256_loadu_si256((const __m256i *)&Mat1->M[8]);
    __m256i A2 = _mm256_loadu_si256((const __m256i *)&Mat1->M[16]);
    __m256i A3 = _mm256_loadu_si256((const __m256i *)&Mat1->M[24]);
    __m256i C0, C1, C2, C3;

    MatMulMatM32_GPT2_preparedB_core(A0, A1, A2, A3, PreparedB, &C0, &C1, &C2, &C3);

    _mm256_storeu_si256((__m256i *)&Mat->M[0], C0);
    _mm256_storeu_si256((__m256i *)&Mat->M[8], C1);
    _mm256_storeu_si256((__m256i *)&Mat->M[16], C2);
    _mm256_storeu_si256((__m256i *)&Mat->M[24], C3);
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

static inline void MatMulMatM32_GPT4_preparedB_rowdirect_core(
    __m256i A0, __m256i A1, __m256i A2, __m256i A3,
    const M32PreparedB_GPT *PreparedB,
    __m256i *C0, __m256i *C1, __m256i *C2, __m256i *C3)
{
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

#define GPT4_PREP_CALC_M32_ROW(A_ROW, C_OUT) do { \
        __m256i aik0 = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8((A_ROW), m0), perm_ctrl); \
        __m256i aik1 = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8((A_ROW), m1), perm_ctrl); \
        __m256i aik2 = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8((A_ROW), m2), perm_ctrl); \
        __m256i aik3 = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8((A_ROW), m3), perm_ctrl); \
        __m256i t0 = _mm256_gf2p8affine_epi64_epi8(aik0, PreparedB->BT[0], 0); \
        __m256i t1 = _mm256_gf2p8affine_epi64_epi8(aik1, PreparedB->BT[1], 0); \
        __m256i t2 = _mm256_gf2p8affine_epi64_epi8(aik2, PreparedB->BT[2], 0); \
        __m256i t3 = _mm256_gf2p8affine_epi64_epi8(aik3, PreparedB->BT[3], 0); \
        __m256i acc01 = _mm256_xor_si256(t0, t1); \
        __m256i acc23 = _mm256_xor_si256(t2, t3); \
        (C_OUT) = _mm256_xor_si256(acc01, acc23); \
    } while (0)

    GPT4_PREP_CALC_M32_ROW(A0, *C0);
    GPT4_PREP_CALC_M32_ROW(A1, *C1);
    GPT4_PREP_CALC_M32_ROW(A2, *C2);
    GPT4_PREP_CALC_M32_ROW(A3, *C3);

#undef GPT4_PREP_CALC_M32_ROW

    const __m256i inv_perm = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
    const __m256i inv_shuf = _mm256_setr_epi8(
        0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
        0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);

    *C0 = _mm256_shuffle_epi8(_mm256_permutevar8x32_epi32(*C0, inv_perm), inv_shuf);
    *C1 = _mm256_shuffle_epi8(_mm256_permutevar8x32_epi32(*C1, inv_perm), inv_shuf);
    *C2 = _mm256_shuffle_epi8(_mm256_permutevar8x32_epi32(*C2, inv_perm), inv_shuf);
    *C3 = _mm256_shuffle_epi8(_mm256_permutevar8x32_epi32(*C3, inv_perm), inv_shuf);
}

static void MatMulMatM32_GPT4_preparedB_rowdirect(const M32 *Mat1, const M32PreparedB_GPT *PreparedB, M32 *Mat)
{
    __m256i A0 = _mm256_loadu_si256((const __m256i *)&Mat1->M[0]);
    __m256i A1 = _mm256_loadu_si256((const __m256i *)&Mat1->M[8]);
    __m256i A2 = _mm256_loadu_si256((const __m256i *)&Mat1->M[16]);
    __m256i A3 = _mm256_loadu_si256((const __m256i *)&Mat1->M[24]);
    __m256i C0, C1, C2, C3;

    MatMulMatM32_GPT4_preparedB_rowdirect_core(A0, A1, A2, A3, PreparedB, &C0, &C1, &C2, &C3);

    _mm256_storeu_si256((__m256i *)&Mat->M[0], C0);
    _mm256_storeu_si256((__m256i *)&Mat->M[8], C1);
    _mm256_storeu_si256((__m256i *)&Mat->M[16], C2);
    _mm256_storeu_si256((__m256i *)&Mat->M[24], C3);
}

static void MatMulMatM32_GPT1_ptr(const M32 *Mat1, const M32 *Mat2, M32 *Mat) {
    __m256i A0 = _mm256_loadu_si256((const __m256i *)&Mat1->M[0]);
    __m256i A1 = _mm256_loadu_si256((const __m256i *)&Mat1->M[8]);
    __m256i A2 = _mm256_loadu_si256((const __m256i *)&Mat1->M[16]);
    __m256i A3 = _mm256_loadu_si256((const __m256i *)&Mat1->M[24]);
    __m256i B0 = _mm256_loadu_si256((const __m256i *)&Mat2->M[0]);
    __m256i B1 = _mm256_loadu_si256((const __m256i *)&Mat2->M[8]);
    __m256i B2 = _mm256_loadu_si256((const __m256i *)&Mat2->M[16]);
    __m256i B3 = _mm256_loadu_si256((const __m256i *)&Mat2->M[24]);
    __m256i C0, C1, C2, C3;

    MatMulMatM32_GPT1_core(A0, A1, A2, A3, B0, B1, B2, B3, &C0, &C1, &C2, &C3);

    _mm256_storeu_si256((__m256i *)&Mat->M[0], C0);
    _mm256_storeu_si256((__m256i *)&Mat->M[8], C1);
    _mm256_storeu_si256((__m256i *)&Mat->M[16], C2);
    _mm256_storeu_si256((__m256i *)&Mat->M[24], C3);
}

static inline uint64_t gpt32_random64(void) {
    uint64_t high = (uint64_t)cus_random();
    uint64_t low = (uint64_t)cus_random();
    return (high << 32) | low;
}

static inline M16RegPair_GPT gpt32_rand_m16_reg(void) {
    M16RegPair_GPT out;
    out.top = _mm_set_epi64x((long long)gpt32_random64(), (long long)gpt32_random64());
    out.bot = _mm_set_epi64x((long long)gpt32_random64(), (long long)gpt32_random64());
    return out;
}

static inline M16RegPair_GPT gpt32_zero_m16_reg(void) {
    M16RegPair_GPT out;
    out.top = _mm_setzero_si128();
    out.bot = _mm_setzero_si128();
    return out;
}

static inline M16RegPair_GPT gpt32_mul_m16(M16RegPair_GPT a, M16RegPair_GPT b) {
    M16RegPair_GPT out;
    MatMulMatM16_reg(a.top, a.bot, b.top, b.bot, &out.top, &out.bot);
    return out;
}

static inline void gpt32_prepare_m16_right(M16RegPair_GPT right,
                                           M16PreparedB_GPT *prepared) {
    const uint64_t I_val = 0x0102040810204080ULL;
    __m128i I = _mm_set1_epi64x((long long)I_val);
    const __m128i mask_B = _mm_setr_epi8(
        1, 3, 5, 7, 9, 11, 13, 15,
        0, 2, 4, 6, 8, 10, 12, 14);

    __m128i b_top = _mm_shuffle_epi8(right.top, mask_B);
    __m128i b_bot = _mm_shuffle_epi8(right.bot, mask_B);
    prepared->B_top_t = _mm_gf2p8affine_epi64_epi8(I, b_top, 0);
    prepared->B_bot_t = _mm_gf2p8affine_epi64_epi8(I, b_bot, 0);
}

static inline void gpt32_muladd_two_left_two_right_m16(
    M16RegPair_GPT left0, M16RegPair_GPT left1,
    M16RegPair_GPT right0, M16RegPair_GPT right1,
    M16RegPair_GPT add00, M16RegPair_GPT add01,
    M16RegPair_GPT add10, M16RegPair_GPT add11,
    M16RegPair_GPT *out00, M16RegPair_GPT *out01,
    M16RegPair_GPT *out10, M16RegPair_GPT *out11) {
    M16PreparedB_GPT prepared0, prepared1;
    gpt32_prepare_m16_right(right0, &prepared0);
    gpt32_prepare_m16_right(right1, &prepared1);

    const uint64_t I_val = 0x0102040810204080ULL;
    __m128i I = _mm_set1_epi64x((long long)I_val);
    const __m128i mask_A_H = _mm_setr_epi8(
        1, 3, 5, 7, 9, 11, 13, 15,
        1, 3, 5, 7, 9, 11, 13, 15);
    const __m128i mask_A_L = _mm_setr_epi8(
        0, 2, 4, 6, 8, 10, 12, 14,
        0, 2, 4, 6, 8, 10, 12, 14);
    const __m128i store_mask = _mm_setr_epi8(
        8, 0, 9, 1, 10, 2, 11, 3,
        12, 4, 13, 5, 14, 6, 15, 7);

#define GPT32_SPLIT_LEFT(LEFT, A00, A01, A10, A11) do { \
        (A00) = _mm_shuffle_epi8((LEFT).top, mask_A_H); \
        (A01) = _mm_shuffle_epi8((LEFT).top, mask_A_L); \
        (A10) = _mm_shuffle_epi8((LEFT).bot, mask_A_H); \
        (A11) = _mm_shuffle_epi8((LEFT).bot, mask_A_L); \
    } while (0)

#define GPT32_CALC_PREPARED_WITH_SPLIT(PREPARED, A00, A01, A10, A11, ADD, OUT) do { \
        __m128i top_cols = _mm_xor_si128( \
            _mm_gf2p8affine_epi64_epi8((PREPARED).B_top_t, (A00), 0), \
            _mm_gf2p8affine_epi64_epi8((PREPARED).B_bot_t, (A01), 0)); \
        __m128i bot_cols = _mm_xor_si128( \
            _mm_gf2p8affine_epi64_epi8((PREPARED).B_top_t, (A10), 0), \
            _mm_gf2p8affine_epi64_epi8((PREPARED).B_bot_t, (A11), 0)); \
        (OUT)->top = _mm_xor_si128( \
            _mm_shuffle_epi8(_mm_gf2p8affine_epi64_epi8(I, top_cols, 0), store_mask), \
            (ADD).top); \
        (OUT)->bot = _mm_xor_si128( \
            _mm_shuffle_epi8(_mm_gf2p8affine_epi64_epi8(I, bot_cols, 0), store_mask), \
            (ADD).bot); \
    } while (0)

    __m128i a00, a01, a10, a11;
    GPT32_SPLIT_LEFT(left0, a00, a01, a10, a11);
    GPT32_CALC_PREPARED_WITH_SPLIT(prepared0, a00, a01, a10, a11, add00, out00);
    GPT32_CALC_PREPARED_WITH_SPLIT(prepared1, a00, a01, a10, a11, add01, out01);

    GPT32_SPLIT_LEFT(left1, a00, a01, a10, a11);
    GPT32_CALC_PREPARED_WITH_SPLIT(prepared0, a00, a01, a10, a11, add10, out10);
    GPT32_CALC_PREPARED_WITH_SPLIT(prepared1, a00, a01, a10, a11, add11, out11);

#undef GPT32_CALC_PREPARED_WITH_SPLIT
#undef GPT32_SPLIT_LEFT
}

static inline void gpt32_store_m32_row8(uint32_t *dst, M16RegPair_GPT left,
                                        M16RegPair_GPT right, int half) {
    __m128i l = half == 0 ? left.top : left.bot;
    __m128i r = half == 0 ? right.top : right.bot;
    __m128i lo = _mm_unpacklo_epi16(r, l);
    __m128i hi = _mm_unpackhi_epi16(r, l);
    _mm_storeu_si128((__m128i *)dst, lo);
    _mm_storeu_si128((__m128i *)(dst + 4), hi);
}

static inline void gpt32_pack_m16_blocks(M16RegPair_GPT lt, M16RegPair_GPT rt,
                                         M16RegPair_GPT lb, M16RegPair_GPT rb,
                                         M32 *out) {
    gpt32_store_m32_row8(out->M, lt, rt, 0);
    gpt32_store_m32_row8(out->M + 8, lt, rt, 1);
    gpt32_store_m32_row8(out->M + 16, lb, rb, 0);
    gpt32_store_m32_row8(out->M + 24, lb, rb, 1);
}

static inline uint32_t gpt32_bounded_rand(uint32_t bound) {
    return (uint32_t)(((uint64_t)cus_random() * (uint64_t)bound) >> 32);
}

static inline uint32_t gpt32_bitstream_take(Gpt32BitstreamP *stream, int count) {
    while (stream->available < count) {
        stream->bits |= (uint64_t)cus_random() << stream->available;
        stream->available += 32;
    }

    uint32_t out = (uint32_t)(stream->bits & ((1u << count) - 1u));
    stream->bits >>= count;
    stream->available -= count;
    return out;
}

static inline uint32_t gpt32_bounded_rand_bitstream(Gpt32BitstreamP *stream,
                                                    uint32_t bound) {
    uint32_t x = bound - 1u;
    int bits = 0;
    do {
        bits++;
        x >>= 1;
    } while (x != 0);

    uint32_t raw = gpt32_bitstream_take(stream, bits);
    return (uint32_t)(((uint64_t)raw * (uint64_t)bound) >> bits);
}

static void gpt32_apply_dynamic_p_matmul(const M32 *base, const M32 *base_inv,
                                         M32 *mat, M32 *mat_inv) {
    uint8_t p[32];
    for (int i = 0; i < 32; i++) {
        p[i] = (uint8_t)i;
    }

    for (int i = 31; i > 0; i--) {
        int j = (int)gpt32_bounded_rand((uint32_t)(i + 1));
        uint8_t tmp = p[i];
        p[i] = p[j];
        p[j] = tmp;
    }

    M32 pinv;
    for (int i = 0; i < 32; i++) {
        mat->M[i] = base->M[p[i]];
        pinv.M[i] = 0;
    }

    for (int i = 0; i < 32; i++) {
        pinv.M[p[i]] = 1u << (31 - i);
    }

    MatMulMatM32_GPT1_ptr(base_inv, &pinv, mat_inv);
}

static void gpt32_apply_dynamic_p_matmul_rowdirect(const M32 *base, const M32 *base_inv,
                                                   M32 *mat, M32 *mat_inv) {
    uint8_t p[32];
    for (int i = 0; i < 32; i++) {
        p[i] = (uint8_t)i;
    }

    for (int i = 31; i > 0; i--) {
        int j = (int)gpt32_bounded_rand((uint32_t)(i + 1));
        uint8_t tmp = p[i];
        p[i] = p[j];
        p[j] = tmp;
    }

    M32 pinv;
    for (int i = 0; i < 32; i++) {
        mat->M[i] = base->M[p[i]];
        pinv.M[i] = 0;
    }

    for (int i = 0; i < 32; i++) {
        pinv.M[p[i]] = 1u << (31 - i);
    }

    MatMulMatM32_GPT4_rowdirect_ptr(base_inv, &pinv, mat_inv);
}

static void gpt32_apply_dynamic_p_matmul_bitstream(const M32 *base, const M32 *base_inv,
                                                   int rowdirect_pinv,
                                                   M32 *mat, M32 *mat_inv) {
    uint8_t p[32];
    for (int i = 0; i < 32; i++) {
        p[i] = (uint8_t)i;
    }

    Gpt32BitstreamP stream = {0, 0};
    for (int i = 31; i > 0; i--) {
        int j = (int)gpt32_bounded_rand_bitstream(&stream, (uint32_t)(i + 1));
        uint8_t tmp = p[i];
        p[i] = p[j];
        p[j] = tmp;
    }

    M32 pinv;
    for (int i = 0; i < 32; i++) {
        mat->M[i] = base->M[p[i]];
        pinv.M[i] = 0;
    }

    for (int i = 0; i < 32; i++) {
        pinv.M[p[i]] = 1u << (31 - i);
    }

    if (rowdirect_pinv) {
        MatMulMatM32_GPT4_rowdirect_ptr(base_inv, &pinv, mat_inv);
    } else {
        MatMulMatM32_GPT1_ptr(base_inv, &pinv, mat_inv);
    }
}

static inline uint64_t gpt32_m8_mul_u64(uint64_t a, uint64_t b) {
    const uint64_t I_val = 0x0102040810204080ULL;
    __m128i I = _mm_set1_epi64x((long long)I_val);
    __m128i b_x = _mm_cvtsi64_si128((long long)b);
    __m128i a_x = _mm_cvtsi64_si128((long long)a);
    __m128i b_t = _mm_gf2p8affine_epi64_epi8(I, b_x, 0);
    __m128i c_cols = _mm_gf2p8affine_epi64_epi8(b_t, a_x, 0);
    __m128i c_rows = _mm_gf2p8affine_epi64_epi8(I, c_cols, 0);
    return (uint64_t)_mm_cvtsi128_si64(c_rows);
}

static inline void gpt32_m8_make_lower_unit_pair_u64(uint64_t *L, uint64_t *Linv) {
    const uint64_t r = gpt32_random64();
    const uint64_t I_val = 0x0102040810204080ULL;

    const uint64_t m1_val = I_val
        ^ ((r & 0xFULL) << 36)
        ^ (((r >> 4) & 0xFULL) << 44)
        ^ (((r >> 8) & 0xFULL) << 52)
        ^ (((r >> 12) & 0xFULL) << 60);
    const uint64_t m2_val = I_val
        ^ (((r >> 16) & 0x3ULL) << 22)
        ^ (((r >> 18) & 0x3ULL) << 30)
        ^ (((r >> 20) & 0x3ULL) << 50)
        ^ (((r >> 22) & 0x3ULL) << 58);
    const uint64_t m3_val = I_val
        ^ (((r >> 24) & 1ULL) << 15)
        ^ (((r >> 25) & 1ULL) << 29)
        ^ (((r >> 26) & 1ULL) << 43)
        ^ (((r >> 27) & 1ULL) << 57);

    *L = gpt32_m8_mul_u64(gpt32_m8_mul_u64(m1_val, m2_val), m3_val);
    *Linv = gpt32_m8_mul_u64(gpt32_m8_mul_u64(m3_val, m2_val), m1_val);
}

static inline void gpt32_m8_make_upper_unit_pair_u64(uint64_t *U, uint64_t *Uinv) {
    const uint64_t r = gpt32_random64();
    const uint64_t I_val = 0x0102040810204080ULL;

    const uint64_t n1_val = I_val
        ^ (r & 0xFULL)
        ^ (((r >> 4) & 0xFULL) << 8)
        ^ (((r >> 8) & 0xFULL) << 16)
        ^ (((r >> 12) & 0xFULL) << 24);
    const uint64_t n2_val = I_val
        ^ (((r >> 16) & 0x3ULL) << 4)
        ^ (((r >> 18) & 0x3ULL) << 12)
        ^ (((r >> 20) & 0x3ULL) << 32)
        ^ (((r >> 22) & 0x3ULL) << 40);
    const uint64_t n3_val = I_val
        ^ (((r >> 24) & 1ULL) << 6)
        ^ (((r >> 25) & 1ULL) << 20)
        ^ (((r >> 26) & 1ULL) << 34)
        ^ (((r >> 27) & 1ULL) << 48);

    *U = gpt32_m8_mul_u64(gpt32_m8_mul_u64(n1_val, n2_val), n3_val);
    *Uinv = gpt32_m8_mul_u64(gpt32_m8_mul_u64(n3_val, n2_val), n1_val);
}

static inline M16RegPair_GPT gpt32_pack_m8_blocks_to_m16(uint64_t lt, uint64_t rt,
                                                         uint64_t lb, uint64_t rb) {
    M16RegPair_GPT out;
    __m128i lt_x = _mm_cvtsi64_si128((long long)lt);
    __m128i rt_x = _mm_cvtsi64_si128((long long)rt);
    __m128i lb_x = _mm_cvtsi64_si128((long long)lb);
    __m128i rb_x = _mm_cvtsi64_si128((long long)rb);
    out.top = _mm_unpacklo_epi8(rt_x, lt_x);
    out.bot = _mm_unpacklo_epi8(rb_x, lb_x);
    return out;
}

static inline void gpt32_make_m16_lower_unit_pair(M16RegPair_GPT *L,
                                                  M16RegPair_GPT *Linv) {
    uint64_t La, La_inv, Lb, Lb_inv;
    gpt32_m8_make_lower_unit_pair_u64(&La, &La_inv);
    gpt32_m8_make_lower_unit_pair_u64(&Lb, &Lb_inv);

    uint64_t X = gpt32_random64();
    uint64_t XLa = gpt32_m8_mul_u64(X, La);
    uint64_t LbInvX = gpt32_m8_mul_u64(Lb_inv, X);

    *L = gpt32_pack_m8_blocks_to_m16(La, 0, XLa, Lb);
    *Linv = gpt32_pack_m8_blocks_to_m16(La_inv, 0, LbInvX, Lb_inv);
}

static inline void gpt32_make_m16_upper_unit_pair(M16RegPair_GPT *U,
                                                  M16RegPair_GPT *Uinv) {
    uint64_t Ua, Ua_inv, Ub, Ub_inv;
    gpt32_m8_make_upper_unit_pair_u64(&Ua, &Ua_inv);
    gpt32_m8_make_upper_unit_pair_u64(&Ub, &Ub_inv);

    uint64_t Y = gpt32_random64();
    uint64_t YUb = gpt32_m8_mul_u64(Y, Ub);
    uint64_t UaInvY = gpt32_m8_mul_u64(Ua_inv, Y);

    *U = gpt32_pack_m8_blocks_to_m16(Ua, YUb, 0, Ub);
    *Uinv = gpt32_pack_m8_blocks_to_m16(Ua_inv, UaInvY, 0, Ub_inv);
}

static void gpt32_radix2_unitriangular_plu_recursive_twoleft_tworight_core(
    int rowdirect_pinv, int bitstream_p, M32 *mat, M32 *mat_inv) {
    M16RegPair_GPT La, La_inv, Lb, Lb_inv;
    M16RegPair_GPT Ua, Ua_inv, Ub, Ub_inv;

    gpt32_make_m16_lower_unit_pair(&La, &La_inv);
    gpt32_make_m16_lower_unit_pair(&Lb, &Lb_inv);
    gpt32_make_m16_upper_unit_pair(&Ua, &Ua_inv);
    gpt32_make_m16_upper_unit_pair(&Ub, &Ub_inv);

    M16RegPair_GPT zero = gpt32_zero_m16_reg();
    M16RegPair_GPT X = gpt32_rand_m16_reg();
    M16RegPair_GPT Y = gpt32_rand_m16_reg();

    M16RegPair_GPT XLa = gpt32_mul_m16(X, La);
    M16RegPair_GPT YUb = gpt32_mul_m16(Y, Ub);
    M16RegPair_GPT LbUb = gpt32_mul_m16(Lb, Ub);

    M16RegPair_GPT base_lt, base_rt, base_lb, base_rb;
    gpt32_muladd_two_left_two_right_m16(La, XLa, Ua, YUb,
                                        zero, zero, zero, LbUb,
                                        &base_lt, &base_rt, &base_lb, &base_rb);

    M16RegPair_GPT UaInvY = gpt32_mul_m16(Ua_inv, Y);
    M16RegPair_GPT LbInvX = gpt32_mul_m16(Lb_inv, X);
    M16RegPair_GPT UaInvLaInv = gpt32_mul_m16(Ua_inv, La_inv);

    M16RegPair_GPT inv_lt, inv_rt, inv_lb, inv_rb;
    gpt32_muladd_two_left_two_right_m16(UaInvY, Ub_inv, LbInvX, Lb_inv,
                                        UaInvLaInv, zero, zero, zero,
                                        &inv_lt, &inv_rt, &inv_lb, &inv_rb);

    M32 base, base_inv;
    gpt32_pack_m16_blocks(base_lt, base_rt, base_lb, base_rb, &base);
    gpt32_pack_m16_blocks(inv_lt, inv_rt, inv_lb, inv_rb, &base_inv);

    if (bitstream_p) {
        gpt32_apply_dynamic_p_matmul_bitstream(&base, &base_inv, rowdirect_pinv,
                                               mat, mat_inv);
    } else if (rowdirect_pinv) {
        gpt32_apply_dynamic_p_matmul_rowdirect(&base, &base_inv, mat, mat_inv);
    } else {
        gpt32_apply_dynamic_p_matmul(&base, &base_inv, mat, mat_inv);
    }
}

void wbmatrix2_genmatpair_m32_hbup(
    M32 *Mat, M32 *Mat_inv) {
    gpt32_radix2_unitriangular_plu_recursive_twoleft_tworight_core(1, 0, Mat, Mat_inv);
}

static inline uint32_t m4_gpt1_gen_pack_rows(uint8_t r0, uint8_t r1, uint8_t r2, uint8_t r3)
{
    return ((uint32_t)(r0 & 0x0fU)) |
           ((uint32_t)(r1 & 0x0fU) << 8) |
           ((uint32_t)(r2 & 0x0fU) << 16) |
           ((uint32_t)(r3 & 0x0fU) << 24);
}

static inline uint8_t m4_gpt1_gen_bit(uint32_t r, unsigned int bit)
{
    return (uint8_t)((r >> bit) & 1U);
}

static inline uint32_t m4_gpt7_gen_swap_bytes_down(uint32_t rows, uint32_t high, uint32_t low)
{
    const uint32_t shift = 8U * (high - low);
    const uint32_t mask = 0xffU << (8U * low);
    const uint32_t x = ((rows >> shift) ^ rows) & mask;
    return rows ^ x ^ (x << shift);
}

static inline uint32_t m4_gpt7_gen_swap_cols_down(uint32_t rows, uint32_t high, uint32_t low)
{
    const uint32_t shift = high - low;
    const uint32_t mask = 0x08080808U >> high;
    const uint32_t x = ((rows >> shift) ^ rows) & mask;
    return rows ^ x ^ (x << shift);
}

static inline void m4_gpt7_gen_store_rows_nomask(uint32_t rows, M4 *Mat)
{
    memcpy(Mat->M, &rows, sizeof(rows));
}

static inline uint32_t m4_gpt10_gen_fast_j2_nocmp(uint32_t j2)
{
    return j2 - ((j2 >> 1) & j2);
}

static inline void m4_gpt12_gen_dynamic_p_apply_interleaved_swaps_nocmp_from_bits(uint32_t r,
                                                                                  uint32_t base_rows,
                                                                                  uint32_t base_inv_rows,
                                                                                  uint32_t *mat_rows,
                                                                                  uint32_t *mat_inv_rows)
{
    const uint32_t j3 = r & 3U;
    r >>= 2;

    uint32_t j2 = r & 3U;
    j2 = m4_gpt10_gen_fast_j2_nocmp(j2);
    r >>= 2;

    const uint32_t j1 = r & 1U;

    uint32_t rows = base_rows;
    uint32_t inv_rows = base_inv_rows;

    rows = m4_gpt7_gen_swap_bytes_down(rows, 3U, j3);
    inv_rows = m4_gpt7_gen_swap_cols_down(inv_rows, 3U, j3);
    rows = m4_gpt7_gen_swap_bytes_down(rows, 2U, j2);
    inv_rows = m4_gpt7_gen_swap_cols_down(inv_rows, 2U, j2);
    rows = m4_gpt7_gen_swap_bytes_down(rows, 1U, j1);
    inv_rows = m4_gpt7_gen_swap_cols_down(inv_rows, 1U, j1);

    *mat_rows = rows;
    *mat_inv_rows = inv_rows;
}

static inline uint32_t m4_gpt11_permuteQPR_inline(uint32_t x)
{
    const uint32_t prime = 4294967291u;
    if (x >= prime) {
        return x;
    }
    const uint32_t residue = (uint32_t)(((uint64_t)x * (uint64_t)x) % prime);
    return (x <= prime / 2U) ? residue : prime - residue;
}

static inline uint32_t m4_gpt11_cus_random_inline(void)
{
    return m4_gpt11_permuteQPR_inline(
        (m4_gpt11_permuteQPR_inline(m_index++) + m_intermediateOffset) ^ 0x5bf03635U);
}

static inline uint8_t m4_gpt4_gen_select_row(uint8_t row, uint8_t bit)
{
    return (uint8_t)(row & (uint8_t)(0U - bit));
}

static inline void m4_gpt4_gen_radix2_base_full_direct_from_bits(uint32_t r, uint32_t *base_rows, uint32_t *base_inv_rows)
{
    const uint8_t a00 = m4_gpt1_gen_bit(r, 0);
    const uint8_t a01 = m4_gpt1_gen_bit(r, 1);
    const uint8_t a10 = m4_gpt1_gen_bit(r, 2);
    const uint8_t a11 = m4_gpt1_gen_bit(r, 3);
    const uint8_t b0 = m4_gpt1_gen_bit(r, 4);
    const uint8_t b1 = m4_gpt1_gen_bit(r, 5);
    const uint8_t x00 = m4_gpt1_gen_bit(r, 6);
    const uint8_t x01 = m4_gpt1_gen_bit(r, 7);
    const uint8_t x10 = m4_gpt1_gen_bit(r, 8);
    const uint8_t x11 = m4_gpt1_gen_bit(r, 9);
    const uint8_t y0 = m4_gpt1_gen_bit(r, 10);
    const uint8_t y1 = m4_gpt1_gen_bit(r, 11);

    const uint8_t l10 = b0;
    const uint8_t l20 = (uint8_t)(a00 ^ (a01 & b0));
    const uint8_t l21 = a01;
    const uint8_t l30 = (uint8_t)(a10 ^ (a11 & b0));
    const uint8_t l31 = a11;
    const uint8_t l32 = b1;

    const uint8_t u01 = y0;
    const uint8_t u02 = x00;
    const uint8_t u03 = (uint8_t)(x01 ^ (x00 & y1));
    const uint8_t u12 = x10;
    const uint8_t u13 = (uint8_t)(x11 ^ (x10 & y1));
    const uint8_t u23 = y1;

    const uint8_t U0 = (uint8_t)(0x08U | (u01 << 2) | (u02 << 1) | u03);
    const uint8_t U1 = (uint8_t)(0x04U | (u12 << 1) | u13);
    const uint8_t U2 = (uint8_t)(0x02U | u23);
    const uint8_t U3 = 0x01U;

    const uint8_t B0 = U0;
    const uint8_t B1 = (uint8_t)(m4_gpt4_gen_select_row(U0, l10) ^ U1);
    const uint8_t B2 = (uint8_t)(m4_gpt4_gen_select_row(U0, l20) ^
                                  m4_gpt4_gen_select_row(U1, l21) ^ U2);
    const uint8_t B3 = (uint8_t)(m4_gpt4_gen_select_row(U0, l30) ^
                                  m4_gpt4_gen_select_row(U1, l31) ^
                                  m4_gpt4_gen_select_row(U2, l32) ^ U3);
    *base_rows = m4_gpt1_gen_pack_rows(B0, B1, B2, B3);

    const uint8_t li0 = 0x08U;
    const uint8_t li1 = (uint8_t)(0x04U | (b0 << 3));
    const uint8_t li2 = (uint8_t)(0x02U | (a01 << 2) | (a00 << 3));
    const uint8_t li3 = (uint8_t)(0x01U | (b1 << 1) |
                                  ((a11 ^ (b1 & a01)) << 2) |
                                  ((a10 ^ (b1 & a00)) << 3));

    const uint8_t ui01 = y0;
    const uint8_t ui02 = (uint8_t)(x00 ^ (y0 & x10));
    const uint8_t ui03 = (uint8_t)(x01 ^ (y0 & x11));
    const uint8_t ui12 = x10;
    const uint8_t ui13 = x11;
    const uint8_t ui23 = y1;

    const uint8_t BI0 = (uint8_t)(li0 ^ m4_gpt4_gen_select_row(li1, ui01) ^
                                   m4_gpt4_gen_select_row(li2, ui02) ^
                                   m4_gpt4_gen_select_row(li3, ui03));
    const uint8_t BI1 = (uint8_t)(li1 ^ m4_gpt4_gen_select_row(li2, ui12) ^
                                   m4_gpt4_gen_select_row(li3, ui13));
    const uint8_t BI2 = (uint8_t)(li2 ^ m4_gpt4_gen_select_row(li3, ui23));
    const uint8_t BI3 = li3;
    *base_inv_rows = m4_gpt1_gen_pack_rows(BI0, BI1, BI2, BI3);
}

void wbmatrix2_genmatpair_m4_hbup(M4 *Mat, M4 *Mat_inv)
{
    const uint32_t r = m4_gpt11_cus_random_inline();
    uint32_t base_rows;
    uint32_t base_inv_rows;
    uint32_t mat_rows;
    uint32_t mat_inv_rows;

    m4_gpt4_gen_radix2_base_full_direct_from_bits(r, &base_rows, &base_inv_rows);
    m4_gpt12_gen_dynamic_p_apply_interleaved_swaps_nocmp_from_bits(r >> 12, base_rows, base_inv_rows,
                                                                   &mat_rows, &mat_inv_rows);
    m4_gpt7_gen_store_rows_nomask(mat_rows, Mat);
    m4_gpt7_gen_store_rows_nomask(mat_inv_rows, Mat_inv);
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

static void MatMulMatM64_GPT17_m32pairbt_rawaccum_aligned_total(
    const M64 *Mat1, const M64 *Mat2, M64 *Mat)
{
    M64PreparedB_M32PairBT_GPT PreparedB __attribute__((aligned(64)));

    MatMulMatM64_GPT12_prepareB_m32pairbt(Mat2, &PreparedB);
    MatMulMatM64_GPT14_preparedB_m32pairbt_rawaccum(Mat1, &PreparedB, Mat);
}

static void MatMulMatM64_GPT19_m32pairbt_rawaccum_streamA_dualprepare_total(
    const M64 *Mat1, const M64 *Mat2, M64 *Mat)
{
    M64PreparedB_M32PairBT_GPT PreparedB __attribute__((aligned(64)));

    prepare_m32pairbt_from_m64_dualhalf_gpt18(Mat2, &PreparedB);
    MatMulMatM64_GPT15_preparedB_m32pairbt_rawaccum_streamA(Mat1, &PreparedB, Mat);
}

static inline void xorM32_gpt(const M32 *a, const M32 *b, M32 *out) {
    for (int i = 0; i < 4; i++) {
        __m256i va = _mm256_loadu_si256((const __m256i *)&a->M[i * 8]);
        __m256i vb = _mm256_loadu_si256((const __m256i *)&b->M[i * 8]);
        _mm256_storeu_si256((__m256i *)&out->M[i * 8], _mm256_xor_si256(va, vb));
    }
}

static inline void pack_M32_to_M64_gpt(
    const M32 *top_left, const M32 *top_right,
    const M32 *bot_left, const M32 *bot_right,
    M64 *out)
{
    for (int i = 0; i < 32; i++) {
        out->M[i] = ((uint64_t)top_left->M[i] << 32) | top_right->M[i];
        out->M[i + 32] = ((uint64_t)bot_left->M[i] << 32) | bot_right->M[i];
    }
}

static inline uint32_t gpt64_bounded_rand(uint32_t bound) {
    return (uint32_t)(((uint64_t)cus_random() * (uint64_t)bound) >> 32);
}

static void gpt64_make_perm(uint8_t p[64]) {
    for (int i = 0; i < 64; i++) {
        p[i] = (uint8_t)i;
    }

    for (int i = 63; i > 0; i--) {
        int j = (int)gpt64_bounded_rand((uint32_t)(i + 1));
        uint8_t tmp = p[i];
        p[i] = p[j];
        p[j] = tmp;
    }
}

static void gpt64_apply_perm_rows_only(const M64 *base, const uint8_t p[64], M64 *mat) {
    for (int i = 0; i < 64; i++) {
        mat->M[i] = base->M[p[i]];
    }
}

static void gpt64_build_pinv_dense_from_perm(const uint8_t p[64], M64 *pinv) {
    for (int i = 0; i < 64; i++) {
        pinv->M[i] = 0;
    }
    for (int i = 0; i < 64; i++) {
        pinv->M[p[i]] = 1ULL << (63 - i);
    }
}

static void gpt64_apply_dynamic_p_matmul_with(
    const M64 *base,
    const M64 *base_inv,
    void (*pinv_mul)(const M64 *, const M64 *, M64 *),
    void (*make_perm)(uint8_t p[64]),
    M64 *mat,
    M64 *mat_inv)
{
    uint8_t p[64];
    M64 pinv;

    make_perm(p);
    gpt64_apply_perm_rows_only(base, p, mat);
    gpt64_build_pinv_dense_from_perm(p, &pinv);
    pinv_mul(base_inv, &pinv, mat_inv);
}

static inline uint32_t gpt64_take_bits(Gpt64Bitstream *stream, unsigned int n) {
    while (stream->bits < n) {
        stream->pool |= (uint64_t)cus_random() << stream->bits;
        stream->bits += 32U;
    }

    const uint64_t mask = (n == 32U) ? 0xffffffffULL : ((1ULL << n) - 1ULL);
    uint32_t out = (uint32_t)(stream->pool & mask);
    stream->pool >>= n;
    stream->bits -= n;
    return out;
}

static inline uint32_t gpt64_bounded_rand_bitstream(Gpt64Bitstream *stream,
                                                    uint32_t bound) {
    uint32_t x = bound - 1u;
    unsigned int bits = 0;
    do {
        bits++;
        x >>= 1;
    } while (x != 0);

    uint32_t raw = gpt64_take_bits(stream, bits);
    return (uint32_t)(((uint64_t)raw * (uint64_t)bound) >> bits);
}

static void gpt64_make_perm_bitstream(uint8_t p[64]) {
    for (int i = 0; i < 64; i++) {
        p[i] = (uint8_t)i;
    }

    Gpt64Bitstream stream = {0, 0};
    for (int i = 63; i > 0; i--) {
        int j = (int)gpt64_bounded_rand_bitstream(&stream, (uint32_t)(i + 1));
        uint8_t tmp = p[i];
        p[i] = p[j];
        p[j] = tmp;
    }
}

static inline void gpt64_apply_perm_rows_and_pinv_nozero(
    const M64 *base,
    const uint8_t p[64],
    M64 *mat,
    M64 *pinv)
{
    for (int i = 0; i < 64; i++) {
        const uint8_t pi = p[i];
        mat->M[i] = base->M[pi];
        pinv->M[pi] = 1ULL << (63 - i);
    }
}

static inline void gpt64_apply_dynamic_p_fastp_gpt14(
    const M64 *base,
    const M64 *base_inv,
    void (*make_perm)(uint8_t p[64]),
    M64 *mat,
    M64 *mat_inv)
{
    uint8_t p[64];
    M64 pinv;

    make_perm(p);
    gpt64_apply_perm_rows_and_pinv_nozero(base, p, mat, &pinv);
    MatMulMatM64_GPT14_m32pairbt_rawaccum_total(base_inv, &pinv, mat_inv);
}

static inline void gpt64_apply_dynamic_p_fastp_gpt17(
    const M64 *base,
    const M64 *base_inv,
    void (*make_perm)(uint8_t p[64]),
    M64 *mat,
    M64 *mat_inv)
{
    uint8_t p[64];
    M64 pinv;

    make_perm(p);
    gpt64_apply_perm_rows_and_pinv_nozero(base, p, mat, &pinv);
    MatMulMatM64_GPT17_m32pairbt_rawaccum_aligned_total(base_inv, &pinv, mat_inv);
}

static inline void gpt64_apply_dynamic_p_fastp_gpt19(
    const M64 *base,
    const M64 *base_inv,
    void (*make_perm)(uint8_t p[64]),
    M64 *mat,
    M64 *mat_inv)
{
    uint8_t p[64];
    M64 pinv;

    make_perm(p);
    gpt64_apply_perm_rows_and_pinv_nozero(base, p, mat, &pinv);
    MatMulMatM64_GPT19_m32pairbt_rawaccum_streamA_dualprepare_total(
        base_inv, &pinv, mat_inv);
}

static inline uint64_t gpt64_random64(void) {
    uint64_t high = (uint64_t)cus_random();
    uint64_t low = (uint64_t)cus_random();
    return (high << 32) | low;
}

static inline uint64_t gpt64_m8_mul_u64(uint64_t a, uint64_t b) {
    const uint64_t I_val = 0x0102040810204080ULL;
    __m128i I = _mm_set1_epi64x((long long)I_val);
    __m128i b_x = _mm_cvtsi64_si128((long long)b);
    __m128i a_x = _mm_cvtsi64_si128((long long)a);
    __m128i b_t = _mm_gf2p8affine_epi64_epi8(I, b_x, 0);
    __m128i c_cols = _mm_gf2p8affine_epi64_epi8(b_t, a_x, 0);
    __m128i c_rows = _mm_gf2p8affine_epi64_epi8(I, c_cols, 0);
    return (uint64_t)_mm_cvtsi128_si64(c_rows);
}

static inline void gpt64_m8_make_lower_unit_pair_u64(uint64_t *L, uint64_t *Linv) {
    const uint64_t r = gpt64_random64();
    const uint64_t I_val = 0x0102040810204080ULL;

    const uint64_t m1_val = I_val
        ^ ((r & 0xFULL) << 36)
        ^ (((r >> 4) & 0xFULL) << 44)
        ^ (((r >> 8) & 0xFULL) << 52)
        ^ (((r >> 12) & 0xFULL) << 60);
    const uint64_t m2_val = I_val
        ^ (((r >> 16) & 0x3ULL) << 22)
        ^ (((r >> 18) & 0x3ULL) << 30)
        ^ (((r >> 20) & 0x3ULL) << 50)
        ^ (((r >> 22) & 0x3ULL) << 58);
    const uint64_t m3_val = I_val
        ^ (((r >> 24) & 1ULL) << 15)
        ^ (((r >> 25) & 1ULL) << 29)
        ^ (((r >> 26) & 1ULL) << 43)
        ^ (((r >> 27) & 1ULL) << 57);

    *L = gpt64_m8_mul_u64(gpt64_m8_mul_u64(m1_val, m2_val), m3_val);
    *Linv = gpt64_m8_mul_u64(gpt64_m8_mul_u64(m3_val, m2_val), m1_val);
}

static inline void gpt64_m8_make_upper_unit_pair_u64(uint64_t *U, uint64_t *Uinv) {
    const uint64_t r = gpt64_random64();
    const uint64_t I_val = 0x0102040810204080ULL;

    const uint64_t n1_val = I_val
        ^ (r & 0xFULL)
        ^ (((r >> 4) & 0xFULL) << 8)
        ^ (((r >> 8) & 0xFULL) << 16)
        ^ (((r >> 12) & 0xFULL) << 24);
    const uint64_t n2_val = I_val
        ^ (((r >> 16) & 0x3ULL) << 4)
        ^ (((r >> 18) & 0x3ULL) << 12)
        ^ (((r >> 20) & 0x3ULL) << 32)
        ^ (((r >> 22) & 0x3ULL) << 40);
    const uint64_t n3_val = I_val
        ^ (((r >> 24) & 1ULL) << 6)
        ^ (((r >> 25) & 1ULL) << 20)
        ^ (((r >> 26) & 1ULL) << 34)
        ^ (((r >> 27) & 1ULL) << 48);

    *U = gpt64_m8_mul_u64(gpt64_m8_mul_u64(n1_val, n2_val), n3_val);
    *Uinv = gpt64_m8_mul_u64(gpt64_m8_mul_u64(n3_val, n2_val), n1_val);
}

static inline Gpt64M16RegPair gpt64_pack_m8_blocks_to_m16(uint64_t lt, uint64_t rt,
                                                          uint64_t lb, uint64_t rb) {
    Gpt64M16RegPair out;
    __m128i lt_x = _mm_cvtsi64_si128((long long)lt);
    __m128i rt_x = _mm_cvtsi64_si128((long long)rt);
    __m128i lb_x = _mm_cvtsi64_si128((long long)lb);
    __m128i rb_x = _mm_cvtsi64_si128((long long)rb);
    out.top = _mm_unpacklo_epi8(rt_x, lt_x);
    out.bot = _mm_unpacklo_epi8(rb_x, lb_x);
    return out;
}

static inline Gpt64M16RegPair gpt64_zero_m16_reg_pair(void) {
    Gpt64M16RegPair out;
    out.top = _mm_setzero_si128();
    out.bot = _mm_setzero_si128();
    return out;
}

static inline Gpt64M16RegPair gpt64_rand_m16_reg_pair(void) {
    Gpt64M16RegPair out;
    out.top = _mm_set_epi64x((long long)gpt64_random64(), (long long)gpt64_random64());
    out.bot = _mm_set_epi64x((long long)gpt64_random64(), (long long)gpt64_random64());
    return out;
}

static inline Gpt64M16RegPair gpt64_mul_m16_reg_pair(Gpt64M16RegPair a,
                                                     Gpt64M16RegPair b) {
    Gpt64M16RegPair out;
    MatMulMatM16_reg(a.top, a.bot, b.top, b.bot, &out.top, &out.bot);
    return out;
}

static inline void gpt64_store_m32_row8(uint32_t *dst, Gpt64M16RegPair left,
                                        Gpt64M16RegPair right, int half) {
    __m128i l = half == 0 ? left.top : left.bot;
    __m128i r = half == 0 ? right.top : right.bot;
    __m128i lo = _mm_unpacklo_epi16(r, l);
    __m128i hi = _mm_unpackhi_epi16(r, l);
    _mm_storeu_si128((__m128i *)dst, lo);
    _mm_storeu_si128((__m128i *)(dst + 4), hi);
}

static inline void gpt64_pack_m16_blocks_to_m32(Gpt64M16RegPair lt,
                                                Gpt64M16RegPair rt,
                                                Gpt64M16RegPair lb,
                                                Gpt64M16RegPair rb,
                                                M32 *out) {
    gpt64_store_m32_row8(out->M, lt, rt, 0);
    gpt64_store_m32_row8(out->M + 8, lt, rt, 1);
    gpt64_store_m32_row8(out->M + 16, lb, rb, 0);
    gpt64_store_m32_row8(out->M + 24, lb, rb, 1);
}

static void gpt64_m32_make_lower_unit_pair_m16reg(M32 *L, M32 *Linv) {
    uint64_t La8, La8_inv, Lb8, Lb8_inv;
    uint64_t Lc8, Lc8_inv, Ld8, Ld8_inv;
    gpt64_m8_make_lower_unit_pair_u64(&La8, &La8_inv);
    gpt64_m8_make_lower_unit_pair_u64(&Lb8, &Lb8_inv);
    gpt64_m8_make_lower_unit_pair_u64(&Lc8, &Lc8_inv);
    gpt64_m8_make_lower_unit_pair_u64(&Ld8, &Ld8_inv);

    uint64_t X0 = gpt64_random64();
    uint64_t X0La = gpt64_m8_mul_u64(X0, La8);
    uint64_t LbInvX0 = gpt64_m8_mul_u64(Lb8_inv, X0);
    Gpt64M16RegPair La = gpt64_pack_m8_blocks_to_m16(La8, 0, X0La, Lb8);
    Gpt64M16RegPair La_inv = gpt64_pack_m8_blocks_to_m16(La8_inv, 0, LbInvX0, Lb8_inv);

    uint64_t X1 = gpt64_random64();
    uint64_t X1Lc = gpt64_m8_mul_u64(X1, Lc8);
    uint64_t LdInvX1 = gpt64_m8_mul_u64(Ld8_inv, X1);
    Gpt64M16RegPair Lb = gpt64_pack_m8_blocks_to_m16(Lc8, 0, X1Lc, Ld8);
    Gpt64M16RegPair Lb_inv = gpt64_pack_m8_blocks_to_m16(Lc8_inv, 0, LdInvX1, Ld8_inv);

    Gpt64M16RegPair zero = gpt64_zero_m16_reg_pair();
    Gpt64M16RegPair X = gpt64_rand_m16_reg_pair();
    Gpt64M16RegPair XLa = gpt64_mul_m16_reg_pair(X, La);
    Gpt64M16RegPair LbInvX = gpt64_mul_m16_reg_pair(Lb_inv, X);

    gpt64_pack_m16_blocks_to_m32(La, zero, XLa, Lb, L);
    gpt64_pack_m16_blocks_to_m32(La_inv, zero, LbInvX, Lb_inv, Linv);
}

static void gpt64_m32_make_upper_unit_pair_m16reg(M32 *U, M32 *Uinv) {
    uint64_t Ua8, Ua8_inv, Ub8, Ub8_inv;
    uint64_t Uc8, Uc8_inv, Ud8, Ud8_inv;
    gpt64_m8_make_upper_unit_pair_u64(&Ua8, &Ua8_inv);
    gpt64_m8_make_upper_unit_pair_u64(&Ub8, &Ub8_inv);
    gpt64_m8_make_upper_unit_pair_u64(&Uc8, &Uc8_inv);
    gpt64_m8_make_upper_unit_pair_u64(&Ud8, &Ud8_inv);

    uint64_t Y0 = gpt64_random64();
    uint64_t Y0Ub = gpt64_m8_mul_u64(Y0, Ub8);
    uint64_t UaInvY0 = gpt64_m8_mul_u64(Ua8_inv, Y0);
    Gpt64M16RegPair Ua = gpt64_pack_m8_blocks_to_m16(Ua8, Y0Ub, 0, Ub8);
    Gpt64M16RegPair Ua_inv = gpt64_pack_m8_blocks_to_m16(Ua8_inv, UaInvY0, 0, Ub8_inv);

    uint64_t Y1 = gpt64_random64();
    uint64_t Y1Ud = gpt64_m8_mul_u64(Y1, Ud8);
    uint64_t UcInvY1 = gpt64_m8_mul_u64(Uc8_inv, Y1);
    Gpt64M16RegPair Ub = gpt64_pack_m8_blocks_to_m16(Uc8, Y1Ud, 0, Ud8);
    Gpt64M16RegPair Ub_inv = gpt64_pack_m8_blocks_to_m16(Uc8_inv, UcInvY1, 0, Ud8_inv);

    Gpt64M16RegPair zero = gpt64_zero_m16_reg_pair();
    Gpt64M16RegPair Y = gpt64_rand_m16_reg_pair();
    Gpt64M16RegPair YUb = gpt64_mul_m16_reg_pair(Y, Ub);
    Gpt64M16RegPair UaInvY = gpt64_mul_m16_reg_pair(Ua_inv, Y);

    gpt64_pack_m16_blocks_to_m32(Ua, YUb, zero, Ub, U);
    gpt64_pack_m16_blocks_to_m32(Ua_inv, UaInvY, zero, Ub_inv, Uinv);
}

static inline void gpt64_mulM32_mode(const M32 *a, const M32 *b, int rowdirect,
                                     M32 *out) {
    if (rowdirect) {
        MatMulMatM32_GPT4_rowdirect_ptr(a, b, out);
    } else {
        MatMulMatM32_GPT1_ptr(a, b, out);
    }
}

static inline void gpt64_muladdM32_mode(const M32 *a, const M32 *b, const M32 *add,
                                        int rowdirect, M32 *out) {
    M32 prod;
    gpt64_mulM32_mode(a, b, rowdirect, &prod);
    xorM32_gpt(&prod, add, out);
}

static inline void zeroM32_gpt(M32 *out) {
    for (int i = 0; i < 4; i++) {
        _mm256_storeu_si256((__m256i *)&out->M[i * 8], _mm256_setzero_si256());
    }
}

static inline void gpt64_prepared_muladd_m32(const M32 *left,
                                             const M32PreparedB_GPT *prepared_right,
                                             const M32 *add,
                                             int rowdirect_prepared,
                                             M32 *out) {
    M32 prod;
    if (rowdirect_prepared) {
        MatMulMatM32_GPT4_preparedB_rowdirect(left, prepared_right, &prod);
    } else {
        MatMulMatM32_GPT2_preparedB(left, prepared_right, &prod);
    }
    xorM32_gpt(&prod, add, out);
}

static void gpt64_muladd_two_left_two_right_m32(
    const M32 *left0, const M32 *left1,
    const M32 *right0, const M32 *right1,
    const M32 *add00, const M32 *add01,
    const M32 *add10, const M32 *add11,
    int rowdirect_prepared,
    M32 *out00, M32 *out01,
    M32 *out10, M32 *out11)
{
    M32PreparedB_GPT prepared0, prepared1;
    MatMulMatM32_GPT2_prepareB(right0, &prepared0);
    MatMulMatM32_GPT2_prepareB(right1, &prepared1);

    gpt64_prepared_muladd_m32(left0, &prepared0, add00, rowdirect_prepared, out00);
    gpt64_prepared_muladd_m32(left0, &prepared1, add01, rowdirect_prepared, out01);
    gpt64_prepared_muladd_m32(left1, &prepared0, add10, rowdirect_prepared, out10);
    gpt64_prepared_muladd_m32(left1, &prepared1, add11, rowdirect_prepared, out11);
}

static void genMatpairM64_radix2_unitriangular_plu_recursive_m16units_gpt(
    int two_left_two_right,
    int rowdirect_prepared,
    int rowdirect_m32,
    int bitstream_p,
    int fastp_pinv,
    void (*pinv_mul)(const M64 *, const M64 *, M64 *),
    M64 *mat,
    M64 *mat_inv)
{
    M32 La, La_inv, Lb, Lb_inv;
    M32 Ua, Ua_inv, Ub, Ub_inv;
    M32 X, Y;
    M32 XLa, YUb, LbUb;
    M32 base_lt, base_rt, base_lb, base_rb;
    M32 UaInvY, LbInvX, UaInvLaInv;
    M32 inv_lt, inv_rt, inv_lb, inv_rb;
    M64 base, base_inv;

    gpt64_m32_make_lower_unit_pair_m16reg(&La, &La_inv);
    gpt64_m32_make_lower_unit_pair_m16reg(&Lb, &Lb_inv);
    gpt64_m32_make_upper_unit_pair_m16reg(&Ua, &Ua_inv);
    gpt64_m32_make_upper_unit_pair_m16reg(&Ub, &Ub_inv);
    wbmatrix2_rand_m32_no_reseed(&X);
    wbmatrix2_rand_m32_no_reseed(&Y);

    gpt64_mulM32_mode(&X, &La, rowdirect_m32, &XLa);
    gpt64_mulM32_mode(&Y, &Ub, rowdirect_m32, &YUb);
    gpt64_mulM32_mode(&Lb, &Ub, rowdirect_m32, &LbUb);

    if (two_left_two_right) {
        M32 zero;
        zeroM32_gpt(&zero);
        gpt64_muladd_two_left_two_right_m32(
            &La, &XLa, &Ua, &YUb,
            &zero, &zero, &zero, &LbUb,
            rowdirect_prepared,
            &base_lt, &base_rt, &base_lb, &base_rb);
    } else {
        gpt64_mulM32_mode(&La, &Ua, rowdirect_m32, &base_lt);
        gpt64_mulM32_mode(&La, &YUb, rowdirect_m32, &base_rt);
        gpt64_mulM32_mode(&XLa, &Ua, rowdirect_m32, &base_lb);
        gpt64_muladdM32_mode(&XLa, &YUb, &LbUb, rowdirect_m32, &base_rb);
    }

    gpt64_mulM32_mode(&Ua_inv, &Y, rowdirect_m32, &UaInvY);
    gpt64_mulM32_mode(&Lb_inv, &X, rowdirect_m32, &LbInvX);
    gpt64_mulM32_mode(&Ua_inv, &La_inv, rowdirect_m32, &UaInvLaInv);

    if (two_left_two_right) {
        M32 zero;
        zeroM32_gpt(&zero);
        gpt64_muladd_two_left_two_right_m32(
            &UaInvY, &Ub_inv, &LbInvX, &Lb_inv,
            &UaInvLaInv, &zero, &zero, &zero,
            rowdirect_prepared,
            &inv_lt, &inv_rt, &inv_lb, &inv_rb);
    } else {
        gpt64_muladdM32_mode(&UaInvY, &LbInvX, &UaInvLaInv, rowdirect_m32, &inv_lt);
        gpt64_mulM32_mode(&UaInvY, &Lb_inv, rowdirect_m32, &inv_rt);
        gpt64_mulM32_mode(&Ub_inv, &LbInvX, rowdirect_m32, &inv_lb);
        gpt64_mulM32_mode(&Ub_inv, &Lb_inv, rowdirect_m32, &inv_rb);
    }

    pack_M32_to_M64_gpt(&base_lt, &base_rt, &base_lb, &base_rb, &base);
    pack_M32_to_M64_gpt(&inv_lt, &inv_rt, &inv_lb, &inv_rb, &base_inv);
    void (*make_perm)(uint8_t p[64]) =
        bitstream_p ? gpt64_make_perm_bitstream : gpt64_make_perm;
    if (fastp_pinv == 14) {
        gpt64_apply_dynamic_p_fastp_gpt14(&base, &base_inv, make_perm, mat, mat_inv);
    } else if (fastp_pinv == 17) {
        gpt64_apply_dynamic_p_fastp_gpt17(&base, &base_inv, make_perm, mat, mat_inv);
    } else if (fastp_pinv == 19) {
        gpt64_apply_dynamic_p_fastp_gpt19(&base, &base_inv, make_perm, mat, mat_inv);
    } else {
        gpt64_apply_dynamic_p_matmul_with(&base, &base_inv, pinv_mul,
                                          make_perm, mat, mat_inv);
    }
}

void wbmatrix2_genmatpair_m64_hbup(
    M64 *Mat, M64 *Mat_inv)
{
    genMatpairM64_radix2_unitriangular_plu_recursive_m16units_gpt(
        0, 0, 0, 0, 0,
        MatMulMatM64_GPT19_m32pairbt_rawaccum_streamA_dualprepare_total,
        Mat, Mat_inv);
}

static inline uint64_t gpt8_random64(void) {
    const uint64_t high = (uint64_t)cus_random();
    const uint64_t low = (uint64_t)cus_random();
    return (high << 32) | low;
}

static inline __m128i gpt8_make_dynamic_p_mask_fast_bitsliced_low8(void) {
    uint8_t p[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    uint32_t r = cus_random();

    const uint32_t j7 = r & 7U; r >>= 3;
    uint8_t t = p[7]; p[7] = p[j7]; p[j7] = t;

    uint32_t j6 = r & 7U; r >>= 3;
    j6 -= (uint32_t)(j6 == 7U);
    t = p[6]; p[6] = p[j6]; p[j6] = t;

    uint32_t j5 = r & 7U; r >>= 3;
    j5 -= ((uint32_t)(j5 >= 6U) << 1);
    t = p[5]; p[5] = p[j5]; p[j5] = t;

    uint32_t j4 = r & 7U; r >>= 3;
    j4 -= (uint32_t)(j4 >= 5U) * 5U;
    t = p[4]; p[4] = p[j4]; p[j4] = t;

    const uint32_t j3 = r & 3U; r >>= 2;
    t = p[3]; p[3] = p[j3]; p[j3] = t;

    uint32_t j2 = r & 3U; r >>= 2;
    j2 -= (uint32_t)(j2 == 3U);
    t = p[2]; p[2] = p[j2]; p[j2] = t;

    const uint32_t j1 = r & 1U;
    t = p[1]; p[1] = p[j1]; p[j1] = t;

    return _mm_loadl_epi64((const __m128i *)p);
}

static inline void gpt8_gen_plu_base_no_p_invcols_chain6_sharedI(
    __m128i *out_base,
    __m128i *out_base_inv_cols,
    __m128i I)
{
    const uint64_t r = gpt8_random64();

    const uint64_t m1_val = 0x0102040810204080ULL
        ^ ((r & 0xFULL) << 36)
        ^ (((r >> 4) & 0xFULL) << 44)
        ^ (((r >> 8) & 0xFULL) << 52)
        ^ (((r >> 12) & 0xFULL) << 60);
    const uint64_t m2_val = 0x0102040810204080ULL
        ^ (((r >> 16) & 0x3ULL) << 22)
        ^ (((r >> 18) & 0x3ULL) << 30)
        ^ (((r >> 20) & 0x3ULL) << 50)
        ^ (((r >> 22) & 0x3ULL) << 58);
    const uint64_t m3_val = 0x0102040810204080ULL
        ^ (((r >> 24) & 1ULL) << 15)
        ^ (((r >> 25) & 1ULL) << 29)
        ^ (((r >> 26) & 1ULL) << 43)
        ^ (((r >> 27) & 1ULL) << 57);

    const __m128i M1 = _mm_cvtsi64_si128((long long)m1_val);
    const __m128i M2 = _mm_cvtsi64_si128((long long)m2_val);
    const __m128i M3 = _mm_cvtsi64_si128((long long)m3_val);

    const uint64_t r_high = r >> 28;
    const uint64_t n1_val = 0x0102040810204080ULL
        ^ (r_high & 0xFULL)
        ^ (((r_high >> 4) & 0xFULL) << 8)
        ^ (((r_high >> 8) & 0xFULL) << 16)
        ^ (((r_high >> 12) & 0xFULL) << 24);
    const uint64_t n2_val = 0x0102040810204080ULL
        ^ (((r_high >> 16) & 0x3ULL) << 4)
        ^ (((r_high >> 18) & 0x3ULL) << 12)
        ^ (((r_high >> 20) & 0x3ULL) << 32)
        ^ (((r_high >> 22) & 0x3ULL) << 40);
    const uint64_t n3_val = 0x0102040810204080ULL
        ^ (((r_high >> 24) & 1ULL) << 6)
        ^ (((r_high >> 25) & 1ULL) << 20)
        ^ (((r_high >> 26) & 1ULL) << 34)
        ^ (((r_high >> 27) & 1ULL) << 48);

    const __m128i N1 = _mm_cvtsi64_si128((long long)n1_val);
    const __m128i N2 = _mm_cvtsi64_si128((long long)n2_val);
    const __m128i N3 = _mm_cvtsi64_si128((long long)n3_val);

    __m128i base_t = _mm_gf2p8affine_epi64_epi8(I, N3, 0);
    base_t = _mm_gf2p8affine_epi64_epi8(base_t, N2, 0);
    base_t = _mm_gf2p8affine_epi64_epi8(base_t, N1, 0);
    base_t = _mm_gf2p8affine_epi64_epi8(base_t, M3, 0);
    base_t = _mm_gf2p8affine_epi64_epi8(base_t, M2, 0);
    base_t = _mm_gf2p8affine_epi64_epi8(base_t, M1, 0);
    *out_base = _mm_gf2p8affine_epi64_epi8(I, base_t, 0);

    __m128i inv_cols = _mm_gf2p8affine_epi64_epi8(I, M1, 0);
    inv_cols = _mm_gf2p8affine_epi64_epi8(inv_cols, M2, 0);
    inv_cols = _mm_gf2p8affine_epi64_epi8(inv_cols, M3, 0);
    inv_cols = _mm_gf2p8affine_epi64_epi8(inv_cols, N1, 0);
    inv_cols = _mm_gf2p8affine_epi64_epi8(inv_cols, N2, 0);
    *out_base_inv_cols = _mm_gf2p8affine_epi64_epi8(inv_cols, N3, 0);
}

void wbmatrix2_genmatpair_m8_hbup(M8 *Mat, M8 *Mat_inv) {
    const uint64_t I_transpose = 0x0102040810204080ULL;
    const __m128i I = _mm_cvtsi64_si128((long long)I_transpose);
    __m128i mat_reg;
    __m128i inv_cols;
    gpt8_gen_plu_base_no_p_invcols_chain6_sharedI(&mat_reg, &inv_cols, I);

    const __m128i p_mask = gpt8_make_dynamic_p_mask_fast_bitsliced_low8();
    mat_reg = _mm_shuffle_epi8(mat_reg, p_mask);
    inv_cols = _mm_shuffle_epi8(inv_cols, p_mask);
    const __m128i inv_reg = _mm_gf2p8affine_epi64_epi8(I, inv_cols, 0);

    _mm_storel_epi64((__m128i *)Mat->M, mat_reg);
    _mm_storel_epi64((__m128i *)Mat_inv->M, inv_reg);
}

static inline void wbmatrix2_seed_public_genmatpair(void)
{
    InitRandom((randseed++) ^ (unsigned int)time(NULL));
}

#define WBMATRIX2_DEFINE_PUBLIC_GENMATPAIR(N, TYPE) \
    void genMatpairM##N(TYPE *mat, TYPE *mat_inv) \
    { \
        wbmatrix2_seed_public_genmatpair(); \
        wbmatrix2_genmatpair_m##N##_hbup(mat, mat_inv); \
    }

WBMATRIX2_DEFINE_PUBLIC_GENMATPAIR(4, M4)
WBMATRIX2_DEFINE_PUBLIC_GENMATPAIR(8, M8)
WBMATRIX2_DEFINE_PUBLIC_GENMATPAIR(16, M16)
WBMATRIX2_DEFINE_PUBLIC_GENMATPAIR(32, M32)
WBMATRIX2_DEFINE_PUBLIC_GENMATPAIR(64, M64)
WBMATRIX2_DEFINE_PUBLIC_GENMATPAIR(128, M128)
WBMATRIX2_DEFINE_PUBLIC_GENMATPAIR(256, M256)

#undef WBMATRIX2_DEFINE_PUBLIC_GENMATPAIR
