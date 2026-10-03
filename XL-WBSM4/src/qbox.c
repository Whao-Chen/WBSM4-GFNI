#include "wbsm4_specialized.h"

static inline __m256i xl_m32_prepare_block(__m256i rows)
{
    const __m256i identity = _mm256_set1_epi64x(0x0102040810204080ULL);
    const __m256i shuffle = _mm256_setr_epi8(
        0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
        0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    const __m256i permute = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

    rows = _mm256_shuffle_epi8(rows, shuffle);
    rows = _mm256_permutevar8x32_epi32(rows, permute);
    return _mm256_gf2p8affine_epi64_epi8(identity, rows, 0);
}

void xl_wbsm4_m32_prepare_right(
    const M32 *right, XLWBSM4M32PreparedRight *prepared)
{
    for (int block = 0; block < 4; ++block) {
        const __m256i rows =
            _mm256_loadu_si256((const __m256i *)&right->M[block * 8]);
        prepared->transposed_blocks[block] = xl_m32_prepare_block(rows);
    }
}

static inline __m256i xl_m32_multiply_block(
    __m256i left_rows, const XLWBSM4M32PreparedRight *prepared)
{
    const __m256i mask0 = _mm256_setr_epi8(
        3, 7, 11, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        3, 7, 11, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i mask1 = _mm256_setr_epi8(
        2, 6, 10, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        2, 6, 10, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i mask2 = _mm256_setr_epi8(
        1, 5, 9, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        1, 5, 9, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i mask3 = _mm256_setr_epi8(
        0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i broadcast = _mm256_setr_epi32(0, 4, 0, 4, 0, 4, 0, 4);
    const __m256i restore_permute =
        _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
    const __m256i restore_shuffle = _mm256_setr_epi8(
        0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
        0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);

    const __m256i a0 = _mm256_permutevar8x32_epi32(
        _mm256_shuffle_epi8(left_rows, mask0), broadcast);
    const __m256i a1 = _mm256_permutevar8x32_epi32(
        _mm256_shuffle_epi8(left_rows, mask1), broadcast);
    const __m256i a2 = _mm256_permutevar8x32_epi32(
        _mm256_shuffle_epi8(left_rows, mask2), broadcast);
    const __m256i a3 = _mm256_permutevar8x32_epi32(
        _mm256_shuffle_epi8(left_rows, mask3), broadcast);
    const __m256i t0 = _mm256_gf2p8affine_epi64_epi8(
        a0, prepared->transposed_blocks[0], 0);
    const __m256i t1 = _mm256_gf2p8affine_epi64_epi8(
        a1, prepared->transposed_blocks[1], 0);
    const __m256i t2 = _mm256_gf2p8affine_epi64_epi8(
        a2, prepared->transposed_blocks[2], 0);
    const __m256i t3 = _mm256_gf2p8affine_epi64_epi8(
        a3, prepared->transposed_blocks[3], 0);
    __m256i product = _mm256_xor_si256(
        _mm256_xor_si256(t0, t1), _mm256_xor_si256(t2, t3));

    product = _mm256_permutevar8x32_epi32(product, restore_permute);
    return _mm256_shuffle_epi8(product, restore_shuffle);
}

static inline void xl_m32_multiply_prepared_rows(
    const __m256i left_rows[4], const XLWBSM4M32PreparedRight *prepared,
    M32 *product)
{
    for (int block = 0; block < 4; ++block) {
        const __m256i rows = xl_m32_multiply_block(left_rows[block], prepared);
        _mm256_storeu_si256((__m256i *)&product->M[block * 8], rows);
    }
}

void xl_wbsm4_m32_multiply_prepared(
    const M32 *left, const XLWBSM4M32PreparedRight *prepared, M32 *product)
{
    __m256i left_rows[4];
    for (int block = 0; block < 4; ++block) {
        left_rows[block] =
            _mm256_loadu_si256((const __m256i *)&left->M[block * 8]);
    }
    xl_m32_multiply_prepared_rows(left_rows, prepared, product);
}

static inline __m128i xl_pack_m8_column(__m256i rows, __m256i mask)
{
    const __m256i packed = _mm256_shuffle_epi8(rows, mask);
    return _mm_unpacklo_epi32(
        _mm256_castsi256_si128(packed),
        _mm256_extracti128_si256(packed, 1));
}

static inline __m256i xl_pack_m8_matrices(__m256i rows)
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
    const __m128i a0 = xl_pack_m8_column(rows, mask3);
    const __m128i a1 = xl_pack_m8_column(rows, mask2);
    const __m128i a2 = xl_pack_m8_column(rows, mask1);
    const __m128i a3 = xl_pack_m8_column(rows, mask0);
    const __m128i lo = _mm_unpacklo_epi64(a0, a1);
    const __m128i hi = _mm_unpacklo_epi64(a2, a3);

    return _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
}

static inline uint8_t xl_xor_lane_low_bytes(__m256i value)
{
    const __m256i pair_xor =
        _mm256_xor_si256(value, _mm256_srli_si256(value, 8));
    const uint8_t lo = (uint8_t)_mm_cvtsi128_si32(
        _mm256_castsi256_si128(pair_xor));
    const uint8_t hi = (uint8_t)_mm_cvtsi128_si32(
        _mm256_extracti128_si256(pair_xor, 1));
    return (uint8_t)(lo ^ hi);
}

static inline uint32_t xl_m32_multiply_number_rows(
    const __m256i rows[4], uint32_t input)
{
    const __m256i x = _mm256_set_epi64x(
        (long long)(uint8_t)input,
        (long long)(uint8_t)(input >> 8),
        (long long)(uint8_t)(input >> 16),
        (long long)(uint8_t)(input >> 24));
    uint32_t output = 0;

    for (int block = 0; block < 4; ++block) {
        const __m256i matrices = xl_pack_m8_matrices(rows[block]);
        const __m256i value =
            _mm256_gf2p8affine_epi64_epi8(x, matrices, 0);
        output |= (uint32_t)xl_xor_lane_low_bytes(value) << (24 - block * 8);
    }
    return output;
}

static inline void xl_affinemix_from_left_rows(
    const __m256i left_rows[4], uint32_t left_vector,
    const Aff32 *right, Aff32 *output, int include_left_vector)
{
    XLWBSM4M32PreparedRight prepared;
    xl_wbsm4_m32_prepare_right(&right->Mat, &prepared);
    xl_m32_multiply_prepared_rows(left_rows, &prepared, &output->Mat);
    output->Vec.V = xl_m32_multiply_number_rows(left_rows, right->Vec.V);
    if (include_left_vector) {
        output->Vec.V ^= left_vector;
    }
}

void xl_wbsm4_affinemix_same_left3(
    const Aff32 *left,
    const Aff32 *right0, const Aff32 *right1, const Aff32 *right2,
    Aff32 *output0, Aff32 *output1, Aff32 *output2)
{
    __m256i left_rows[4];
    for (int block = 0; block < 4; ++block) {
        left_rows[block] =
            _mm256_loadu_si256((const __m256i *)&left->Mat.M[block * 8]);
    }
    xl_affinemix_from_left_rows(
        left_rows, left->Vec.V, right0, output0, 1);
    xl_affinemix_from_left_rows(
        left_rows, left->Vec.V, right1, output1, 1);
    xl_affinemix_from_left_rows(
        left_rows, left->Vec.V, right2, output2, 1);
}

void xl_wbsm4_affinemix_same_left2_without_second_left_vector(
    const Aff32 *left, const Aff32 *right0, const Aff32 *right1,
    Aff32 *output0, Aff32 *output1)
{
    __m256i left_rows[4];
    for (int block = 0; block < 4; ++block) {
        left_rows[block] =
            _mm256_loadu_si256((const __m256i *)&left->Mat.M[block * 8]);
    }
    xl_affinemix_from_left_rows(
        left_rows, left->Vec.V, right0, output0, 1);
    xl_affinemix_from_left_rows(
        left_rows, left->Vec.V, right1, output1, 0);
}

void xl_wbsm4_affine8x4_prepare(
    const Aff8 affine[4], XLWBSM4Affine8x4Prepared *prepared)
{
    uint64_t lanes[4];
    for (int index = 0; index < 4; ++index) {
        lanes[index] = (uint64_t)_mm_cvtsi128_si64(
            _mm_loadl_epi64((const __m128i *)affine[index].Mat.M));
    }
    prepared->matrices = _mm256_loadu_si256((const __m256i *)lanes);
    prepared->vector = (uint32_t)affine[0].Vec.V |
        ((uint32_t)affine[1].Vec.V << 8) |
        ((uint32_t)affine[2].Vec.V << 16) |
        ((uint32_t)affine[3].Vec.V << 24);
}

uint32_t xl_wbsm4_affine8x4_apply(
    const XLWBSM4Affine8x4Prepared *prepared, uint8_t input)
{
    const __m256i x = _mm256_set1_epi64x((long long)input);
    const __m256i value =
        _mm256_gf2p8affine_epi64_epi8(x, prepared->matrices, 0);
    const __m256i gather = _mm256_setr_epi8(
        0, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        0, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i packed = _mm256_shuffle_epi8(value, gather);
    const uint32_t lo = (uint32_t)_mm_cvtsi128_si32(
        _mm256_castsi256_si128(packed));
    const uint32_t hi = (uint32_t)_mm_cvtsi128_si32(
        _mm256_extracti128_si256(packed, 1));
    const uint32_t linear = (lo & 0x0000ffffu) |
        ((hi & 0x000000ffu) << 16) | ((hi & 0x0000ff00u) << 16);
    return linear ^ prepared->vector;
}

static inline __m256i xl_qbox_raw_block(
    __m256i q_rows, const XLWBSM4M32PreparedRight *prepared_csl)
{
    const __m256i identity = _mm256_set1_epi64x(0x0102040810204080ULL);
    const __m256i mask0 = _mm256_setr_epi8(
        3, 7, 11, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        3, 7, 11, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i mask1 = _mm256_setr_epi8(
        2, 6, 10, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        2, 6, 10, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i mask2 = _mm256_setr_epi8(
        1, 5, 9, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        1, 5, 9, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i mask3 = _mm256_setr_epi8(
        0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i broadcast = _mm256_setr_epi32(0, 4, 0, 4, 0, 4, 0, 4);
    const __m256i a0 = _mm256_permutevar8x32_epi32(
        _mm256_shuffle_epi8(q_rows, mask0), broadcast);
    const __m256i a1 = _mm256_permutevar8x32_epi32(
        _mm256_shuffle_epi8(q_rows, mask1), broadcast);
    const __m256i a2 = _mm256_permutevar8x32_epi32(
        _mm256_shuffle_epi8(q_rows, mask2), broadcast);
    const __m256i a3 = _mm256_permutevar8x32_epi32(
        _mm256_shuffle_epi8(q_rows, mask3), broadcast);
    const __m256i t0 = _mm256_gf2p8affine_epi64_epi8(
        prepared_csl->transposed_blocks[0], a0, 0);
    const __m256i t1 = _mm256_gf2p8affine_epi64_epi8(
        prepared_csl->transposed_blocks[1], a1, 0);
    const __m256i t2 = _mm256_gf2p8affine_epi64_epi8(
        prepared_csl->transposed_blocks[2], a2, 0);
    const __m256i t3 = _mm256_gf2p8affine_epi64_epi8(
        prepared_csl->transposed_blocks[3], a3, 0);
    const __m256i product = _mm256_xor_si256(
        _mm256_xor_si256(t0, t1), _mm256_xor_si256(t2, t3));

    return _mm256_gf2p8affine_epi64_epi8(identity, product, 0);
}

static inline __m256i xl_qbox_collect_lane(
    __m256i raw0, __m256i raw1, __m256i raw2, __m256i raw3,
    __m256i lane_selector)
{
    __m256i output = _mm256_blend_epi32(
        _mm256_permutevar8x32_epi32(raw0, lane_selector),
        _mm256_permutevar8x32_epi32(raw1, lane_selector), 0x0c);
    output = _mm256_blend_epi32(
        output, _mm256_permutevar8x32_epi32(raw2, lane_selector), 0x30);
    return _mm256_blend_epi32(
        output, _mm256_permutevar8x32_epi32(raw3, lane_selector), 0xc0);
}

void xl_wbsm4_qbox_prepare(
    const M32 *q_matrix, const XLWBSM4M32PreparedRight *prepared_csl,
    XLWBSM4QBoxLinearPrepared *prepared)
{
    const __m256i q0 =
        _mm256_loadu_si256((const __m256i *)&q_matrix->M[0]);
    const __m256i q1 =
        _mm256_loadu_si256((const __m256i *)&q_matrix->M[8]);
    const __m256i q2 =
        _mm256_loadu_si256((const __m256i *)&q_matrix->M[16]);
    const __m256i q3 =
        _mm256_loadu_si256((const __m256i *)&q_matrix->M[24]);
    const __m256i raw0 = xl_qbox_raw_block(q0, prepared_csl);
    const __m256i raw1 = xl_qbox_raw_block(q1, prepared_csl);
    const __m256i raw2 = xl_qbox_raw_block(q2, prepared_csl);
    const __m256i raw3 = xl_qbox_raw_block(q3, prepared_csl);
    const __m256i lane3 = _mm256_setr_epi32(6, 7, 6, 7, 6, 7, 6, 7);
    const __m256i lane2 = _mm256_setr_epi32(4, 5, 4, 5, 4, 5, 4, 5);
    const __m256i lane1 = _mm256_setr_epi32(2, 3, 2, 3, 2, 3, 2, 3);
    const __m256i lane0 = _mm256_setr_epi32(0, 1, 0, 1, 0, 1, 0, 1);

    prepared->byte_maps[0] =
        xl_qbox_collect_lane(raw0, raw1, raw2, raw3, lane3);
    prepared->byte_maps[1] =
        xl_qbox_collect_lane(raw0, raw1, raw2, raw3, lane2);
    prepared->byte_maps[2] =
        xl_qbox_collect_lane(raw0, raw1, raw2, raw3, lane1);
    prepared->byte_maps[3] =
        xl_qbox_collect_lane(raw0, raw1, raw2, raw3, lane0);
}

static inline uint32_t xl_qbox_apply_one(__m256i matrices, uint8_t input)
{
    const __m256i x = _mm256_set1_epi64x((long long)input);
    const __m256i value = _mm256_gf2p8affine_epi64_epi8(x, matrices, 0);
    const __m256i gather = _mm256_setr_epi8(
        0, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        0, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i packed = _mm256_shuffle_epi8(value, gather);
    const uint32_t lo = (uint32_t)_mm_cvtsi128_si32(
        _mm256_castsi256_si128(packed));
    const uint32_t hi = (uint32_t)_mm_cvtsi128_si32(
        _mm256_extracti128_si256(packed, 1));
    const uint32_t b0 = lo & 0xffu;
    const uint32_t b1 = (lo >> 8) & 0xffu;
    const uint32_t b2 = hi & 0xffu;
    const uint32_t b3 = (hi >> 8) & 0xffu;
    return (b0 << 24) | (b1 << 16) | (b2 << 8) | b3;
}

void xl_wbsm4_qbox_apply4(
    const XLWBSM4QBoxLinearPrepared *prepared,
    const uint8_t inputs[4], uint32_t outputs[4])
{
    for (int index = 0; index < 4; ++index) {
        outputs[index] =
            xl_qbox_apply_one(prepared->byte_maps[index], inputs[index]);
    }
}
