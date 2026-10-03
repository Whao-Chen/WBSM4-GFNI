#include "wbsm4.h"
#include "sbox.h"
#include "wbsm4_specialized.h"

static const uint32_t sm4_csl_xor_matrix[32] = {
    0xA0202080, 0x50101040, 0x28080820, 0x14040410,
    0x0A020208, 0x05010104, 0x02808082, 0x01404041,
    0x80A02020, 0x40501010, 0x20280808, 0x10140404,
    0x080A0202, 0x04050101, 0x82028080, 0x41014040,
    0x2080A020, 0x10405010, 0x08202808, 0x04101404,
    0x02080A02, 0x01040501, 0x80820280, 0x40410140,
    0x202080A0, 0x10104050, 0x08082028, 0x04041014,
    0x0202080A, 0x01010405, 0x80808202, 0x40404101
};

void initMatrixL(M32 *csl_matrix)
{
    for (int row = 0; row < 32; ++row) {
        csl_matrix->M[row] = sm4_csl_xor_matrix[row];
    }
}

void sm4_wb_gen_secrect_sbox(sm4_context *sm4_key)
{
    uint8_t *output = (uint8_t *)skbox_enc;
    for (int round = 0; round < 32; ++round) {
        const uint32_t key = sm4_key->sk[round];
        for (int input = 0; input < 256; ++input) {
            output[input] = SBOX[input ^ ((key >> 24) & 0xff)];
            output[256 + input] = SBOX[input ^ ((key >> 16) & 0xff)];
            output[512 + input] = SBOX[input ^ ((key >> 8) & 0xff)];
            output[768 + input] = SBOX[input ^ (key & 0xff)];
        }
        output += 1024;
    }
}

void sm4_wb_gen_affine(Sm4Whitebox *sm4_wb_ctx)
{
    (void)sm4_wb_ctx;
    for (int round = 0; round < SM4_NUM_ROUNDS + 4; ++round) {
        genaffinepairM32(&P[round][0], &P[round][1]);
        for (int byte = 0; byte < 4; ++byte) {
            genaffinepairM8(&E[round][byte][0], &E[round][byte][1]);
        }
        affinecomM8to32(
            E[round][0][1], E[round][1][1],
            E[round][2][1], E[round][3][1], &EC[round][1]);
        genaffinepairM32(&Q[round][0], &Q[round][1]);
    }
}

void sm4_wb_combine_tables(Sm4Whitebox *sm4_wb_ctx, M32 csl_matrix)
{
    XLWBSM4M32PreparedRight prepared_csl;
    xl_wbsm4_m32_prepare_right(&csl_matrix, &prepared_csl);

    for (int round = 0; round < SM4_NUM_ROUNDS; ++round) {
        Aff8 e_apply[4] = {
            E[round][0][0], E[round][1][0],
            E[round][2][0], E[round][3][0]
        };
        XLWBSM4Affine8x4Prepared prepared_e;
        XLWBSM4QBoxLinearPrepared prepared_qbox;
        const uint32_t q_vector = Q[round][0].Vec.V;

        xl_wbsm4_affinemix_same_left3(
            &EC[round][1],
            &P[round + 1][1], &P[round + 2][1], &P[round + 3][1],
            &sm4_wb_ctx->M[round][0], &sm4_wb_ctx->M[round][1],
            &sm4_wb_ctx->M[round][2]);

        xl_wbsm4_affine8x4_prepare(e_apply, &prepared_e);
        xl_wbsm4_qbox_prepare(
            &Q[round][0].Mat, &prepared_csl, &prepared_qbox);

        for (int input = 0; input < 256; ++input) {
            uint8_t substituted[4];
            uint32_t output[4];
            const uint32_t encoded =
                xl_wbsm4_affine8x4_apply(&prepared_e, (uint8_t)input);

            substituted[0] = skbox_enc[round][0][(uint8_t)encoded];
            substituted[1] = skbox_enc[round][1][(uint8_t)(encoded >> 8)];
            substituted[2] = skbox_enc[round][2][(uint8_t)(encoded >> 16)];
            substituted[3] = skbox_enc[round][3][(uint8_t)(encoded >> 24)];
            xl_wbsm4_qbox_apply4(&prepared_qbox, substituted, output);

            sm4_wb_ctx->Table[round][0][input] = output[0];
            sm4_wb_ctx->Table[round][1][input] = output[1];
            sm4_wb_ctx->Table[round][2][input] = output[2];
            sm4_wb_ctx->Table[round][3][input] = output[3] ^ q_vector;
        }

        xl_wbsm4_affinemix_same_left2_without_second_left_vector(
            &P[round + 4][0], &P[round][1], &Q[round][1],
            &sm4_wb_ctx->C[round], &sm4_wb_ctx->D[round]);
    }

    for (int index = 0; index < 4; ++index) {
        sm4_wb_ctx->SE[index] = P[index][0];
        sm4_wb_ctx->FE[index] = P[SM4_NUM_ROUNDS + index][1];
    }
}

void sm4_wb_gen_tables_direction(uint8_t *key, Sm4Whitebox *sm4_wb_ctx, int decrypt)
{
    sm4_context context;
    M32 csl_matrix;
if (decrypt) sm4_setkey_dec(&context, key);
    else sm4_setkey_enc(&context, key);
    sm4_wb_gen_secrect_sbox(&context);
    sm4_wb_gen_affine(sm4_wb_ctx);
    initMatrixL(&csl_matrix);
    sm4_wb_combine_tables(sm4_wb_ctx, csl_matrix);
}

void sm4_wb_gen_tables(uint8_t *key, Sm4Whitebox *sm4_wb_ctx)
{
    sm4_wb_gen_tables_direction(key, sm4_wb_ctx, 0);
}
