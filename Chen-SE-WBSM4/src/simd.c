#include "wbsm4_se_cl.h"

#if defined(__AVX2__) || defined(__SSE2__)
#include <immintrin.h>
#endif

// wbsm4_se_cl_context *wbsm4_se_cl_context_init(int encmode) {
//     wbsm4_se_cl_context *ctx = malloc(sizeof(wbsm4_se_cl_context));
//     if (ctx == NULL) {
//         goto cleanup;
//     }
//     memset(ctx, 0, sizeof(wbsm4_se_cl_context));
//     ctx->encmode = encmode;
//     return ctx;
// cleanup:
//     return NULL;
// }

// void wbsm4_se_cl_context_free(wbsm4_se_cl_context *ctx) {
//     memset(ctx, 0, sizeof(wbsm4_se_cl_context));
// }

static void apply_lookup_xor_layer_scalar(const u8 table[16][256][16], const u8 input[16], u8 output[16]) {
    u8 acc[16] = {0};
    for (int share = 0; share < 16; share++) {
        const u8 *entry = table[share][input[share]];
        for (int byte = 0; byte < 16; byte++) {
            acc[byte] ^= entry[byte];
        }
    }
    for (int byte = 0; byte < 16; byte++) {
        output[byte] = acc[byte];
    }
}

static void apply_lookup_xor_layer(const u8 table[16][256][16], const u8 input[16], u8 output[16]) {
#if defined(__AVX2__) || defined(__SSE2__)
    __m128i acc = _mm_setzero_si128();
    for (int share = 0; share < 16; share++) {
        const __m128i entry = _mm_loadu_si128((const __m128i *)table[share][input[share]]);
        acc = _mm_xor_si128(acc, entry);
    }
    _mm_storeu_si128((__m128i *)output, acc);
#else
    apply_lookup_xor_layer_scalar(table, input, output);
#endif
}

int wbsm4_se_cl_gen_table_with_external_direction(const unsigned char *key, int external_encoding, int decrypt) {

    sm4_context *sm4_key = sm4_context_init();
    // if (ctx->encmode) {
    //     sm4_setkey_enc(sm4_key, key);
    // } else {
    //     sm4_setkey_dec(sm4_key, key);
    // }
if (decrypt) sm4_setkey_dec(sm4_key, key);
    else sm4_setkey_enc(sm4_key, key);

    u8 P[64][128][128];
	u8 P_b[65][17][128];
	u8 P_inv[64][128][128];
	u8 F[128][128];
	u8 G[128][128];

	u8 A_1[128][128];
	u8 A_2[128][128];
	u8 A_3[128][128];
	u8 A_13[128][128];
	u8 A_1_b[128];
	u8 A_2_b[32][128];
	u8 A_3_b[128];
	u8 A_13_b[31][128];

	u8 STypeI_O[32][128][128];
	u8 STypeI_O_b[32][128];
	u8 STypeI_I[32][128][128];
	u8 STypeI_I_b[32][128];
	u8 STypeII_O[32][128][128];
	u8 STypeII_O_b[32][128];
	u8 STypeII_I[32][128][128];
	u8 STypeII_I_b[32][128];

    //Coding auxiliary variables
	u8 F8[16][8][8];
	u8 G8[16][8][8];
	u8 STypeI_OP[32][16][8][8];
	u8 STypeI_OP_b[32][16][8];
	u8 STypeI_IP[32][16][8][8];
	u8 STypeI_IP_b[32][16][8];
	u8 STypeII_OP[32][16][8][8];
	u8 STypeII_OP_b[32][16][8];
	u8 STypeII_IP[32][16][8][8];
	u8 STypeII_IP_b[32][16][8];
	u8 S1_f4a[32][4][256];
	u8 S2_f4a[32][4][256];
	u8 S1_f4a_inv[32][4][256];
	u8 S2_f4a_inv[32][4][256];
	
    getP128(P, P_inv,G);

    getP_b(P_b);
    if (external_encoding) {
        getP8w(F8, STypeI_OP, STypeI_IP, STypeII_OP, STypeII_IP);
    } else {
        getP8(F8, G8, STypeI_OP, STypeI_IP, STypeII_OP, STypeII_IP);
        for (int i = 0; i < 128; i++) {
            for (int j = 0; j < 128; j++) {
                G[i][j] = (i == j) ? 1 : 0;
            }
        }
    }
	
    generateOI(STypeI_OP, STypeI_IP, STypeII_OP, STypeII_IP, STypeI_OP_b, STypeI_IP_b, STypeII_OP_b, STypeII_IP_b);
	getA(A_1, A_2, A_3, A_13);
	getA_b(sm4_key->sk, A_1_b, A_2_b, A_3_b, A_13_b);

	gOI128(STypeI_OP, STypeI_OP_b, STypeI_O, STypeI_O_b);
	gOI128(STypeI_IP, STypeI_IP_b, STypeI_I, STypeI_I_b);
	gOI128(STypeII_OP, STypeII_OP_b, STypeII_O, STypeII_O_b);
	gOI128(STypeII_IP, STypeII_IP_b, STypeII_I, STypeII_I_b);

	gS_4(S1_f4a, S1_f4a_inv, S1_f4);
	gS2_4(S2_f4a, S2_f4a_inv, S2_f4);

    getTypeI_c(F8, A_1, A_1_b, P, P_b, TypeI_c);
	getTypeII_c(P_inv[0], P_b[0], STypeI_O[0], STypeI_O_b[0], TypeII_c);

	for (int i = 0; i < 31; i++)
	{
		getTypeI_a(S1_f4a_inv[i], P[i * 2 + 1], P_b[i * 2 + 1], A_2, A_2_b[i], STypeI_IP[i], STypeI_IP_b[i], TypeI_a[i]);
		getTypeII_a(P_inv[i * 2 + 1], P_b[i * 2 + 1], STypeII_O[i], STypeII_O_b[i], TypeII_a[i]);
		getTypeI_b(S2_f4a_inv[i], P[i * 2 + 2], P_b[i * 2 + 2], A_13, A_13_b[i], STypeII_IP[i], STypeII_IP_b[i], TypeI_b[i]);
		getTypeII_b(P_inv[i * 2 + 2], P_b[i * 2 + 2], STypeI_O[i + 1], STypeI_O_b[i + 1], TypeII_b[i]);
	}

	getTypeI_a(S1_f4a_inv[31], P[63], P_b[63], A_2, A_2_b[31], STypeI_IP[31], STypeI_IP_b[31], TypeI_a[31]);
	getTypeII_a(P_inv[63], P_b[63], STypeII_O[31], STypeII_O_b[31], TypeII_a[31]);
    getTypeI_d(S2_f4a_inv[31], A_3, A_3_b, STypeII_IP[31], STypeII_IP_b[31], TypeI_d, G);
    return 1;
}

int wbsm4_se_cl_gen_table(const unsigned char *key) {
    return wbsm4_se_cl_gen_table_with_external(key, 1);
}

int wbsm4_se_cl_encrypt(const unsigned char *input, unsigned char *output) {
    u8 states[16];

    apply_lookup_xor_layer(TypeI_c, input, states);
    apply_lookup_xor_layer(TypeII_c, states, states);

    for (int i = 0; i < 4; i++) {
        states[i] = S1_f4[0][i][states[i]];
    }

    // 31 rounds of table lookup
    for (int i = 0; i < 31; i++) {
        apply_lookup_xor_layer(TypeI_a[i], states, states);
        apply_lookup_xor_layer(TypeII_a[i], states, states);

        for (int j = 0; j < 4; j++) {
            states[j] = S2_f4[i][j][states[j]];
        }

        apply_lookup_xor_layer(TypeI_b[i], states, states);
        apply_lookup_xor_layer(TypeII_b[i], states, states);

        for (int j = 0; j < 4; j++) {
            states[j] = S1_f4[i + 1][j][states[j]];
        }
    }

    apply_lookup_xor_layer(TypeI_a[31], states, states);
    apply_lookup_xor_layer(TypeII_a[31], states, states);

    for (int i = 0; i < 4; i++) {
        states[i] = S2_f4[31][i][states[i]];
    }

    apply_lookup_xor_layer(TypeI_d, states, states);

    // Reverse
    for (int i = 0; i < 4; i++) {
        output[i] = states[i + 12];
        output[i + 4] = states[i + 8];
        output[i + 8] = states[i + 4];
        output[i + 12] = states[i];
    }

    return 1;
}

int wbsm4_se_cl_ecb_encrypt(const unsigned char *input, unsigned char *output, size_t len) {
    assert (len % 16 == 0);
    while (len) {
        wbsm4_se_cl_encrypt(input,output);
        len -= 16;
        input += 16;
        output += 16;
    }
}

int wbsm4_se_cl_gen_table_with_external(const unsigned char *key, int external_encoding) {
    return wbsm4_se_cl_gen_table_with_external_direction(key, external_encoding, 0);
}
