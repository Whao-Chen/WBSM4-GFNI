#include "wbsm4_se_cl.h"

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
	u8 state16[16][16];
	u8 state128[16][128];
	u8 states128[128];
    for (int i = 0; i < 16; i++)
	{
		for (int j = 0; j < 16; j++)
		{
			state16[i][j] = TypeI_c[i][input[i]][j];
		}
	}

	for (int i = 0; i < 16; i++)
	{
		for (int j = 0; j < 16; j++)
		{
			for (int k = 0; k < 8; k++)
			{
				state128[i][j * 8 + k] = (state16[i][j] >> (7 - k)) & 1;
			}
		}
	}

	for (int i = 0; i < 128; i++)
	{
		states128[i] = 0;
		for (int j = 0; j < 16; j++)
		{
			states128[i] = states128[i] ^ state128[j][i];
		}
	}

	for (int i = 0; i < 16; i++)
	{
		states[i] = states128[i * 8 + 0] * 128 + states128[i * 8 + 1] * 64 + states128[i * 8 + 2] * 32 + states128[i * 8 + 3] * 16 + states128[i * 8 + 4] * 8 + states128[i * 8 + 5] * 4 + states128[i * 8 + 6] * 2 + states128[i * 8 + 7];

	}
	

	// TypeII_C
	for (int i = 0; i < 16; i++)
	{
		for (int j = 0; j < 16; j++)
		{
			state16[i][j] = TypeII_c[i][states[i]][j];
		}
	}

	for (int i = 0; i < 16; i++)
	{
		for (int j = 0; j < 16; j++)
		{
			for (int k = 0; k < 8; k++)
			{
				state128[i][j * 8 + k] = (state16[i][j] >> (7 - k)) & 1;
			}
		}
	}

	for (int i = 0; i < 128; i++)
	{
		states128[i] = 0;
		for (int j = 0; j < 16; j++)
		{
			states128[i] = states128[i] ^ state128[j][i];
		}
	}

	for (int i = 0; i < 16; i++)
	{
		states[i] = states128[i * 8 + 0] * 128 + states128[i * 8 + 1] * 64 + states128[i * 8 + 2] * 32 + states128[i * 8 + 3] * 16 + states128[i * 8 + 4] * 8 + states128[i * 8 + 5] * 4 + states128[i * 8 + 6] * 2 + states128[i * 8 + 7];

	}

	for (int i = 0; i < 4; i++)
	{

		states[i] = S1_f4[0][i][states[i]];

	}

	
	//31 rounds of table lookup
	for (int i = 0; i < 31; i++)
	{

		// TypeI_a
		for (int j = 0; j < 16; j++)
		{
			for (int k = 0; k < 16; k++)
			{
				state16[j][k] = TypeI_a[i][j][states[j]][k];
			}
		}

		for (int i1 = 0; i1 < 16; i1++)
		{
			for (int j = 0; j < 16; j++)
			{
				for (int k = 0; k < 8; k++)
				{
					state128[i1][j * 8 + k] = (state16[i1][j] >> (7 - k)) & 1;
				}
			}
		}
		for (int i1 = 0; i1 < 128; i1++)
		{
			states128[i1] = 0;
			for (int j = 0; j < 16; j++)
			{
				states128[i1] = states128[i1] ^ state128[j][i1];
			}
		}
		for (int i1 = 0; i1 < 16; i1++)
		{
			states[i1] = states128[i1 * 8 + 0] * 128 + states128[i1 * 8 + 1] * 64 + states128[i1 * 8 + 2] * 32 + states128[i1 * 8 + 3] * 16 + states128[i1 * 8 + 4] * 8 + states128[i1 * 8 + 5] * 4 + states128[i1 * 8 + 6] * 2 + states128[i1 * 8 + 7];

		}
		

		// TypeII_a
		for (int j = 0; j < 16; j++)
		{
			for (int k = 0; k < 16; k++)
			{
				state16[j][k] = TypeII_a[i][j][states[j]][k];
			}
		}
		for (int i1 = 0; i1 < 16; i1++)
		{
			for (int j = 0; j < 16; j++)
			{
				for (int k = 0; k < 8; k++)
				{
					state128[i1][j * 8 + k] = (state16[i1][j] >> (7 - k)) & 1;
				}
			}
		}
		for (int i1 = 0; i1 < 128; i1++)
		{
			states128[i1] = 0;
			for (int j = 0; j < 16; j++)
			{
				states128[i1] = states128[i1] ^ state128[j][i1];
			}
		}
		for (int i1 = 0; i1 < 16; i1++)
		{
			states[i1] = states128[i1 * 8 + 0] * 128 + states128[i1 * 8 + 1] * 64 + states128[i1 * 8 + 2] * 32 + states128[i1 * 8 + 3] * 16 + states128[i1 * 8 + 4] * 8 + states128[i1 * 8 + 5] * 4 + states128[i1 * 8 + 6] * 2 + states128[i1 * 8 + 7];

		}
		

		//S2
		for (int j = 0; j < 4; j++)
		{

			states[j] = S2_f4[i][j][states[j]];

		}


		// TypeI_b
		for (int j = 0; j < 16; j++)
		{
			for (int k = 0; k < 16; k++)
			{
				state16[j][k] = TypeI_b[i][j][states[j]][k];
			}
		}
		for (int i1 = 0; i1 < 16; i1++)
		{
			for (int j = 0; j < 16; j++)
			{
				for (int k = 0; k < 8; k++)
				{
					state128[i1][j * 8 + k] = (state16[i1][j] >> (7 - k)) & 1;
				}
			}
		}
		for (int i1 = 0; i1 < 128; i1++)
		{
			states128[i1] = 0;
			for (int j = 0; j < 16; j++)
			{
				states128[i1] = states128[i1] ^ state128[j][i1];
			}
		}
		for (int i1 = 0; i1 < 16; i1++)
		{
			states[i1] = states128[i1 * 8 + 0] * 128 + states128[i1 * 8 + 1] * 64 + states128[i1 * 8 + 2] * 32 + states128[i1 * 8 + 3] * 16 + states128[i1 * 8 + 4] * 8 + states128[i1 * 8 + 5] * 4 + states128[i1 * 8 + 6] * 2 + states128[i1 * 8 + 7];

		}


		// TypeII_b
		for (int j = 0; j < 16; j++)
		{
			for (int k = 0; k < 16; k++)
			{
				state16[j][k] = TypeII_b[i][j][states[j]][k];
			}
		}
		for (int i1 = 0; i1 < 16; i1++)
		{
			for (int j = 0; j < 16; j++)
			{
				for (int k = 0; k < 8; k++)
				{
					state128[i1][j * 8 + k] = (state16[i1][j] >> (7 - k)) & 1;
				}
			}
		}
		for (int i1 = 0; i1 < 128; i1++)
		{
			states128[i1] = 0;
			for (int j = 0; j < 16; j++)
			{
				states128[i1] = states128[i1] ^ state128[j][i1];
			}
		}
		for (int i1 = 0; i1 < 16; i1++)
		{
			states[i1] = states128[i1 * 8 + 0] * 128 + states128[i1 * 8 + 1] * 64 + states128[i1 * 8 + 2] * 32 + states128[i1 * 8 + 3] * 16 + states128[i1 * 8 + 4] * 8 + states128[i1 * 8 + 5] * 4 + states128[i1 * 8 + 6] * 2 + states128[i1 * 8 + 7];

		}


		//S1
		for (int j = 0; j < 4; j++)
		{
			states[j] = S1_f4[i + 1][j][states[j]];
		}
		

	}


	// TypeI_a
	for (int i = 0; i < 16; i++)
	{
		for (int j = 0; j < 16; j++)
		{
			state16[i][j] = TypeI_a[31][i][states[i]][j];
		}
	}
	for (int i1 = 0; i1 < 16; i1++)
	{
		for (int j = 0; j < 16; j++)
		{
			for (int k = 0; k < 8; k++)
			{
				state128[i1][j * 8 + k] = (state16[i1][j] >> (7 - k)) & 1;
			}
		}
	}
	for (int i1 = 0; i1 < 128; i1++)
	{
		states128[i1] = 0;
		for (int j = 0; j < 16; j++)
		{
			states128[i1] = states128[i1] ^ state128[j][i1];
		}
	}
	for (int i1 = 0; i1 < 16; i1++)
	{
		states[i1] = states128[i1 * 8 + 0] * 128 + states128[i1 * 8 + 1] * 64 + states128[i1 * 8 + 2] * 32 + states128[i1 * 8 + 3] * 16 + states128[i1 * 8 + 4] * 8 + states128[i1 * 8 + 5] * 4 + states128[i1 * 8 + 6] * 2 + states128[i1 * 8 + 7];

	}


	// TypeII_a
	for (int j = 0; j < 16; j++)
	{
		for (int k = 0; k < 16; k++)
		{
			state16[j][k] = TypeII_a[31][j][states[j]][k];
		}
	}
	for (int i1 = 0; i1 < 16; i1++)
	{
		for (int j = 0; j < 16; j++)
		{
			for (int k = 0; k < 8; k++)
			{
				state128[i1][j * 8 + k] = (state16[i1][j] >> (7 - k)) & 1;
			}
		}
	}
	for (int i1 = 0; i1 < 128; i1++)
	{
		states128[i1] = 0;
		for (int j = 0; j < 16; j++)
		{
			states128[i1] = states128[i1] ^ state128[j][i1];
		}
	}
	for (int i1 = 0; i1 < 16; i1++)
	{
		states[i1] = states128[i1 * 8 + 0] * 128 + states128[i1 * 8 + 1] * 64 + states128[i1 * 8 + 2] * 32 + states128[i1 * 8 + 3] * 16 + states128[i1 * 8 + 4] * 8 + states128[i1 * 8 + 5] * 4 + states128[i1 * 8 + 6] * 2 + states128[i1 * 8 + 7];

	}


	//S2
	for (int i = 0; i < 4; i++)
	{

		states[i] = S2_f4[31][i][states[i]];

	}
	

	// TypeI_d
	for (int i = 0; i < 16; i++)
	{
		for (int j = 0; j < 16; j++)
		{
			state16[i][j] = TypeI_d[i][states[i]][j];
		}
	}
	for (int i1 = 0; i1 < 16; i1++)
	{
		for (int j = 0; j < 16; j++)
		{
			for (int k = 0; k < 8; k++)
			{
				state128[i1][j * 8 + k] = (state16[i1][j] >> (7 - k)) & 1;
			}
		}
	}
	for (int i1 = 0; i1 < 128; i1++)
	{
		states128[i1] = 0;
		for (int j = 0; j < 16; j++)
		{
			states128[i1] = states128[i1] ^ state128[j][i1];
		}
	}
	for (int i1 = 0; i1 < 16; i1++)
	{
		states[i1] = states128[i1 * 8 + 0] * 128 + states128[i1 * 8 + 1] * 64 + states128[i1 * 8 + 2] * 32 + states128[i1 * 8 + 3] * 16 + states128[i1 * 8 + 4] * 8 + states128[i1 * 8 + 5] * 4 + states128[i1 * 8 + 6] * 2 + states128[i1 * 8 + 7];

	}
	

	//Reverse
	for (int i = 0; i < 4; i++)
	{
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

static unsigned long long fnv1a64_update(unsigned long long h, const void *data, size_t len) {
    const unsigned char *p = (const unsigned char *)data;
    for (size_t i = 0; i < len; i++) {
        h ^= (unsigned long long)p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

unsigned long long wbsm4_se_cl_table_fingerprint(void) {
    unsigned long long h = 1469598103934665603ULL;
    h = fnv1a64_update(h, TypeI_c, sizeof(TypeI_c));
    h = fnv1a64_update(h, TypeII_c, sizeof(TypeII_c));
    h = fnv1a64_update(h, TypeI_a, sizeof(TypeI_a));
    h = fnv1a64_update(h, TypeII_a, sizeof(TypeII_a));
    h = fnv1a64_update(h, TypeI_b, sizeof(TypeI_b));
    h = fnv1a64_update(h, TypeII_b, sizeof(TypeII_b));
    h = fnv1a64_update(h, TypeI_d, sizeof(TypeI_d));
    h = fnv1a64_update(h, S1_f4, sizeof(S1_f4));
    h = fnv1a64_update(h, S2_f4, sizeof(S2_f4));
    return h;
}

unsigned long long wbsm4_se_cl_table_bytes(void) {
    return (unsigned long long)(sizeof(TypeI_c) + sizeof(TypeII_c) +
                                sizeof(TypeI_a) + sizeof(TypeII_a) +
                                sizeof(TypeI_b) + sizeof(TypeII_b) +
                                sizeof(TypeI_d) + sizeof(S1_f4) +
                                sizeof(S2_f4));
}

size_t wbsm4_se_cl_table_region_count(void) {
    return 9;
}

const char *wbsm4_se_cl_table_region_name(size_t region) {
    switch (region) {
        case 0: return "TypeI_c";
        case 1: return "TypeII_c";
        case 2: return "TypeI_a";
        case 3: return "TypeII_a";
        case 4: return "TypeI_b";
        case 5: return "TypeII_b";
        case 6: return "TypeI_d";
        case 7: return "S1_f4";
        case 8: return "S2_f4";
        default: return "invalid";
    }
}

unsigned long long wbsm4_se_cl_table_region_fingerprint(size_t region) {
    unsigned long long h = 1469598103934665603ULL;
    switch (region) {
        case 0: return fnv1a64_update(h, TypeI_c, sizeof(TypeI_c));
        case 1: return fnv1a64_update(h, TypeII_c, sizeof(TypeII_c));
        case 2: return fnv1a64_update(h, TypeI_a, sizeof(TypeI_a));
        case 3: return fnv1a64_update(h, TypeII_a, sizeof(TypeII_a));
        case 4: return fnv1a64_update(h, TypeI_b, sizeof(TypeI_b));
        case 5: return fnv1a64_update(h, TypeII_b, sizeof(TypeII_b));
        case 6: return fnv1a64_update(h, TypeI_d, sizeof(TypeI_d));
        case 7: return fnv1a64_update(h, S1_f4, sizeof(S1_f4));
        case 8: return fnv1a64_update(h, S2_f4, sizeof(S2_f4));
        default: return 0;
    }
}

unsigned long long wbsm4_se_cl_table_region_bytes(size_t region) {
    switch (region) {
        case 0: return (unsigned long long)sizeof(TypeI_c);
        case 1: return (unsigned long long)sizeof(TypeII_c);
        case 2: return (unsigned long long)sizeof(TypeI_a);
        case 3: return (unsigned long long)sizeof(TypeII_a);
        case 4: return (unsigned long long)sizeof(TypeI_b);
        case 5: return (unsigned long long)sizeof(TypeII_b);
        case 6: return (unsigned long long)sizeof(TypeI_d);
        case 7: return (unsigned long long)sizeof(S1_f4);
        case 8: return (unsigned long long)sizeof(S2_f4);
        default: return 0;
    }
}

int wbsm4_se_cl_gen_table_with_external(const unsigned char *key, int external_encoding) {
    return wbsm4_se_cl_gen_table_with_external_direction(key, external_encoding, 0);
}
