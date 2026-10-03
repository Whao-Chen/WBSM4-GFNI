#include "wbsm4.h"
#include "sbox.h"

#include <immintrin.h>
#include <string.h>

#ifndef BAIWU_GFNI_TR
#define BAIWU_GFNI_TR 1
#endif

static uint32_t baiwu_generation_seed = UINT32_C(0x26080513);

void baiwu_set_generation_seed(uint32_t seed)
{
    baiwu_generation_seed = seed;
}

static const uint8_t GFNI_SEQUENCE[64] = {
     0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63
};

__attribute__((noinline))
static __m512i sm4_sbox_zmm(__m512i input)
{
    const __m512i pre_matrix = _mm512_set1_epi64(
        (long long)UINT64_C(0x4c287db91a22505d));
    const __m512i post_matrix = _mm512_set1_epi64(
        (long long)UINT64_C(0xf3ab34a974a6b589));
    const __m512i pre = _mm512_gf2p8affine_epi64_epi8(
        input, pre_matrix, 0x3e);
    return _mm512_gf2p8affineinv_epi64_epi8(
        pre, post_matrix, 0xd3);
}

__attribute__((noinline))
static __m512i matmul_m8_zmm(const M8 *matrix, __m512i input)
{
    uint64_t rows;
    memcpy(&rows, matrix->M, sizeof(rows));
    return _mm512_gf2p8affine_epi64_epi8(
        input, _mm512_set1_epi64((long long)rows), 0);
}

__attribute__((noinline))
static __m512i matmul_rows_zmm(uint64_t rows, __m512i input)
{
    return _mm512_gf2p8affine_epi64_epi8(
        input, _mm512_set1_epi64((long long)rows), 0);
}

static uint64_t extract_m8_rows(const M32 *matrix,
                                int input_byte,
                                int output_byte)
{
    uint64_t rows = 0;
    int row;
    for (row = 0; row < 8; ++row) {
        rows |= (uint64_t)(uint8_t)(
            matrix->M[output_byte * 8 + row] >>
            (24 - 8 * input_byte)) << (8 * row);
    }
    return rows;
}

static void store_words64(uint32_t *output,
                          __m512i byte0, __m512i byte1,
                          __m512i byte2, __m512i byte3,
                          uint32_t xor_value)
{
#define STORE_CHUNK(CHUNK) do {                                               \
        __m512i words = _mm512_slli_epi32(                                   \
            _mm512_cvtepu8_epi32(                                            \
                _mm512_extracti32x4_epi32(byte0, (CHUNK))), 24);             \
        words = _mm512_or_si512(words, _mm512_slli_epi32(                    \
            _mm512_cvtepu8_epi32(                                            \
                _mm512_extracti32x4_epi32(byte1, (CHUNK))), 16));            \
        words = _mm512_or_si512(words, _mm512_slli_epi32(                    \
            _mm512_cvtepu8_epi32(                                            \
                _mm512_extracti32x4_epi32(byte2, (CHUNK))), 8));             \
        words = _mm512_or_si512(words, _mm512_cvtepu8_epi32(                 \
            _mm512_extracti32x4_epi32(byte3, (CHUNK))));                     \
        words = _mm512_xor_si512(                                            \
            words, _mm512_set1_epi32((int)xor_value));                       \
        _mm512_storeu_si512((void *)(output + 16 * (CHUNK)), words);         \
    } while (0)

    STORE_CHUNK(0);
    STORE_CHUNK(1);
    STORE_CHUNK(2);
    STORE_CHUNK(3);
#undef STORE_CHUNK
}

M32 L_matrix = {
    .M[0] = 0xA0202080, 
    .M[1] = 0x50101040, 
    .M[2] = 0x28080820, 
    .M[3] = 0x14040410,
    .M[4] = 0xA020208, 
    .M[5] = 0x5010104, 
    .M[6] = 0x2808082, 
    .M[7] = 0x1404041, 
    .M[8] = 0x80A02020, 
    .M[9] = 0x40501010, 
    .M[10] = 0x20280808, 
    .M[11] = 0x10140404, 
    .M[12] = 0x80A0202, 
    .M[13] = 0x4050101, 
    .M[14] = 0x82028080, 
    .M[15] = 0x41014040, 
    .M[16] = 0x2080A020, 
    .M[17] = 0x10405010, 
    .M[18] = 0x8202808, 
    .M[19] = 0x4101404, 
    .M[20] = 0x2080A02, 
    .M[21] = 0x1040501, 
    .M[22] = 0x80820280, 
    .M[23] = 0x40410140, 
    .M[24] = 0x202080A0, 
    .M[25] = 0x10104050, 
    .M[26] = 0x8082028, 
    .M[27] = 0x4041014, 
    .M[28] = 0x202080A, 
    .M[29] = 0x1010405, 
    .M[30] = 0x80808202, 
    .M[31] = 0x40404101
};

void printstate(unsigned char * in)
{
    int i;
    for(i = 0; i < 16; i++) 
    {
        printf("%.2X", in[i]);
    }
    printf("\n");
}

void wbsm4_gen_direction(uint8_t *key, int decrypt)
{
    int i, j, r, x, y;
    uint8_t temp_u8_x, temp_u8_y, temp_u8;
    uint32_t temp_u32;
    uint32_t TD0_u32[6], TD1_u32[6], TD2_u32[3], TR_u32[4], Lc[36], Ec0[32], Ec1[32];
    M32 L[36];
    M32 L_inv[36];
    M8 E[32][2][4];
    M8 E_inv[32][2][4];
    M32 Ei_inv[32][2];
    M32 M[32][2][3];
    M32 C[32];
    M32 LL;

    sm4_context ctx;
if (decrypt) sm4_setkey_dec(&ctx, key);
    else sm4_setkey_enc(&ctx, key);
    InitRandom(baiwu_generation_seed);

    for(r = 0; r < 36; r++) 
    {
        genMatpairM32(&L[r], &L_inv[r]);
        Lc[r] = cus_random();
    }

    for(r = 0; r < 32; r++) 
    {
        for(j = 0; j < 4; j++) 
        {
            genMatpairM8(&E[r][0][j], &E_inv[r][0][j]);
            genMatpairM8(&E[r][1][j], &E_inv[r][1][j]);
        }

        MatrixcomM8to32(E_inv[r][0][0], E_inv[r][0][1], E_inv[r][0][2], E_inv[r][0][3], &Ei_inv[r][0]);
        MatrixcomM8to32(E_inv[r][1][0], E_inv[r][1][1], E_inv[r][1][2], E_inv[r][1][3], &Ei_inv[r][1]);

        MatMulMatM32(Ei_inv[r][0], L_inv[r + 1], &M[r][0][0]);
        MatMulMatM32(Ei_inv[r][0], L_inv[r + 2], &M[r][0][1]);
        MatMulMatM32(Ei_inv[r][0], L_inv[r + 3], &M[r][0][2]);

        MatMulMatM32(Ei_inv[r][1], L_inv[r + 1], &M[r][1][0]);
        MatMulMatM32(Ei_inv[r][1], L_inv[r + 2], &M[r][1][1]);
        MatMulMatM32(Ei_inv[r][1], L_inv[r + 3], &M[r][1][2]);

        MatMulMatM32(L[r + 4], L_inv[r], &C[r]);
    }

    for (r = 0; r < 32; r++)
    {
        MatMulMatM32(L[r + 4], L_matrix, &LL);

        for(i = 0; i < 6; i++)
        {
            TD0_u32[i] = cus_random();
            TD1_u32[i] = cus_random();
        }
        for(i = 0; i < 4; i++)
        {
            TR_u32[i] = cus_random();
        }
         for(i = 0; i < 3; i++)
        {
            TD2_u32[i] = cus_random();
        }
        Ec0[r] = TD0_u32[0] ^ TD0_u32[1] ^ TD0_u32[2] ^ TD0_u32[3] ^ TD0_u32[4] ^ TD0_u32[5];
        Ec1[r] = TD1_u32[0] ^ TD1_u32[1] ^ TD1_u32[2] ^ TD1_u32[3] ^ TD1_u32[4] ^ TD1_u32[5];

        for(x = 0; x < 256; x++) 
        {
            for(j = 0; j < 4; j++)
            {
                temp_u8 = x ^ ((Lc[r] >> (24 - j * 8)) & 0xff);
                temp_u32 = (uint32_t)temp_u8 << (24 - j * 8);
                TD[r][0][j][x] = MatMulNumM32(C[r], temp_u32);
            }
            for(j = 0; j < 3; j++)
            {
                TD[r][0][j][x] ^= TD2_u32[j];
            }
            TD[r][0][3][x] ^= Lc[r + 4] ^ TD2_u32[0] ^ TD2_u32[1] ^ TD2_u32[2] ^ TR_u32[0] ^ TR_u32[1] ^ TR_u32[2] ^ TR_u32[3]; 

            for(i = 1; i < 4; i++) 
            {
                temp_u8 = x ^ ((Lc[r + i] >> 24) & 0xff);
                temp_u32 = (uint32_t)temp_u8 << 24;
                TD[r][i][0][x] = MatMulNumM32(M[r][0][i - 1], temp_u32);
                temp_u8 = x ^ ((Lc[r + i] >> 16) & 0xff);
                temp_u32 = (uint32_t)temp_u8 << 16;
                TD[r][i][1][x] = MatMulNumM32(M[r][0][i - 1], temp_u32);

                temp_u8 = x ^ ((Lc[r + i] >> 8) & 0xff);
                temp_u32 = (uint32_t)temp_u8 << 8;
                TD[r][i][2][x] = MatMulNumM32(M[r][1][i - 1], temp_u32);
                temp_u8 = x ^ (Lc[r + i] & 0xff);
                temp_u32 = temp_u8;
                TD[r][i][3][x] = MatMulNumM32(M[r][1][i - 1], temp_u32);
            }

            j = 0;
            for(i = 1; i < 4; i++)
            {
                TD[r][i][0][x] ^= TD0_u32[j++];
                TD[r][i][1][x] ^= TD0_u32[j++];
            }

            j = 0;
            for(i = 1; i < 4; i++)
            {
                TD[r][i][2][x] ^= TD1_u32[j++];
                TD[r][i][3][x] ^= TD1_u32[j++];
            }
        }

#if BAIWU_GFNI_TR
        for(x = 0; x < 256; x++)
        {
            for(j = 0; j < 4; j++)
            {
                uint64_t output_rows[4];
                const __m512i sequence = _mm512_loadu_si512(
                    (const void *)GFNI_SEQUENCE);
                temp_u8_x = x ^ ((Ec0[r] >> (24 - j * 8)) & 0xff);
                temp_u8_x = MatMulNumM8(E[r][0][j], temp_u8_x);
                for(i = 0; i < 4; ++i)
                {
                    output_rows[i] = extract_m8_rows(&LL, j, i);
                }

                for(y = 0; y < 256; y += 64)
                {
                    __m512i y_values = _mm512_add_epi8(
                        sequence, _mm512_set1_epi8((char)y));
                    __m512i encoded_y = matmul_m8_zmm(
                        &E[r][1][j],
                        _mm512_xor_si512(
                            y_values,
                            _mm512_set1_epi8((char)(uint8_t)(
                                Ec1[r] >> (24 - j * 8)))));
                    __m512i substituted = sm4_sbox_zmm(_mm512_xor_si512(
                        encoded_y,
                        _mm512_set1_epi8((char)(uint8_t)(
                            temp_u8_x ^
                            (uint8_t)(ctx.sk[r] >> (24 - j * 8))))));
                    __m512i output0 = matmul_rows_zmm(
                        output_rows[0], substituted);
                    __m512i output1 = matmul_rows_zmm(
                        output_rows[1], substituted);
                    __m512i output2 = matmul_rows_zmm(
                        output_rows[2], substituted);
                    __m512i output3 = matmul_rows_zmm(
                        output_rows[3], substituted);
                    store_words64(&TR[r][j][x][y],
                                  output0, output1, output2, output3,
                                  TR_u32[j]);
                }
            }
        }
#else
        for(x = 0; x < 256; x++)
        {
            for(y = 0; y < 256; y++)
            {
                for(j = 0; j < 4; j++)
                {
                    temp_u8_x = x ^ ((Ec0[r] >> (24 - j * 8)) & 0xff);
                    temp_u8_x = MatMulNumM8(E[r][0][j], temp_u8_x);

                    temp_u8_y = y ^ ((Ec1[r] >> (24 - j * 8)) & 0xff);
                    temp_u8_y = MatMulNumM8(E[r][1][j], temp_u8_y);
                    temp_u8 = SBOX[temp_u8_x ^ temp_u8_y ^ ((ctx.sk[r] >> (24 - j * 8)) & 0xff)];
                    temp_u32 = (uint32_t)temp_u8 << (24 - j * 8);
                    TR[r][j][x][y] = MatMulNumM32(LL, temp_u32);
                    TR[r][j][x][y] ^= TR_u32[j];
                }
            }
        }
#endif
    }

    //external encoding
    for(i = 0; i < 4; i++) 
    {
        SE[i].Mat = L[i];
        SE[i].Vec.V = Lc[i];

        FE[i].Mat = L_inv[35 - i];
        FE[i].Vec.V = MatMulNumM32(L_inv[35 - i], Lc[35 - i]);
    }
}

void wbsm4_encrypt(unsigned char IN[], unsigned char OUT[])
{
    int r, i, j;
    uint32_t x[36];
    uint32_t s0, s1;
    
    x[0] = GET32(IN);
    x[1] = GET32(IN + 4);
    x[2] = GET32(IN + 8);
    x[3] = GET32(IN + 12);

    for(r = 0; r < 32; r++)
    {
        x[r + 4] = 0;
        s0 = 0;
        s1 = 0;

        for(i = 1; i < 4; i++)
        {
            s0 ^= TD[r][i][0][(x[r + i] >> 24) & 0xff];
            s0 ^= TD[r][i][1][(x[r + i] >> 16) & 0xff];
            s1 ^= TD[r][i][2][(x[r + i] >> 8) & 0xff];
            s1 ^= TD[r][i][3][x[r + i] & 0xff];
        }
        for(j = 0; j < 4; j++)
        {
            x[r + 4] ^= TR[r][j][(s0 >> (24 - j * 8)) & 0xff][(s1 >> (24 - j * 8)) & 0xff];
            x[r + 4] ^= TD[r][0][j][(x[r] >> (24 - j * 8)) & 0xff];
        }
    }

    PUT32(x[35], OUT);
    PUT32(x[34], OUT + 4);
    PUT32(x[33], OUT + 8);
    PUT32(x[32], OUT + 12);
}

void wbsm4_gen(uint8_t *key)
{
    wbsm4_gen_direction(key, 0);
}
