/* GFNI construction and its table-based validation executor.
 * Derived from SCNUcrypto's Chen self-equivalence implementation.
 * Released under Apache-2.0 with author authorization; see THIRD_PARTY_NOTICES.md.
 */
#include <cpuid.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <x86intrin.h>
#include "generateTable.h"
#include "sm4_new.h"
#define NOINLINE __attribute__((noinline))
#define NOIPA __attribute__((noinline, noipa))
#define GETP128_BINARY_PATH "p128_precomputed.bin"
static int g_decrypt;
enum {
    TABLE_I = 16,
    TABLE_X = 256,
    TABLE_OUT = 16,
    BENCH_SAMPLES = 5,
    GETP128_MODE_TEXT = 0,
    GETP128_MODE_CACHE = 1,
    GETP128_MODE_BINARY = 2
};

typedef struct {
    u8 TypeI_c[16][256][16];
    u8 TypeII_c[16][256][16];
    u8 TypeI_a[32][16][256][16];
    u8 TypeII_a[32][16][256][16];
    u8 TypeI_b[31][16][256][16];
    u8 TypeII_b[31][16][256][16];
    u8 TypeI_d[16][256][16];
    u8 S1_f4[32][4][256];
    u8 S2_f4[32][4][256];
} ChenTables;
static ChenTables g_tables;
typedef struct {
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
} ChenWork;
typedef struct {
    uint8_t rows[TABLE_I][TABLE_OUT][8];
    uint64_t rowq[TABLE_I][TABLE_OUT];
    uint8_t vec[TABLE_I][TABLE_OUT];
} BytePackedFamily;
typedef struct {
    uint64_t total;
    uint64_t setup;
    uint64_t s1_s2;
    uint64_t typei_c;
    uint64_t typeii_c;
    uint64_t typei_a;
    uint64_t typeii_a;
    uint64_t typei_b;
    uint64_t typeii_b;
    uint64_t typei_d;
} PhaseProfile;








typedef struct {
    u8 P[64][128][128];
    u8 P_inv[64][128][128];
    u8 G[128][128];
} P128Cache;
typedef struct {
    uint8_t magic[8];
    uint32_t version;
    uint32_t header_bytes;
    uint32_t p_count;
    uint32_t rows;
    uint32_t cols;
    uint64_t p_bytes;
    uint64_t p_inv_bytes;
    uint64_t g_bytes;
    uint64_t payload_bytes;
    uint64_t payload_checksum;
} P128BinaryHeader;
static ChenWork g_work;
static P128Cache g_p128_cache;
static P128Cache g_p128_binary_scratch;
static u8 g_zero_b[16][8];
static int g_getp128_mode = GETP128_MODE_TEXT;
static int g_p128_cache_ready = 0;
static unsigned long long g_p128_cache_loads = 0;
static unsigned long long g_p128_binary_loads = 0;
static unsigned long long g_p128_binary_writes = 0;
static unsigned long long g_p128_binary_last_checksum = 0;
static uint64_t cycles_now(void) {
    unsigned int aux;
    _mm_lfence();
    uint64_t t = __rdtscp(&aux);
    _mm_lfence();
    return t;
}
static unsigned long long fnv1a64_update(unsigned long long h, const void *data, size_t len) {
    const unsigned char *p = (const unsigned char *)data;
    for (size_t i = 0; i < len; i++) {
        h ^= (unsigned long long)p[i];
        h *= 1099511628211ULL;
    }
    return h;
}
static size_t p128_p_bytes(void) {
    return sizeof(g_p128_cache.P);
}
static size_t p128_p_inv_bytes(void) {
    return sizeof(g_p128_cache.P_inv);
}
static size_t p128_g_bytes(void) {
    return sizeof(g_p128_cache.G);
}
static void getP128_cached(u8 P[64][128][128],
                                 u8 P_inv[64][128][128],
                                 u8 G128[128][128]) {
    if (!g_p128_cache_ready) {
        getP128(g_p128_cache.P, g_p128_cache.P_inv, g_p128_cache.G);
        g_p128_cache_ready = 1;
        g_p128_cache_loads++;
    }
    memcpy(P, g_p128_cache.P, sizeof(g_p128_cache.P));
    memcpy(P_inv, g_p128_cache.P_inv, sizeof(g_p128_cache.P_inv));
    memcpy(G128, g_p128_cache.G, sizeof(g_p128_cache.G));
}
static unsigned long long p128_payload_checksum(u8 P[64][128][128],
                                                u8 P_inv[64][128][128],
                                                u8 G128[128][128]) {
    unsigned long long h = 1469598103934665603ULL;
    h = fnv1a64_update(h, P, p128_p_bytes());
    h = fnv1a64_update(h, P_inv, p128_p_inv_bytes());
    h = fnv1a64_update(h, G128, p128_g_bytes());
    return h;
}
static void fill_p128_binary_header(P128BinaryHeader *header,
                                    unsigned long long checksum) {
    static const uint8_t magic[8] = {'P', '1', '2', '8', 'B', 'I', 'N', '1'};
    memset(header, 0, sizeof(*header));
    memcpy(header->magic, magic, sizeof(magic));
    header->version = 1;
    header->header_bytes = (uint32_t)sizeof(*header);
    header->p_count = 64;
    header->rows = 128;
    header->cols = 128;
    header->p_bytes = (uint64_t)p128_p_bytes();
    header->p_inv_bytes = (uint64_t)p128_p_inv_bytes();
    header->g_bytes = (uint64_t)p128_g_bytes();
    header->payload_bytes = header->p_bytes + header->p_inv_bytes + header->g_bytes;
    header->payload_checksum = checksum;
}
static int validate_p128_binary_header(const P128BinaryHeader *header) {
    P128BinaryHeader expected;
    fill_p128_binary_header(&expected, header->payload_checksum);
    if (memcmp(header->magic, expected.magic, sizeof(expected.magic)) != 0) {
        return 0;
    }
    return header->version == expected.version &&
           header->header_bytes == expected.header_bytes &&
           header->p_count == expected.p_count &&
           header->rows == expected.rows &&
           header->cols == expected.cols &&
           header->p_bytes == expected.p_bytes &&
           header->p_inv_bytes == expected.p_inv_bytes &&
           header->g_bytes == expected.g_bytes &&
           header->payload_bytes == expected.payload_bytes;
}
static int write_p128_binary_file(const char *path,
                                  u8 P[64][128][128],
                                  u8 P_inv[64][128][128],
                                  u8 G128[128][128]) {
    unsigned long long checksum = p128_payload_checksum(P, P_inv, G128);
    P128BinaryHeader header;
    fill_p128_binary_header(&header, checksum);
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        printf("getp128_binary_write_open=FAIL,path=%s,errno=%d\n", path, errno);
        return 0;
    }
    int ok = 1;
    ok &= fwrite(&header, sizeof(header), 1, fp) == 1;
    ok &= fwrite(P, 1, p128_p_bytes(), fp) == p128_p_bytes();
    ok &= fwrite(P_inv, 1, p128_p_inv_bytes(), fp) == p128_p_inv_bytes();
    ok &= fwrite(G128, 1, p128_g_bytes(), fp) == p128_g_bytes();
    if (fclose(fp) != 0) {
        ok = 0;
    }
    if (!ok) {
        printf("getp128_binary_write=FAIL,path=%s\n", path);
        return 0;
    }
    g_p128_binary_writes++;
    g_p128_binary_last_checksum = checksum;
    return 1;
}
static int read_p128_binary_file(const char *path,
                                 u8 P[64][128][128],
                                 u8 P_inv[64][128][128],
                                 u8 G128[128][128],
                                 int print_errors) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        if (print_errors) {
            printf("getp128_binary_read_open=FAIL,path=%s,errno=%d\n", path, errno);
        }
        return 0;
    }
    P128BinaryHeader header;
    int ok = fread(&header, sizeof(header), 1, fp) == 1;
    if (ok && !validate_p128_binary_header(&header)) {
        ok = 0;
        if (print_errors) {
            printf("getp128_binary_header_gate=FAIL,path=%s\n", path);
        }
    }
    if (ok) {
        ok &= fread(P, 1, p128_p_bytes(), fp) == p128_p_bytes();
        ok &= fread(P_inv, 1, p128_p_inv_bytes(), fp) == p128_p_inv_bytes();
        ok &= fread(G128, 1, p128_g_bytes(), fp) == p128_g_bytes();
    }
    if (fclose(fp) != 0) {
        ok = 0;
    }
    if (!ok) {
        if (print_errors) {
            printf("getp128_binary_read=FAIL,path=%s\n", path);
        }
        return 0;
    }
    unsigned long long checksum = p128_payload_checksum(P, P_inv, G128);
    if (checksum != header.payload_checksum) {
        if (print_errors) {
            printf("getp128_binary_checksum_gate=FAIL,path=%s,expected=%016llx,actual=%016llx\n",
                   path, (unsigned long long)header.payload_checksum, checksum);
        }
        return 0;
    }
    g_p128_binary_loads++;
    g_p128_binary_last_checksum = checksum;
    return 1;
}
static int getP128_binary_exact(u8 P[64][128][128],
                                u8 P_inv[64][128][128],
                                u8 G128[128][128]) {
    errno = 0;
    FILE *fp = fopen(GETP128_BINARY_PATH, "rb");
    if (!fp) {
        if (errno != ENOENT) {
            printf("getp128_binary_probe=FAIL,path=%s,errno=%d\n", GETP128_BINARY_PATH, errno);
            return 0;
        }
        getP128(g_p128_binary_scratch.P, g_p128_binary_scratch.P_inv, g_p128_binary_scratch.G);
        if (!write_p128_binary_file(GETP128_BINARY_PATH,
                                    g_p128_binary_scratch.P,
                                    g_p128_binary_scratch.P_inv,
                                    g_p128_binary_scratch.G)) {
            return 0;
        }
    } else {
        fclose(fp);
    }
    return read_p128_binary_file(GETP128_BINARY_PATH, P, P_inv, G128, 1);
}
static int load_p128_for_active_mode(u8 P[64][128][128],
                                     u8 P_inv[64][128][128],
                                     u8 G128[128][128]) {
    if (g_getp128_mode == GETP128_MODE_CACHE) {
        getP128_cached(P, P_inv, G128);
        return 1;
    }
    if (g_getp128_mode == GETP128_MODE_BINARY) {
        return getP128_binary_exact(P, P_inv, G128);
    }
    getP128(P, P_inv, G128);
    return 1;
}
static uint8_t parity8(uint8_t x) {
    x ^= (uint8_t)(x >> 4);
    x &= 0x0fU;
    return (uint8_t)((0x6996U >> x) & 1U);
}
static uint8_t apply_rows_swar(const uint8_t rows[8], uint8_t x) {
    uint8_t y = 0;
    for (int i = 0; i < 8; i++) {
        y |= (uint8_t)(parity8((uint8_t)(rows[i] & x)) << (7 - i));
    }
    return y;
}
static uint64_t pack_rows_qword(const uint8_t rows[8]) {
    uint64_t mat_rows = 0;
    for (int i = 0; i < 8; i++) {
        mat_rows |= ((uint64_t)rows[i]) << (8 * i);
    }
    return mat_rows;
}
static uint8_t apply_rows_gfni_qword(uint64_t mat_rows, uint8_t x) {
    __m128i in = _mm_cvtsi32_si128((int)x);
    __m128i mat = _mm_cvtsi64_si128((long long)mat_rows);
    __m128i out = _mm_gf2p8affine_epi64_epi8(in, mat, 0);
    return (uint8_t)_mm_cvtsi128_si32(out);
}
static NOIPA __m128i gfni_affine_imm(__m128i x, __m128i mat, int imm) {
    switch (imm) {
    case 0x3e:
        return _mm_gf2p8affine_epi64_epi8(x, mat, 0x3e);
    case 0xd3:
        return _mm_gf2p8affine_epi64_epi8(x, mat, 0xd3);
    default:
        return _mm_setzero_si128();
    }
}
static NOIPA __m128i gfni_affineinv_imm(__m128i x, __m128i mat, int imm) {
    switch (imm) {
    case 0x3e:
        return _mm_gf2p8affineinv_epi64_epi8(x, mat, 0x3e);
    case 0xd3:
        return _mm_gf2p8affineinv_epi64_epi8(x, mat, 0xd3);
    default:
        return _mm_setzero_si128();
    }
}
static NOIPA void make_two_stage_table(uint8_t out[256],
                                       int op1, uint64_t mat1, int imm1,
                                       int op2, uint64_t mat2, int imm2) {
    const __m128i m1 = _mm_set1_epi64x((long long)mat1);
    const __m128i m2 = _mm_set1_epi64x((long long)mat2);
    for (int block = 0; block < 16; block++) {
        uint8_t in[16];
        for (int j = 0; j < 16; j++) {
            in[j] = (uint8_t)(block * 16 + j);
        }
        __m128i x = _mm_loadu_si128((const __m128i *)in);
        x = op1 ? gfni_affineinv_imm(x, m1, imm1) : gfni_affine_imm(x, m1, imm1);
        x = op2 ? gfni_affineinv_imm(x, m2, imm2) : gfni_affine_imm(x, m2, imm2);
        _mm_storeu_si128((__m128i *)(out + block * 16), x);
    }
}
static NOIPA void make_gfni_sbox(uint8_t out[256]) {
    make_two_stage_table(out, 0, 0x4C287DB91A22505DULL, 0x3e,
                         1, 0xF3AB34A974A6B589ULL, 0xd3);
}
static int make_inverse_table(const uint8_t sbox[256], uint8_t inv[256]) {
    uint8_t seen[256];
    memset(seen, 0, sizeof(seen));
    memset(inv, 0, 256);
    for (int i = 0; i < 256; i++) {
        uint8_t y = sbox[i];
        if (seen[y]) {
            return 0;
        }
        seen[y] = 1;
        inv[y] = (uint8_t)i;
    }
    return 1;
}
static int gfni_gS_4_gfni_source(u8 s1a[32][4][256],
                                      u8 s1a_inv[32][4][256],
                                      u8 s1[32][4][256]) {
    uint8_t gfni_sbox[256];
    make_gfni_sbox(gfni_sbox);
    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 4; j++) {
            ga(s1a[i][j], s1a_inv[i][j]);
            for (int k = 0; k < 256; k++) {
                s1[i][j][k] = s1a[i][j][gfni_sbox[k]];
            }
        }
    }
    return 1;
}
static int gfni_gS2_4_gfni_source(u8 s2a[32][4][256],
                                       u8 s2a_inv[32][4][256],
                                       u8 s2[32][4][256]) {
    uint8_t gfni_sbox[256];
    uint8_t gfni_inv[256];
    make_gfni_sbox(gfni_sbox);
    if (!make_inverse_table(gfni_sbox, gfni_inv)) {
        return 0;
    }
    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 4; j++) {
            ga(s2a[i][j], s2a_inv[i][j]);
            for (int k = 0; k < 256; k++) {
                s2[i][j][k] = s2a[i][j][gfni_inv[k]];
            }
        }
    }
    return 1;
}
static void pack_m128_8_colbits_local(u8 b[128][8], uint64_t colbits[8][2]) {
    for (int col = 0; col < 8; col++) {
        uint64_t lo = 0;
        uint64_t hi = 0;
        for (int k = 0; k < 64; k++) {
            lo |= ((uint64_t)(b[k][col] & 1u)) << k;
            hi |= ((uint64_t)(b[k + 64][col] & 1u)) << k;
        }
        colbits[col][0] = lo;
        colbits[col][1] = hi;
    }
}
static void m128_128_8_pack_rows_qword_prepared_local(const uint64_t rowbits[128][2],
                                                      u8 b[128][8],
                                                      uint8_t rows[TABLE_OUT][8],
                                                      uint64_t rowq[TABLE_OUT]) {
    uint64_t colbits[8][2];
    pack_m128_8_colbits_local(b, colbits);
    for (int out = 0; out < TABLE_OUT; out++) {
        for (int row = 0; row < 8; row++) {
            uint8_t row_byte = 0;
            int src_row = out * 8 + row;
            for (int col = 0; col < 8; col++) {
                unsigned int parity =
                    (unsigned int)__builtin_popcountll(rowbits[src_row][0] & colbits[col][0]) ^
                    (unsigned int)__builtin_popcountll(rowbits[src_row][1] & colbits[col][1]);
                row_byte |= (uint8_t)((parity & 1u) << (7 - col));
            }
            rows[out][row] = row_byte;
        }
        rowq[out] = pack_rows_qword(rows[out]);
    }
}
static void pack_m128_rowbits_local(u8 a[128][128], uint64_t rowbits[128][2]) {
    for (int r = 0; r < 128; r++) {
        uint64_t lo = 0;
        uint64_t hi = 0;
        for (int c = 0; c < 64; c++) {
            lo |= ((uint64_t)(a[r][c] & 1u)) << c;
            hi |= ((uint64_t)(a[r][c + 64] & 1u)) << c;
        }
        rowbits[r][0] = lo;
        rowbits[r][1] = hi;
    }
}
static void pack_v128_bits_local(const u8 b[128], uint64_t vbits[2]) {
    uint64_t lo = 0;
    uint64_t hi = 0;
    for (int c = 0; c < 64; c++) {
        lo |= ((uint64_t)(b[c] & 1u)) << c;
        hi |= ((uint64_t)(b[c + 64] & 1u)) << c;
    }
    vbits[0] = lo;
    vbits[1] = hi;
}
static void m128_128_pack_vec_prepared_local(const uint64_t rowbits[128][2],
                                             const u8 b[128],
                                             const u8 xor_vec[128],
                                             int has_xor_vec,
                                             uint8_t vec[TABLE_OUT]) {
    uint64_t vbits[2];
    pack_v128_bits_local(b, vbits);
    for (int out = 0; out < TABLE_OUT; out++) {
        uint8_t packed = 0;
        for (int row = 0; row < 8; row++) {
            int src_row = out * 8 + row;
            unsigned int parity =
                (unsigned int)__builtin_popcountll(rowbits[src_row][0] & vbits[0]) ^
                (unsigned int)__builtin_popcountll(rowbits[src_row][1] & vbits[1]);
            u8 bit = (u8)(parity & 1u);
            if (has_xor_vec) {
                bit ^= (u8)(xor_vec[src_row] & 1u);
            }
            packed |= (uint8_t)((bit & 1u) << (7 - row));
        }
        vec[out] = packed;
    }
}
static void make_typei_shares(u8 A_b[128], int bit_outer_share_order, u8 shares[16][128]) {
    if (bit_outer_share_order) {
        for (int bit = 0; bit < 128; bit++) {
            for (int share = 0; share < 15; share++) {
                shares[share][bit] = (u8)(rand() % 2);
            }
        }
    } else {
        for (int share = 0; share < 15; share++) {
            for (int bit = 0; bit < 128; bit++) {
                shares[share][bit] = (u8)(rand() % 2);
            }
        }
    }
    for (int bit = 0; bit < 128; bit++) {
        shares[15][bit] = A_b[bit];
        for (int share = 0; share < 15; share++) {
            shares[15][bit] ^= shares[share][bit];
        }
    }
}
static uint8_t typei_source_byte(u8 source[4][256], int i, int j) {
    if (!source) {
        return (uint8_t)j;
    }
    if (i < 4) {
        return source[i][j & 255];
    }
    return (uint8_t)j;
}
static void precompute_typei_bytepacked(u8 source[4][256],
                                        u8 Final[128][128],
                                        u8 Final_b[16][128],
                                        int has_final_vec,
                                        u8 A[128][128],
                                        u8 A_b[128],
                                        u8 IP[16][8][8],
                                        u8 IP_b[16][8],
                                        int bit_outer_share_order,
                                        BytePackedFamily *bp) {
    u8 shares[16][128];
    u8 block[128][8];
    u8 am[128][8];
    u8 ab[128];
    u8 base[128];
    u8 c[128][8];
    u8 d[128];

    uint64_t final_rowbits[128][2];


    memset(bp, 0, sizeof(*bp));
    make_typei_shares(A_b, bit_outer_share_order, shares);

    pack_m128_rowbits_local(Final, final_rowbits);

    for (int i = 0; i < TABLE_I; i++) {
        for (int r = 0; r < 128; r++) {
            for (int col = 0; col < 8; col++) {
                block[r][col] = A[r][i * 8 + col];
            }
        }
        M128_8(block, IP[i], am);
        M128_1(block, IP_b[i], ab);
        for (int r = 0; r < 128; r++) {
            base[r] = (u8)(ab[r] ^ shares[i][r]);
        }

        m128_128_8_pack_rows_qword_prepared_local((const uint64_t (*)[2])final_rowbits,
                                                  am,
                                                  bp->rows[i],
                                                  bp->rowq[i]);




        m128_128_pack_vec_prepared_local((const uint64_t (*)[2])final_rowbits,
                                         base,
                                         Final_b[i],
                                         has_final_vec,
                                         bp->vec[i]);








        for (int out = 0; out < TABLE_OUT; out++) {




        }
    }
}
static void get_typei_bytepacked(u8 source[4][256],
                                 u8 Final[128][128],
                                 u8 Final_b[16][128],
                                 int has_final_vec,
                                 u8 A[128][128],
                                 u8 A_b[128],
                                 u8 IP[16][8][8],
                                 u8 IP_b[16][8],
                                 int bit_outer_share_order,
                                 int use_gfni,
                                 u8 out[TABLE_I][TABLE_X][TABLE_OUT]) {
    BytePackedFamily bp;
    precompute_typei_bytepacked(source, Final, Final_b, has_final_vec,
                                A, A_b, IP, IP_b, bit_outer_share_order, &bp);








    for (int i = 0; i < TABLE_I; i++) {
        for (int j = 0; j < TABLE_X; j++) {
            uint8_t x = typei_source_byte(source, i, j);
            for (int k = 0; k < TABLE_OUT; k++) {
                uint8_t y = use_gfni ? apply_rows_gfni_qword(bp.rowq[i][k], x)
                                     : apply_rows_swar(bp.rows[i][k], x);
                out[i][j][k] = (u8)(y ^ bp.vec[i][k]);
            }
        }
    }
}
static void precompute_typeii_bytepacked(u8 P_inv[128][128],
                                         u8 P_b[17][128],
                                         u8 O[128][128],
                                         u8 O_b[128],
                                         BytePackedFamily *bp) {
    u8 block[128][8];
    u8 pb[8];
    u8 ab[128];
    u8 c[128][8];
    u8 d[128];

    uint64_t o_rowbits[128][2];


    memset(bp, 0, sizeof(*bp));

    pack_m128_rowbits_local(O, o_rowbits);

    for (int i = 0; i < TABLE_I; i++) {
        for (int r = 0; r < 128; r++) {
            for (int col = 0; col < 8; col++) {
                block[r][col] = P_inv[r][i * 8 + col];
            }
        }
        for (int col = 0; col < 8; col++) {
            pb[col] = P_b[16][i * 8 + col];
        }
        M128_1(block, pb, ab);

        m128_128_pack_vec_prepared_local((const uint64_t (*)[2])o_rowbits,
                                         ab,
                                         O_b,
                                         1,
                                         bp->vec[i]);







        m128_128_8_pack_rows_qword_prepared_local((const uint64_t (*)[2])o_rowbits,
                                                  block,
                                                  bp->rows[i],
                                                  bp->rowq[i]);



        for (int out = 0; out < TABLE_OUT; out++) {




        }
    }
}
static void get_typeii_bytepacked(u8 P_inv[128][128],
                                  u8 P_b[17][128],
                                  u8 O[128][128],
                                  u8 O_b[128],
                                  int use_gfni,
                                  u8 out[TABLE_I][TABLE_X][TABLE_OUT]) {
    BytePackedFamily bp;
    precompute_typeii_bytepacked(P_inv, P_b, O, O_b, &bp);








    for (int i = 0; i < TABLE_I; i++) {
        for (int j = 0; j < TABLE_X; j++) {
            uint8_t x = (uint8_t)j;
            for (int k = 0; k < TABLE_OUT; k++) {
                uint8_t y = use_gfni ? apply_rows_gfni_qword(bp.rowq[i][k], x)
                                     : apply_rows_swar(bp.rows[i][k], x);
                out[i][j][k] = (u8)(y ^ bp.vec[i][k]);
            }
        }
    }
}
static int setup_work(const unsigned char key[16], int external_encoding) {
    sm4_context *sm4_key = sm4_context_init();
    if (!sm4_key) {
        return 0;
    }
    if (g_decrypt) sm4_setkey_dec(sm4_key, (unsigned char *)key);
    else sm4_setkey_enc(sm4_key, (unsigned char *)key);

    if (!load_p128_for_active_mode(g_work.P, g_work.P_inv, g_work.G)) {
        sm4_context_free(sm4_key);
        free(sm4_key);
        return 0;
    }
    getP_b(g_work.P_b);
    if (external_encoding) {
        getP8w(g_work.F8, g_work.STypeI_OP, g_work.STypeI_IP,
               g_work.STypeII_OP, g_work.STypeII_IP);
    } else {






        getP8(g_work.F8, g_work.G8, g_work.STypeI_OP, g_work.STypeI_IP,
              g_work.STypeII_OP, g_work.STypeII_IP);

        for (int i = 0; i < 128; i++) {
            for (int j = 0; j < 128; j++) {
                g_work.G[i][j] = (i == j) ? 1 : 0;
            }
        }
    }
    generateOI(g_work.STypeI_OP, g_work.STypeI_IP,
               g_work.STypeII_OP, g_work.STypeII_IP,
               g_work.STypeI_OP_b, g_work.STypeI_IP_b,
               g_work.STypeII_OP_b, g_work.STypeII_IP_b);
    getA(g_work.A_1, g_work.A_2, g_work.A_3, g_work.A_13);
    getA_b((u32 *)sm4_key->sk, g_work.A_1_b, g_work.A_2_b,
           g_work.A_3_b, g_work.A_13_b);
    gOI128(g_work.STypeI_OP, g_work.STypeI_OP_b,
           g_work.STypeI_O, g_work.STypeI_O_b);
    gOI128(g_work.STypeI_IP, g_work.STypeI_IP_b,
           g_work.STypeI_I, g_work.STypeI_I_b);
    gOI128(g_work.STypeII_OP, g_work.STypeII_OP_b,
           g_work.STypeII_O, g_work.STypeII_O_b);
    gOI128(g_work.STypeII_IP, g_work.STypeII_IP_b,
           g_work.STypeII_I, g_work.STypeII_I_b);

    sm4_context_free(sm4_key);
    free(sm4_key);
    return 1;
}
static int build_tables_gfni(ChenTables *t,
                                          const unsigned char key[16],
                                          int external_encoding,
                                          int use_gfni,
                                          PhaseProfile *p) {
    memset(p, 0, sizeof(*p));
    uint64_t total0 = cycles_now();
    uint64_t t0 = cycles_now();
    if (!setup_work(key, external_encoding)) {
        return 0;
    }
    uint64_t t1 = cycles_now();
    p->setup = t1 - t0;

    t0 = cycles_now();
    if (use_gfni) {
        if (!gfni_gS_4_gfni_source(g_work.S1_f4a, g_work.S1_f4a_inv, t->S1_f4)) {
            return 0;
        }
        if (!gfni_gS2_4_gfni_source(g_work.S2_f4a, g_work.S2_f4a_inv, t->S2_f4)) {
            return 0;
        }
    } else {
        gS_4(g_work.S1_f4a, g_work.S1_f4a_inv, t->S1_f4);
        gS2_4(g_work.S2_f4a, g_work.S2_f4a_inv, t->S2_f4);
    }
    t1 = cycles_now();
    p->s1_s2 = t1 - t0;

    t0 = cycles_now();
    get_typei_bytepacked(NULL, g_work.P[0], g_work.P_b[0], 1,
                         g_work.A_1, g_work.A_1_b,
                         g_work.F8, g_zero_b,
                         0, use_gfni, t->TypeI_c);
    t1 = cycles_now();
    p->typei_c = t1 - t0;

    t0 = cycles_now();
    get_typeii_bytepacked(g_work.P_inv[0], g_work.P_b[0],
                          g_work.STypeI_O[0], g_work.STypeI_O_b[0],
                          use_gfni, t->TypeII_c);
    t1 = cycles_now();
    p->typeii_c = t1 - t0;

    for (int i = 0; i < 31; i++) {
        t0 = cycles_now();
        get_typei_bytepacked(g_work.S1_f4a_inv[i], g_work.P[i * 2 + 1],
                             g_work.P_b[i * 2 + 1], 1,
                             g_work.A_2, g_work.A_2_b[i],
                             g_work.STypeI_IP[i], g_work.STypeI_IP_b[i],
                             0, use_gfni, t->TypeI_a[i]);
        t1 = cycles_now();
        p->typei_a += t1 - t0;

        t0 = cycles_now();
        get_typeii_bytepacked(g_work.P_inv[i * 2 + 1], g_work.P_b[i * 2 + 1],
                              g_work.STypeII_O[i], g_work.STypeII_O_b[i],
                              use_gfni, t->TypeII_a[i]);
        t1 = cycles_now();
        p->typeii_a += t1 - t0;

        t0 = cycles_now();
        get_typei_bytepacked(g_work.S2_f4a_inv[i], g_work.P[i * 2 + 2],
                             g_work.P_b[i * 2 + 2], 1,
                             g_work.A_13, g_work.A_13_b[i],
                             g_work.STypeII_IP[i], g_work.STypeII_IP_b[i],
                             0, use_gfni, t->TypeI_b[i]);
        t1 = cycles_now();
        p->typei_b += t1 - t0;

        t0 = cycles_now();
        get_typeii_bytepacked(g_work.P_inv[i * 2 + 2], g_work.P_b[i * 2 + 2],
                              g_work.STypeI_O[i + 1], g_work.STypeI_O_b[i + 1],
                              use_gfni, t->TypeII_b[i]);
        t1 = cycles_now();
        p->typeii_b += t1 - t0;
    }

    t0 = cycles_now();
    get_typei_bytepacked(g_work.S1_f4a_inv[31], g_work.P[63], g_work.P_b[63], 1,
                         g_work.A_2, g_work.A_2_b[31],
                         g_work.STypeI_IP[31], g_work.STypeI_IP_b[31],
                         0, use_gfni, t->TypeI_a[31]);
    t1 = cycles_now();
    p->typei_a += t1 - t0;

    t0 = cycles_now();
    get_typeii_bytepacked(g_work.P_inv[63], g_work.P_b[63],
                          g_work.STypeII_O[31], g_work.STypeII_O_b[31],
                          use_gfni, t->TypeII_a[31]);
    t1 = cycles_now();
    p->typeii_a += t1 - t0;

    t0 = cycles_now();
    get_typei_bytepacked(g_work.S2_f4a_inv[31], g_work.G, g_work.P_b[0], 0,
                         g_work.A_3, g_work.A_3_b,
                         g_work.STypeII_IP[31], g_work.STypeII_IP_b[31],
                         1, use_gfni, t->TypeI_d);
    t1 = cycles_now();
    p->typei_d = t1 - t0;
    p->total = cycles_now() - total0;
    return 1;
}
static void apply_2d_layer(const u8 table[16][256][16], const u8 in[16], u8 out[16]) {
    for (int dst = 0; dst < 16; dst++) {
        u8 v = 0;
        for (int src = 0; src < 16; src++) {
            v ^= table[src][in[src]][dst];
        }
        out[dst] = v;
    }
}
static void apply_3d_layer(const u8 table[16][256][16], u8 state[16]) {
    u8 out[16];
    apply_2d_layer(table, state, out);
    memcpy(state, out, sizeof(out));
}
static void encrypt_with_tables(const ChenTables *t, const unsigned char input[16], unsigned char output[16]) {
    u8 state[16];
    u8 tmp[16];

    apply_2d_layer(t->TypeI_c, input, state);
    apply_2d_layer(t->TypeII_c, state, tmp);
    memcpy(state, tmp, sizeof(state));
    for (int i = 0; i < 4; i++) {
        state[i] = t->S1_f4[0][i][state[i]];
    }

    for (int r = 0; r < 31; r++) {
        apply_3d_layer(t->TypeI_a[r], state);
        apply_3d_layer(t->TypeII_a[r], state);
        for (int i = 0; i < 4; i++) {
            state[i] = t->S2_f4[r][i][state[i]];
        }
        apply_3d_layer(t->TypeI_b[r], state);
        apply_3d_layer(t->TypeII_b[r], state);
        for (int i = 0; i < 4; i++) {
            state[i] = t->S1_f4[r + 1][i][state[i]];
        }
    }

    apply_3d_layer(t->TypeI_a[31], state);
    apply_3d_layer(t->TypeII_a[31], state);
    for (int i = 0; i < 4; i++) {
        state[i] = t->S2_f4[31][i][state[i]];
    }
    apply_3d_layer(t->TypeI_d, state);

    for (int i = 0; i < 4; i++) {
        output[i] = state[i + 12];
        output[i + 4] = state[i + 8];
        output[i + 8] = state[i + 4];
        output[i + 12] = state[i];
    }
}

static PhaseProfile g_last_profile;
int chen_gfni_prepare(void) {
    getP128_cached(g_work.P, g_work.P_inv, g_work.G);
    return 0;
}
int chen_gfni_generate(const unsigned char key[16], unsigned int seed, int decrypt) {
    g_decrypt = decrypt;
    srand(seed);
    g_getp128_mode = GETP128_MODE_CACHE;
    return build_tables_gfni(&g_tables, key, 0, 1, &g_last_profile) ? 0 : -1;
}
unsigned long long chen_gfni_construction_ticks(void) { return g_last_profile.total; }
void chen_gfni_execute(const unsigned char *in, unsigned char *out, size_t blocks) {
    for (size_t i=0;i<blocks;i++) encrypt_with_tables(&g_tables,in+16*i,out+16*i);
}
