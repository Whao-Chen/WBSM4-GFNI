#include "WBMatrix/WBMatrix.h"
#include "WBMatrixOpt/WBMatrixOpt.h"
#include <immintrin.h>
#include <stdint.h>


typedef struct M16RegPair_GPT {
    __m128i top;
    __m128i bot;
} M16RegPair_GPT;

/*
 * 生成 64-bit 原始随机块。
 *
 * 用途：
 * - 将两次 cus_random() 拼成一个 uint64_t，用于 M8 block、M16 block 的随机行数据。
 * - 该 helper 只生成原始随机比特，不保证生成的块自身可逆；可逆性由 Schur
 *   或 Feistel/lifting 结构保证。
 *
 * 空间复杂度：O(1)。随机源语义保持 cus_random()，不引入 fast-RNG 替换。
 */
static inline uint64_t gpt32_random64(void) {
    uint64_t high = (uint64_t)cus_random();
    uint64_t low = (uint64_t)cus_random();
    return (high << 32) | low;
}


/*
 * 从公开 M16 结构加载到内部 M16RegPair_GPT。
 *
 * 数据编排：
 * - mat->M[0..7] 进入 top，mat->M[8..15] 进入 bot。
 * - 每个 lane 仍是一行 uint16_t，保持 MSB-first 列 bit 解释。
 *
 * 空间复杂度：O(1)。服务 Schur 子生成器输出与 M32 block core 的衔接。
 */


/*
 * 生成一个随机 M16 block 的内部寄存器表示。
 *
 * 用途：
 * - Schur 路线中作为任意 B/C block。
 * - Feistel/lifting 路线中作为任意 A/B/C/D 参数。
 * - 生成结果不要求可逆，整体矩阵可逆性由公式结构保证。
 *
 * 空间复杂度：两个 XMM 输出，O(1)。不改变随机源语义。
 */
static inline M16RegPair_GPT gpt32_rand_m16_reg(void) {
    M16RegPair_GPT out;
    out.top = _mm_set_epi64x((long long)gpt32_random64(), (long long)gpt32_random64());
    out.bot = _mm_set_epi64x((long long)gpt32_random64(), (long long)gpt32_random64());
    return out;
}


/*
 * 构造 M16 零矩阵的内部寄存器表示。
 *
 * 用途：服务 Schur pair-right helper 中 `Add ^ A*B` 的 zero addend。
 * 空间复杂度：O(1)，不影响采样覆盖。
 */
static inline M16RegPair_GPT gpt32_zero_m16_reg(void) {
    M16RegPair_GPT out;
    out.top = _mm_setzero_si128();
    out.bot = _mm_setzero_si128();
    return out;
}

/*
 * M16 block 加法 helper。
 *
 * GF(2) 上矩阵加法就是逐 bit XOR；输入输出都保持 top/bot 行格式。
 * 服务 Schur、Feistel/lifting 以及 prepared-right muladd；O(1)，无静态表。
 */



static inline M16RegPair_GPT gpt32_mul_m16(M16RegPair_GPT a, M16RegPair_GPT b) {
    M16RegPair_GPT out;
    MatMulMatM16_reg(a.top, a.bot, b.top, b.bot, &out.top, &out.bot);
    return out;
}

/*
 * fused `Out = Add ^ A*B` 的 M16 行格式 helper。
 *
 * 用途：
 * - Schur 公式中的 D 和 inv00。
 * - Feistel/lifting 展开中的单位块加乘。
 *
 * 注意：乘积先回到标准 M16 行格式再 XOR，不使用未经验证的 raw GFNI 中间域。
 * 空间复杂度：O(1)，不改变采样覆盖。
 */



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

/*
 * 将一组 top 或 bot M16 block 行打包为 M32 的连续 8 行。
 *
 * 输入语义：
 * - left/right 是 M32 同一 block 行的左右两个 M16 block。
 * - half=0 选择 top 8 行，half=1 选择 bot 8 行。
 *
 * 数据编排：
 * - M32 每行应为 left16 << 16 | right16。
 * - _mm_unpack*_epi16(right, left) 在小端 uint32_t 存储下形成该行值。
 *
 * 空间复杂度：O(1)。只服务 gpt32_pack_m16_blocks，不影响覆盖率。
 */
static inline void gpt32_store_m32_row8(uint32_t *dst, M16RegPair_GPT left,
                                        M16RegPair_GPT right, int half) {
    __m128i l = half == 0 ? left.top : left.bot;
    __m128i r = half == 0 ? right.top : right.bot;
    __m128i lo = _mm_unpacklo_epi16(r, l);
    __m128i hi = _mm_unpackhi_epi16(r, l);
    _mm_storeu_si128((__m128i *)dst, lo);
    _mm_storeu_si128((__m128i *)(dst + 4), hi);
}

/*
 * 将 2x2 个 M16 block 打包成公开 M32 行格式。
 *
 * 数据编排：
 * - 每个 M16 block 在寄存器中保存为 top/bot 两个 __m128i，每个 lane 是一行
 *   uint16_t。
 * - M32 的一行是 left_block_row << 16 | right_block_row。
 * - _mm_unpack*_epi16(right, left) 正好在小端内存中形成这个 uint32_t 行值。
 *
 * 空间复杂度：只使用寄存器临时，O(1)，无静态表。
 */
static inline void gpt32_pack_m16_blocks(M16RegPair_GPT lt, M16RegPair_GPT rt,
                                         M16RegPair_GPT lb, M16RegPair_GPT rb,
                                         M32 *out) {
    gpt32_store_m32_row8(out->M, lt, rt, 0);
    gpt32_store_m32_row8(out->M + 8, lt, rt, 1);
    gpt32_store_m32_row8(out->M + 16, lb, rb, 0);
    gpt32_store_m32_row8(out->M + 24, lb, rb, 1);
}

/*
 * 轻量 bounded random helper。
 *
 * 算法思想：
 * - 使用 multiply-high 将 cus_random() 映射到 [0, bound)，用于 Fisher-Yates。
 * - 这不是 rejection sampling；若严格均匀性是目标，需要单独验证 cus_random()
 *   和映射偏差。本项目当前只按动态 P support 和实测速度描述。
 *
 * 空间复杂度：O(1)。服务所有 GPT 动态 P helper。
 */
static inline uint32_t gpt32_bounded_rand(uint32_t bound) {
    return (uint32_t)(((uint64_t)cus_random() * (uint64_t)bound) >> 32);
}

typedef struct Gpt32BitstreamP {
    uint64_t bits;
    int available;
} Gpt32BitstreamP;

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

/*
 * 动态 M32 行置换 helper。
 *
 * 算法思想：
 * - 正向矩阵做 Fisher-Yates 行交换，等价于 Mat = P * Base。
 * - 逆矩阵同步做列交换，等价于 Mat_inv = BaseInv * P^-1。
 *
 * 数据编排和成本：
 * - 行交换直接交换 M32.M[i]。
 * - 列交换在 32 个 uint32_t 行上用 AVX2 分 4 组处理，每次交换两个 bit 列。
 * - 该方法覆盖运行时 32! 行置换空间，不依赖静态 P 表；但每次置换最多
 *   31 轮列位交换，速度明显慢于无最终 P 的速度版。
 *
 * 空间复杂度：O(1)，只使用栈上临时寄存器和两个输出矩阵，无静态表。
 */


/*
 * 动态 M32 P 的第二种实现：行重排 + 构造 P^-1 + M32 快乘。
 *
 * 算法思想：
 * - 一次 Fisher-Yates 生成 p[32]。
 * - Mat 直接做一次行重排，得到 Mat[i] = Base[p[i]]。
 * - 构造行格式 P^-1，其中第 p[i] 行在第 i 列为 1。
 * - 逆矩阵用 MatMulMatM32_ptr 计算 BaseInv * P^-1。
 *
 * 与逐 swap 列置换的关系：
 * - 数学语义相同，都是 Mat=P*Base、Mat_inv=BaseInv*P^-1。
 * - 差异是本方法用一次 M32 快速乘法替代多次全矩阵列 bit-swap，适合比较
 *   动态 P 生产路线是否能降到可接受成本。
 *
 * 空间复杂度：
 * - 栈上 p[32] 约 32B，Pinv 一个 M32 约 128B；无静态 P 表。
 * - 如果批量缓存 P/P^-1，则空间按样本数线性增长。本函数不缓存。
 */
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

    MatMulMatM32_ptr(base_inv, &pinv, mat_inv);
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

    MatMulMatM32_rowdirect_ptr(base_inv, &pinv, mat_inv);
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
        MatMulMatM32_rowdirect_ptr(base_inv, &pinv, mat_inv);
    } else {
        MatMulMatM32_ptr(base_inv, &pinv, mat_inv);
    }
}


/*
 * 直接右乘 P^-1 的列置换 helper。
 *
 * 算法思想：
 * - 对于 Mat = P * Base，逆矩阵应为 BaseInv * P^-1。
 * - p[i] 表示新矩阵第 i 行来自 Base 的第 p[i] 行，所以右乘 P^-1 的
 *   效果是：输出第 i 列取输入第 p[i] 列。
 *
 * 数据编排：
 * - M32 行格式是 MSB-first，逻辑列 j 对应 `1u << (31 - j)`。
 * - 本 helper 不构造 dense P^-1，也不调用通用 MatMulMatM32；每行用
 *   p[32] 直接重新排列 32 个 bit 列。
 *
 * 空间复杂度：
 * - 只读 p[32]，输出一个 M32；不额外分配 pinv，O(1)，无静态 P 表。
 * - 标量 bit-permute 不针对特定测速，主要用于验证“专用置换能否替代
 *   P^-1 + M32 快乘”。如果慢于 GFNI 快乘，应保留为负结果。
 */


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


void genMatpairM32_radix2_unitriangular_plu_dynamicP_recursiveBlocks_twoLeftTwoRight_rowdirectPinv(
    M32 *Mat, M32 *Mat_inv) {
    gpt32_radix2_unitriangular_plu_recursive_twoleft_tworight_core(1, 0, Mat, Mat_inv);
}

