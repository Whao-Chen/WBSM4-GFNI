#include "WBMatrixOpt/WBMatrixOpt.h"
#include <immintrin.h>
#include <stdint.h>

/*
 * 辅助函数：将16x16矩阵(uint16 M[16])转换为4个8x8子矩阵块并打包到256位寄存器中
 * 
 * 输入 M16 布局:
 * Row 0: [High0 Low0]
 * ...
 * Row15: [High15 Low15]
 *
 * 输出 YMM 布局 (64-bit lanes):
 * Lane 0: A00 (Rows 0-7, Low Bytes)
 * Lane 1: A01 (Rows 0-7, High Bytes)
 * Lane 2: A10 (Rows 8-15, Low Bytes)
 * Lane 3: A11 (Rows 8-15, High Bytes)
 */


/*
 * 辅助函数：将计算好的4个8x8子矩阵块打包回 M16 格式
 * 
 * 输入: [C00, C01, C10, C11] 分别在 Lane 0-3
 * 输出: 恢复交错的 uint16 格式
 */


/*
 * 方案1: MatMulMatM16_Opt1
 * 使用 256-bit GFNI (AVX512VL + GFNI)
 * 分两次并行计算：一次计算 4 个子乘法，总共两次
 */


/*
 * 方案2: MatMulMatM16_Opt2
 * 使用 512-bit GFNI (AVX512F + GFNI)
 * 单指令流：一条 512-bit 指令并行完成所有 8 个子矩阵乘法
 */


/*
 * 辅助函数：从16x16矩阵中提取并打包数据到128位寄存器
 * type 0: A00 (64-bit) | A00 (64-bit) (from Rows 0-7, Bytes 1,3,5...)
 * type 1: A01 (64-bit) | A01 (64-bit) (from Rows 0-7, Bytes 0,2,4...)
 * type 2: A10 (64-bit) | A10 (64-bit) (from Rows 8-15, Bytes 1,3,5...)
 * type 3: A11 (64-bit) | A11 (64-bit) (from Rows 8-15, Bytes 0,2,4...)
 */


/*
 * 专用 B 加载: 我们需要 [ B00 | B01 ] 这样的布局来一次性计算 C00 和 C01
 * B00: Rows 0-7, High Bytes
 * B01: Rows 0-7, Low Bytes
 * 
 * Target xmm: [ High Bytes | Low Bytes ]
 */


// 核心 GFNI 128 运算


/*
 * 方案3: MatMulMatM16_Opt3
 * 纯 128-bit 实现，适合无 AVX512 或为了降低功耗
 */


/*
 * 方案4: MatMulMatM16_Opt4
 * 极致 AVX-512 优化，使用专用的 512-bit 数据编排逻辑
 * 避免了 Opt2 中 "256 unpack -> broadcast -> 512 permute" 的冗余路径
 */


/*
 * 方案5: MatMulMatM16_Opt5 (AVX-512 Wide Parallel)
 * 核心思想：
 * 不再拆分 4 个独立的 8x8 乘法，而是利用 512 位寄存器宽度，
 * 将 B 视为宽矩阵，A 视为广播系数。
 * 
 * B 的布局策略 (Planar):
 * 我们将 B 的 Rows 0-7 加载并重排为: [ B00 (64bit) | B01 (64bit) | B00 | B01 ... ]
 * 这样一次 GFNI 乘法就能同时算出 C00 和 C01 的部分结果。
 */


/*
 * 方案6: MatMulMatM16_Opt6 (Reduction Parallel)
 * 对应你描述的新思路：
 * 利用 128-bit Lane 容纳 [Matrix_Part2 | Matrix_Part1]
 * 一次乘法后，在 Lane 内部进行 XOR 规约。
 * 
 * 4个 128-bit Lanes 分别负责最终的 C00, C01, C10, C11。
 */


/*
 * 方案7: MatMulMatM16_Opt7 (Streamlined Planar)
 * 核心思想：
 * 进一步精简 Opt5 的数据路径。
 * 1. B 矩阵一次性转置为 "All Top" 和 "All Bot" 形态，不再在计算中途 shuffle。
 * 2. A 矩阵直接以 Planar 格式广播，与 B 完美对应。
 * 3. 利用 ZMM 的海量寄存器文件，以空间换时间，减少指令依赖链。
 */


/*
 * 方案3 Plus: MatMulMatM16_Opt3_1
 * 基于 128-bit Opt3 的改进。
 * 优化点：
 * 1. 一次性加载整个 A 和 B 到 4 个 xmm 寄存器，减少重复的内存访问指令。
 * 2. 使用 _mm_shuffle_epi32 代替部分 _mm_shuffle_epi8，前者在某些架构上延迟更低。
 */


/*
 * MatMulMatM16_reg: 纯寄存器版本的矩阵矩阵乘法 (16x16 * 16x16)
 * - 基于 M8 row-direct GFNI 顺序，子块乘法直接产生行格式结果。
 * - 设计用于在 `genMatpair` 等内部循环中完全避免 结构体和内存访存开销。
 * - 接收4个__m128i参数 (左矩阵的上下半部、右矩阵的上下半部)，直接返回乘积的2个__m128i向量。
 */
void MatMulMatM16_reg(__m128i A0, __m128i A1, __m128i B0, __m128i B1, __m128i *C0, __m128i *C1) {
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

/*
 * MatMulVecM16_Opt1: Matrix-Vector Multiplication (16x16 * 16x1)
 * Based on MatMulMatM16_Opt3 logic (128-bit GFNI)
 */

