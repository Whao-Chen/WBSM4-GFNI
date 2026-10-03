#include "wbsm4_se_local.h"
#include "wbsm4_se_avx2.h"

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))

//! test
static void dump_hex(uint8_t * h, int len)
{
    while(len--)
    {   
        printf("%02hhx ",*h++);
        if(len%32==0) printf("\n");
    }
    printf("\n");
}

static Aff8 A[2039], B[2039];

void MatrixdivM32to4(M32 mat32, M4 m4[8][8])
{
    uint64_t m[2][4] = {
        {0xf0f0f0f0f0f0f0f0,0xf0f0f0f0f0f0f0f0,0xf0f0f0f0f0f0f0f0,0xf0f0f0f0f0f0f0f0},//取前4位
        {0x0f0f0f0f0f0f0f0f,0x0f0f0f0f0f0f0f0f,0x0f0f0f0f0f0f0f0f,0x0f0f0f0f0f0f0f0f},//取后4位
    };
    __m256i M0,mask[2],tmp,swap;
    uint8_t first4[32][4] = {0};
    uint8_t second4[32][4] = {0};
    mask[0] = _mm256_loadu_si256((const __m256i *)m[0]);
    mask[1] = _mm256_loadu_si256((const __m256i *)m[1]);
    swap = _mm256_setr_epi8(3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12,3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12);
    for (size_t i = 0; i < 4; i++)
    {
        M0 = _mm256_loadu_si256((const __m256i *)mat32.M + i);
        M0 = _mm256_shuffle_epi8(M0,swap);
        // printf("load:\n");
        // dump_hex(&M0,32);
        tmp = _mm256_and_si256(M0,mask[0]);
        tmp = _mm256_srli_epi32(tmp,4);
        _mm256_storeu_si256((__m256i*)first4 + i, tmp);

        tmp = _mm256_and_si256(M0,mask[1]);
        _mm256_storeu_si256((__m256i*)second4 + i, tmp);
    }
    
    //将first4、second4中的数据放进data
    for (size_t i = 0; i < 8; i++)
    {
        for (size_t j = 0; j < 8; j+=2)
        {
            m4[i][j].M[0] = first4[4 * i    ][j / 2];
            m4[i][j].M[1] = first4[4 * i + 1][j / 2];
            m4[i][j].M[2] = first4[4 * i + 2][j / 2];
            m4[i][j].M[3] = first4[4 * i + 3][j / 2];
            
            m4[i][j + 1].M[0] = second4[4 * i    ][j / 2];
            m4[i][j + 1].M[1] = second4[4 * i + 1][j / 2];
            m4[i][j + 1].M[2] = second4[4 * i + 2][j / 2];
            m4[i][j + 1].M[3] = second4[4 * i + 3][j / 2];
        }
    }
    
}

wbsm4_se_avx2_context *wbsm4_se_avx2_context_init(int encmode) {
    wbsm4_se_avx2_context *ctx = malloc(sizeof(wbsm4_se_avx2_context));
    if (ctx == NULL) {
        goto cleanup;
    }
    memset(ctx, 0, sizeof(wbsm4_se_avx2_context));
    ctx->encmode = encmode;
    return ctx;
cleanup:
    return NULL;
}

int se_gen(se_context *ctx, int encmode, const uint8_t *key) {
    int i;
    Aff32 P[36];
    Aff32 K[32];
    Aff32 L;
    Aff32 P_inv[36];
    Aff32 AA[32], BB[32];

    sm4_context *sm4_key = sm4_context_init();
    if (encmode) {
        sm4_setkey_enc(sm4_key, key);
    } else {
        sm4_setkey_dec(sm4_key, key);
    }
    
    InitRandom(((unsigned int) time(NULL)));
    InitRandom(1);

    SE_initial(A,B);

    //affine L for linear layer
    L.Mat = L_matrix;
    L.Vec.V = 0;

    //! 有问题
    for (i = 0; i < 36; i++) {
        //affine P
       // genaffinepairM32(&P[i], &P_inv[i]);
       identityM32(&P[i].Mat);
       identityM32(&P_inv[i].Mat);
       P[i].Vec.V = 0;
       P_inv[i].Vec.V = 0;
    }

    for (i = 0; i < 32; i++) {
        // combine 4 A8 to 1 A32
        int a0, a1, a2, a3;
        
        a0 = cus_random() % 2039;
        a1 = cus_random() % 2039;
        a2 = cus_random() % 2039;
        a3 = cus_random() % 2039;
        
        affinecomM8to32(A[a0], A[a1], A[a2], A[a3], &AA[i]);
        affinecomM8to32(B[a0], B[a1], B[a2], B[a3], &BB[i]);

        //affine K for round key
        identityM32(&K[i].Mat);
        K[i].Vec.V = sm4_key->sk[i];

        //affine M
        affinemixM32(K[i], P_inv[i + 1], &ctx->M[i][0]);
        affinemixM32(AA[i], ctx->M[i][0], &ctx->M[i][0]);

        affinemixM32(K[i], P_inv[i + 2], &ctx->M[i][1]);
        affinemixM32(AA[i], ctx->M[i][1], &ctx->M[i][1]);

        affinemixM32(K[i], P_inv[i + 3], &ctx->M[i][2]);
        affinemixM32(AA[i], ctx->M[i][2], &ctx->M[i][2]);

        //affine C D, C for Xi0, D for Pi+4 L B
        affinemixM32(P[i + 4], P_inv[i], &ctx->C[i]);

        affinemixM32(L, BB[i], &ctx->D[i]);
        affinemixM32(P[i + 4], ctx->D[i], &ctx->D[i]);

        //! 换成常数方便测试 
        uint32_t temp_u32 = cus_random();
      
        ctx->C[i].Vec.V ^= temp_u32;
        ctx->D[i].Vec.V ^= P[i + 4].Vec.V ^ temp_u32;
    }

    //external encoding
    for (i = 0; i < 4; i++) {
        ctx->SE[i].Mat = P[i].Mat;
        ctx->SE[i].Vec = P[i].Vec;

        ctx->FE[i].Mat = P_inv[35 - i].Mat;
        ctx->FE[i].Vec = P_inv[35 - i].Vec;
    }
    
    return 1;
}

void wbsm4_se_avx2_context_free(wbsm4_se_avx2_context *ctx) {
    memset(ctx, 0, sizeof(wbsm4_se_avx2_context));
}

uint8_t GetMatMulVecM4(M4 Mat, uint8_t Vec){
    V4 ret, tmp;
    tmp.V = Vec;
    MatMulVecM4(Mat, tmp, &ret);
    return ret.V;
}

void GEN_TABLE_DATA_F4(M4 TMP[8][8], uint8_t index, uint8_t data[4][4][16]) {
    //每4bit需要4-8查表4次，32bit中每8bit的前4bit共有4块数据，4*4
    //前 4bit
    data[0][0][index] = (GetMatMulVecM4(TMP[0][0], index) << 4) + GetMatMulVecM4(TMP[1][0], index);//TODO 顺序可能出错
    data[1][0][index] = (GetMatMulVecM4(TMP[0][2], index) << 4) + GetMatMulVecM4(TMP[1][2], index);
    data[2][0][index] = (GetMatMulVecM4(TMP[0][4], index) << 4) + GetMatMulVecM4(TMP[1][4], index);
    data[3][0][index] = (GetMatMulVecM4(TMP[0][6], index) << 4) + GetMatMulVecM4(TMP[1][6], index);

    data[0][1][index] = (GetMatMulVecM4(TMP[2][0], index) << 4) + GetMatMulVecM4(TMP[3][0], index);
    data[1][1][index] = (GetMatMulVecM4(TMP[2][2], index) << 4) + GetMatMulVecM4(TMP[3][2], index);
    data[2][1][index] = (GetMatMulVecM4(TMP[2][4], index) << 4) + GetMatMulVecM4(TMP[3][4], index);
    data[3][1][index] = (GetMatMulVecM4(TMP[2][6], index) << 4) + GetMatMulVecM4(TMP[3][6], index);

    data[0][2][index] = (GetMatMulVecM4(TMP[4][0], index) << 4) + GetMatMulVecM4(TMP[5][0], index);
    data[1][2][index] = (GetMatMulVecM4(TMP[4][2], index) << 4) + GetMatMulVecM4(TMP[5][2], index);
    data[2][2][index] = (GetMatMulVecM4(TMP[4][4], index) << 4) + GetMatMulVecM4(TMP[5][4], index);
    data[3][2][index] = (GetMatMulVecM4(TMP[4][6], index) << 4) + GetMatMulVecM4(TMP[5][6], index);

    data[0][3][index] = (GetMatMulVecM4(TMP[6][0], index) << 4) + GetMatMulVecM4(TMP[7][0], index);
    data[1][3][index] = (GetMatMulVecM4(TMP[6][2], index) << 4) + GetMatMulVecM4(TMP[7][2], index);
    data[2][3][index] = (GetMatMulVecM4(TMP[6][4], index) << 4) + GetMatMulVecM4(TMP[7][4], index);
    data[3][3][index] = (GetMatMulVecM4(TMP[6][6], index) << 4) + GetMatMulVecM4(TMP[7][6], index);
    // 加密时的查表顺序是第一个4bit 从data[0][0] -> data[0][3] 第二个4bit 从data[1][0] -> data[1][3]
}
void GEN_TABLE_DATA_E4(M4 TMP[8][8], uint8_t index, uint8_t data[4][4][16]) {
    //每4bit需要4-8查表4次，32bit中每8bit的前4bit共有4块数据，4*4
    //后 4bit
    data[0][0][index] = (GetMatMulVecM4(TMP[0][1], index) << 4) + GetMatMulVecM4(TMP[1][1], index);
    data[1][0][index] = (GetMatMulVecM4(TMP[0][3], index) << 4) + GetMatMulVecM4(TMP[1][3], index);
    data[2][0][index] = (GetMatMulVecM4(TMP[0][5], index) << 4) + GetMatMulVecM4(TMP[1][5], index);
    data[3][0][index] = (GetMatMulVecM4(TMP[0][7], index) << 4) + GetMatMulVecM4(TMP[1][7], index);

    data[0][1][index] = (GetMatMulVecM4(TMP[2][1], index) << 4) + GetMatMulVecM4(TMP[3][1], index);
    data[1][1][index] = (GetMatMulVecM4(TMP[2][3], index) << 4) + GetMatMulVecM4(TMP[3][3], index);
    data[2][1][index] = (GetMatMulVecM4(TMP[2][5], index) << 4) + GetMatMulVecM4(TMP[3][5], index);
    data[3][1][index] = (GetMatMulVecM4(TMP[2][7], index) << 4) + GetMatMulVecM4(TMP[3][7], index);

    data[0][2][index] = (GetMatMulVecM4(TMP[4][1], index) << 4) + GetMatMulVecM4(TMP[5][1], index);
    data[1][2][index] = (GetMatMulVecM4(TMP[4][3], index) << 4) + GetMatMulVecM4(TMP[5][3], index);
    data[2][2][index] = (GetMatMulVecM4(TMP[4][5], index) << 4) + GetMatMulVecM4(TMP[5][5], index);
    data[3][2][index] = (GetMatMulVecM4(TMP[4][7], index) << 4) + GetMatMulVecM4(TMP[5][7], index);

    data[0][3][index] = (GetMatMulVecM4(TMP[6][1], index) << 4) + GetMatMulVecM4(TMP[7][1], index);
    data[1][3][index] = (GetMatMulVecM4(TMP[6][3], index) << 4) + GetMatMulVecM4(TMP[7][3], index);
    data[2][3][index] = (GetMatMulVecM4(TMP[6][5], index) << 4) + GetMatMulVecM4(TMP[7][5], index);
    data[3][3][index] = (GetMatMulVecM4(TMP[6][7], index) << 4) + GetMatMulVecM4(TMP[7][7], index);
}

int wbsm4_se_avx2_gen_table(wbsm4_se_avx2_context *ctx, const uint8_t *key) {
    //原始wbsme_se_context中的矩阵
    se_context se_ctx;
    se_gen(&se_ctx,ctx->encmode,key);
    uint8_t i, j, x, tmp_u8 = 0;

    M4 M0_TMP[8][8];
    // printf("test:\n");
    // MatrixdivM32to4(se_ctx.M[0][0].Mat, M0_TMP);
    // for (size_t i = 0; i < 32; i++)
    // {
    //     printf("%08x \n",se_ctx.M[0][0].Mat.M[i]);
    // }
    // printf("\n");
    // for (size_t i = 0; i < 4; i++)
    // {
    //     printf("%02x ", M0_TMP[1][0].M[i]);
    // }
    // for (size_t i = 0; i < 4; i++)
    // {
    //     printf("%02x ", M0_TMP[1][1].M[i]);
    // }
    // for (size_t i = 0; i < 4; i++)
    // {
    //     printf("%02x ", M0_TMP[1][2].M[i]);
    // }
    // printf("test end \n");

    for(i = 0; i < 32; i++)//32轮M0，M1, M2, C, D等
	{
        M4 M0_TMP[8][8];
        M4 M1_TMP[8][8];
        M4 M2_TMP[8][8];
        M4 C_TMP[8][8];
        M4 D_TMP[8][8];//TODO 还差SE和FE

        // M32->M4[8][8]
 
        MatrixdivM32to4(se_ctx.M[i][0].Mat, M0_TMP);//TODO 测试是否能正常传值
        MatrixdivM32to4(se_ctx.M[i][1].Mat, M1_TMP);
        MatrixdivM32to4(se_ctx.M[i][2].Mat, M2_TMP);
        MatrixdivM32to4(se_ctx.C[i].Mat, C_TMP);
        MatrixdivM32to4(se_ctx.D[i].Mat, D_TMP);
        
        PUT32(se_ctx.M[i][0].Vec.V, ctx->M0_V[i]);
        PUT32(se_ctx.M[i][1].Vec.V, ctx->M1_V[i]);
        PUT32(se_ctx.M[i][2].Vec.V, ctx->M2_V[i]);
        PUT32(se_ctx.C[i].Vec.V, ctx->C_V[i]);
        PUT32(se_ctx.D[i].Vec.V, ctx->D_V[i]);

        for (x = 0; x < 16; x++)// 4bit的遍历
        {
            //每4bit需要4-8查表4次，32bit中每8bit的前4bit共有4块数据，4*4
			GEN_TABLE_DATA_F4(M0_TMP,x,ctx->M0_F4[i]);
            GEN_TABLE_DATA_F4(M1_TMP,x,ctx->M1_F4[i]);
            GEN_TABLE_DATA_F4(M2_TMP,x,ctx->M2_F4[i]);
            GEN_TABLE_DATA_F4(C_TMP,x,ctx->C_F4[i]);
            GEN_TABLE_DATA_F4(D_TMP,x,ctx->D_F4[i]);

            GEN_TABLE_DATA_E4(M0_TMP,x,ctx->M0_E4[i]);
            GEN_TABLE_DATA_E4(M1_TMP,x,ctx->M1_E4[i]);
            GEN_TABLE_DATA_E4(M2_TMP,x,ctx->M2_E4[i]);
            GEN_TABLE_DATA_E4(C_TMP,x,ctx->C_E4[i]);
            GEN_TABLE_DATA_E4(D_TMP,x,ctx->D_E4[i]);
        }
	}
    for (size_t i = 0; i < 4; i++)
    {
        M4 SE_TMP[8][8];
        M4 FE_TMP[8][8];
        
        PUT32(se_ctx.SE[i].Vec.V, ctx->SE_V[i]);
        PUT32(se_ctx.FE[i].Vec.V, ctx->FE_V[i]);

        MatrixdivM32to4(se_ctx.SE[i].Mat, SE_TMP);//TODO 测试是否能正常传值
        MatrixdivM32to4(se_ctx.FE[i].Mat, FE_TMP);
        for (x = 0; x < 16; x++)// 4bit的遍历
        {
            //每4bit需要4-8查表4次，32bit中每8bit的前4bit共有4块数据，4*4
            GEN_TABLE_DATA_F4(SE_TMP,x,ctx->SE_F4[i]);
            GEN_TABLE_DATA_F4(FE_TMP,x,ctx->FE_F4[i]);
            GEN_TABLE_DATA_E4(SE_TMP,x,ctx->SE_E4[i]);
            GEN_TABLE_DATA_E4(FE_TMP,x,ctx->FE_E4[i]);
        }

    }
  
    return 1;
}
// !这里面的某个值没初始化
void AFFINE_ENCODING(__m256i IN[4], __m256i OUT[4], uint8_t table_f4[4][4][16], uint8_t table_e4[4][4][16], uint8_t table_vec[4]) {
        __m256i input_f4[4], input_e4[4];
        __m256i tmp_f4,tmp_e4;
        __m128i tmp_128_mat;// 表的大小只有128
        __m256i tmp_mat;
        __m256i tmp_vec;
        __m256i mask[2];
        uint64_t m[2][4] = {
            {0xf0f0f0f0f0f0f0f0,0xf0f0f0f0f0f0f0f0,0xf0f0f0f0f0f0f0f0,0xf0f0f0f0f0f0f0f0},//取前4位
            {0x0f0f0f0f0f0f0f0f,0x0f0f0f0f0f0f0f0f,0x0f0f0f0f0f0f0f0f,0x0f0f0f0f0f0f0f0f},//取后4位
        };
        mask[0] = _mm256_loadu_si256((__m256i *)m[0]);// todo 测试一下预先加载会不会更快
        mask[1] = _mm256_loadu_si256((__m256i *)m[1]);
        for (size_t i = 0; i < 4; i++)//从明文中取4bit
        {
            input_f4[i] = _mm256_and_si256(mask[0],IN[i]);
            input_f4[i] = _mm256_srli_epi32(input_f4[i],4);
            input_e4[i] = _mm256_and_si256(mask[1],IN[i]);
        }
        

        for (size_t i = 0; i < 4; i++)//32bit分成4组8bit，每个8bit的前4bit
        {
            // 以第一个8bit为例，即input_space[0]，其值等于A(0,0) * x0 + A(0,1) * x1 + A(0,2) * x2 + A(0,3) * x3 涉及到32bit的其他部分
            
            //todo 看是否需要抽成一个函数
            // A(0,0) * x0
            tmp_128_mat = _mm_loadu_si128(table_f4[0][i]);//todo 表装载，直接在ctx存放已经复制了的数据会不会更快
            tmp_mat = _mm256_setr_m128i(tmp_128_mat, tmp_128_mat);
            tmp_f4 = _mm256_shuffle_epi8(tmp_mat, input_f4[0]);// 4-8查表
            // if (i == 0)
            // {
            //     //! 这个表不对
            //     printf("table_f4:");
            //     dump_hex(table_f4[i][0],16);
              
            //     printf("tmp_MAT:");
            //     dump_hex(&tmp_mat,32);
            // }
            tmp_128_mat = _mm_loadu_si128(table_e4[0][i]);
            tmp_mat = _mm256_setr_m128i(tmp_128_mat, tmp_128_mat);
            tmp_e4 = _mm256_shuffle_epi8(tmp_mat, input_e4[0]);// 4-8查表
            
            OUT[i] = tmp_e4 ^ tmp_f4;//两个4-8查表异或起来，放进中间值暂存
            // if (i == 0)
            // {
            //     printf("tmp_MAT:");
            //     dump_hex(&tmp_mat,32);
            //     printf("tmp_f4_x0:");
            //     dump_hex(&tmp_f4,32);
            //     printf("tmp_e4:");
            //     dump_hex(&tmp_e4,32);
            //     printf("out:");
            //     dump_hex(OUT+i,32);
            // }
            // A(0,1) * x1
            tmp_128_mat = _mm_loadu_si128(table_f4[1][i]);
            tmp_mat = _mm256_setr_m128i(tmp_128_mat, tmp_128_mat);
            tmp_f4 = _mm256_shuffle_epi8(tmp_mat,input_f4[1]);// 4-8查表

            tmp_128_mat = _mm_loadu_si128(table_e4[1][i]);
            tmp_mat = _mm256_setr_m128i(tmp_128_mat, tmp_128_mat);
            tmp_e4 = _mm256_shuffle_epi8(tmp_mat,input_e4[1]);// 4-8查表
            
            // if (i == 0)
            // {
            //     printf("tmp_f4_x1:");
            //     dump_hex(&tmp_f4,32);
            //     printf("tmp_e4:");
            //     dump_hex(&tmp_e4,32);
            // }
            OUT[i] = OUT[i] ^ tmp_e4 ^ tmp_f4;//两个4-8查表异或起来

            // A(0,2) * x2
            tmp_128_mat = _mm_loadu_si128(table_f4[2][i]);
            tmp_mat = _mm256_setr_m128i(tmp_128_mat, tmp_128_mat);
            tmp_f4 = _mm256_shuffle_epi8(tmp_mat,input_f4[2]);// 4-8查表

            tmp_128_mat = _mm_loadu_si128(table_e4[2][i]);
            tmp_mat = _mm256_setr_m128i(tmp_128_mat, tmp_128_mat);
            tmp_e4 = _mm256_shuffle_epi8(tmp_mat,input_e4[2]);// 4-8查表
            // if (i == 0)
            // {
            //     printf("tmp_f4_x2:");
            //     dump_hex(&tmp_f4,32);
            //     printf("tmp_e4:");
            //     dump_hex(&tmp_e4,32);
            // }
            OUT[i] = OUT[i] ^ tmp_e4 ^ tmp_f4;//两个4-8查表异或起来

            // A(0,3) * x3
            tmp_128_mat = _mm_loadu_si128(table_f4[3][i]);
            tmp_mat = _mm256_setr_m128i(tmp_128_mat, tmp_128_mat);
            tmp_f4 = _mm256_shuffle_epi8(tmp_mat,input_f4[3]);// 4-8查表

            tmp_128_mat = _mm_loadu_si128(table_e4[3][i]);
            tmp_mat = _mm256_setr_m128i(tmp_128_mat, tmp_128_mat);
            tmp_e4 = _mm256_shuffle_epi8(tmp_mat,input_e4[3]);// 4-8查表
            // if (i == 0)
            // {
            //     printf("tmp_f4_x3:");
            //     dump_hex(&tmp_f4,32);
            //     printf("tmp_e4:");
            //     dump_hex(&tmp_e4,32);
            // }
            OUT[i] = OUT[i] ^ tmp_e4 ^ tmp_f4;//两个4-8查表异或起来
        }
        
        for (size_t i = 0; i < 4; i++)
        {
            tmp_vec = _mm256_set1_epi8(table_vec[i]);
            OUT[i] = OUT[i] ^ tmp_vec;
        }
        // dump_hex(OUT,128);

}

void DATA_PACK( __m256i OUT[16]){
    // printf("\npack input:\n");
    // dump_hex(OUT,512);
    __m256i state[16];
    for(int i=0;i<8;i++){
            state[i] = _mm256_unpacklo_epi8(OUT[2*i],OUT[2*i+1]);
            state[i+8] = _mm256_unpackhi_epi8(OUT[2*i],OUT[2*i+1]);
        }
        // printf("\nOUT-after:\n");
        // dump_hex(state,256);
        OUT[0] = _mm256_unpacklo_epi16(state[0],state[1]);
        OUT[1] = _mm256_unpacklo_epi16(state[2],state[3]);
        OUT[2] = _mm256_unpacklo_epi16(state[4],state[5]);
        OUT[3] = _mm256_unpacklo_epi16(state[6],state[7]);

        OUT[4] = _mm256_unpackhi_epi16(state[0],state[1]);
        OUT[5] = _mm256_unpackhi_epi16(state[2],state[3]);
        OUT[6] = _mm256_unpackhi_epi16(state[4],state[5]);
        OUT[7] = _mm256_unpackhi_epi16(state[6],state[7]);
        // printf("\nOUT(对lo进行操作):\n");
        // dump_hex(OUT,128);  

        OUT[8] = _mm256_unpacklo_epi16(state[8],state[9]);
        OUT[9] = _mm256_unpacklo_epi16(state[10],state[11]);
        OUT[10] = _mm256_unpacklo_epi16(state[12],state[13]);
        OUT[11] = _mm256_unpacklo_epi16(state[14],state[15]);

        OUT[12] = _mm256_unpackhi_epi16(state[8],state[9]);
        OUT[13] = _mm256_unpackhi_epi16(state[10],state[11]);
        OUT[14] = _mm256_unpackhi_epi16(state[12],state[13]);
        OUT[15] = _mm256_unpackhi_epi16(state[14],state[15]);  
        // printf("\nOUT(对hi进行操作):\n");
        // dump_hex(OUT+8,128);  


        state[0] = _mm256_unpacklo_epi32(OUT[0],OUT[1]);//0-1
        state[1] = _mm256_unpacklo_epi32(OUT[2],OUT[3]);
        state[2] = _mm256_unpacklo_epi32(OUT[4],OUT[5]);//4-5
        state[3] = _mm256_unpacklo_epi32(OUT[6],OUT[7]);
        
        state[4] = _mm256_unpackhi_epi32(OUT[0],OUT[1]);//2-3
        state[5] = _mm256_unpackhi_epi32(OUT[2],OUT[3]);
        state[6] = _mm256_unpackhi_epi32(OUT[4],OUT[5]);//6-7
        state[7] = _mm256_unpackhi_epi32(OUT[6],OUT[7]);
        // printf("\nstate:\n");
        // dump_hex(state,128); 

        state[8] = _mm256_unpacklo_epi32(OUT[8],OUT[9]);//0-1
        state[9] = _mm256_unpacklo_epi32(OUT[10],OUT[11]);
        state[10] = _mm256_unpacklo_epi32(OUT[12],OUT[13]);//4-5
        state[11] = _mm256_unpacklo_epi32(OUT[14],OUT[15]);
        
        state[12] = _mm256_unpackhi_epi32(OUT[8],OUT[9]);//2-3
        state[13] = _mm256_unpackhi_epi32(OUT[10],OUT[11]);
        state[14] = _mm256_unpackhi_epi32(OUT[12],OUT[13]);//6-7
        state[15] = _mm256_unpackhi_epi32(OUT[14],OUT[15]);
        // printf("\nstate:\n");
        // dump_hex(state+8,128);  
        
        OUT[0] = _mm256_unpacklo_epi64(state[0],state[1]);
        OUT[1] = _mm256_unpackhi_epi64(state[0],state[1]);

        OUT[2] = _mm256_unpacklo_epi64(state[4],state[5]);
        OUT[3] = _mm256_unpackhi_epi64(state[4],state[5]);

        OUT[4] = _mm256_unpacklo_epi64(state[2],state[3]);
        OUT[5] = _mm256_unpackhi_epi64(state[2],state[3]);

        OUT[6] = _mm256_unpacklo_epi64(state[6],state[7]);
        OUT[7] = _mm256_unpackhi_epi64(state[6],state[7]);


        OUT[8] = _mm256_unpacklo_epi64(state[8],state[9]);
        OUT[9] = _mm256_unpackhi_epi64(state[8],state[9]);

        OUT[10] = _mm256_unpacklo_epi64(state[12],state[13]);
        OUT[11] = _mm256_unpackhi_epi64(state[12],state[13]);

        OUT[12] = _mm256_unpacklo_epi64(state[10],state[11]);
        OUT[13] = _mm256_unpackhi_epi64(state[10],state[11]);

        OUT[14] = _mm256_unpacklo_epi64(state[14],state[15]);
        OUT[15] = _mm256_unpackhi_epi64(state[14],state[15]);
//         printf("\npack output:\n");
//         dump_hex(OUT,512);
}

int wbsm4_se_avx2_encrypt(const uint8_t IN[512], uint8_t OUT[512], wbsm4_se_avx2_context *ctx)// todo 是否传这个ctx
    {
        __m256i state1[16];
        __m256i state2[16];
        __m256i tmp[4];
        __m256i mask;

        mask = _mm256_set1_epi32(0xFF);
        memcpy(state1, IN, 512);
        DATA_PACK(state1);

        // dump_hex(state1,512);
        // x0 = affineU32(ctx->SE[0], x0);
        AFFINE_ENCODING(state1     , state1     , ctx->SE_F4[0], ctx->SE_E4[0], ctx->SE_V[0]);
        // x1 = affineU32(ctx->SE[1], x1);
        AFFINE_ENCODING(state1 +  4, state1 +  4, ctx->SE_F4[1], ctx->SE_E4[1], ctx->SE_V[1]);
        // x2 = affineU32(ctx->SE[2], x2);
        AFFINE_ENCODING(state1 +  8, state1 +  8, ctx->SE_F4[2], ctx->SE_E4[2], ctx->SE_V[2]);
        // x3 = affineU32(ctx->SE[3], x3);
        AFFINE_ENCODING(state1 + 12, state1 + 12, ctx->SE_F4[3], ctx->SE_E4[3], ctx->SE_V[3]);
        // printf("test test\n");
        // dump_hex(state1,512);

        
        for (size_t i = 0; i < 32; i+=4)
        {
            // 这里的state1是x 而state2是xt
            //round1
            
            AFFINE_ENCODING(state1 +  4, state2 +  4, ctx->M0_F4[i], ctx->M0_E4[i], ctx->M0_V[i]);
            AFFINE_ENCODING(state1 +  8, state2 +  8, ctx->M1_F4[i], ctx->M1_E4[i], ctx->M1_V[i]);
            AFFINE_ENCODING(state1 + 12, state2 + 12, ctx->M2_F4[i], ctx->M2_E4[i], ctx->M2_V[i]);     
            AFFINE_ENCODING(state1     , state2     , ctx->C_F4[i], ctx->C_E4[i], ctx->C_V[i]);
  
            tmp[0] = state2[4] ^ state2[8] ^ state2[12];
            tmp[1] = state2[5] ^ state2[9] ^ state2[13];
            tmp[2] = state2[6] ^ state2[10] ^ state2[14];
            tmp[3] = state2[7] ^ state2[11] ^ state2[15];
            // printf("input:\n");
            // dump_hex(tmp,128);
            //todo sbox gather是对每32bit的低8bit作为序列对256个32bit'进行查表  需要抽离成一个函数？
            tmp[0] = _mm256_i32gather_epi32((const int*)T0, _mm256_and_si256(mask, tmp[0]), 4)
                    ^ _mm256_i32gather_epi32((const int*)T1, _mm256_and_si256(mask, _mm256_srli_epi32(tmp[0], 8)), 4)
                    ^ _mm256_i32gather_epi32((const int*)T2, _mm256_and_si256(mask, _mm256_srli_epi32(tmp[0], 16)), 4) 
                    ^ _mm256_i32gather_epi32((const int*)T3, _mm256_and_si256(mask, _mm256_srli_epi32(tmp[0], 24)), 4);

            tmp[1] = _mm256_i32gather_epi32((const int*)T0, _mm256_and_si256(mask, tmp[1]), 4)
                    ^ _mm256_i32gather_epi32((const int*)T1, _mm256_and_si256(mask, _mm256_srli_epi32(tmp[1], 8)), 4)
                    ^ _mm256_i32gather_epi32((const int*)T2, _mm256_and_si256(mask, _mm256_srli_epi32(tmp[1], 16)), 4) 
                    ^ _mm256_i32gather_epi32((const int*)T3, _mm256_and_si256(mask, _mm256_srli_epi32(tmp[1], 24)), 4);

            tmp[2] = _mm256_i32gather_epi32((const int*)T0, _mm256_and_si256(mask, tmp[2]), 4)
                    ^ _mm256_i32gather_epi32((const int*)T1, _mm256_and_si256(mask, _mm256_srli_epi32(tmp[2], 8)), 4)
                    ^ _mm256_i32gather_epi32((const int*)T2, _mm256_and_si256(mask, _mm256_srli_epi32(tmp[2], 16)), 4) 
                    ^ _mm256_i32gather_epi32((const int*)T3, _mm256_and_si256(mask, _mm256_srli_epi32(tmp[2], 24)), 4);

            tmp[3] = _mm256_i32gather_epi32((const int*)T0, _mm256_and_si256(mask, tmp[3]), 4)
                    ^ _mm256_i32gather_epi32((const int*)T1, _mm256_and_si256(mask, _mm256_srli_epi32(tmp[3], 8)), 4)
                    ^ _mm256_i32gather_epi32((const int*)T2, _mm256_and_si256(mask, _mm256_srli_epi32(tmp[3], 16)), 4) 
                    ^ _mm256_i32gather_epi32((const int*)T3, _mm256_and_si256(mask, _mm256_srli_epi32(tmp[3], 24)), 4);
            // printf("input:\n");
            // dump_hex(tmp,128);
            // ! D的表有问题
            // __m256i tmp1[4];
            // for (size_t y = 0; y < 4; y++)
            // {
            //     tmp1[y] = _mm256_set1_epi32(0);
            // }
            // printf("input:\n");
            // dump_hex(tmp,128);
            AFFINE_ENCODING(tmp, tmp, ctx->D_F4[i], ctx->D_E4[i], ctx->D_V[i]);
            
            // __m256i test[4];
            // test[0] = _mm256_set1_epi32(0x3c3c3c3c);
            // test[1] = _mm256_set1_epi32(0x00000000);
            // test[2] = _mm256_set1_epi32(0x00000000);
            // test[3] = _mm256_set1_epi32(0x00000000);
            // printf("affine test:\n");
            // dump_hex(test,128);
            // AFFINE_ENCODING(test, test, ctx->D_F4[0], ctx->D_E4[0], ctx->D_V[0]);
            // dump_hex(test,128);

            state1[0] = state2[0] ^ tmp[0];
            state1[1] = state2[1] ^ tmp[1];
            state1[2] = state2[2] ^ tmp[2];
            state1[3] = state2[3] ^ tmp[3];
         
            //下面的是接下来的3轮，将4轮放在一起约减了32bit赋值的消耗    
            //round2 
            // todo 是否需要将其抽离成函数       
            AFFINE_ENCODING(state1     , state2     , ctx->M2_F4[i + 1], ctx->M2_E4[i + 1], ctx->M2_V[i + 1]);
            AFFINE_ENCODING(state1 +  8, state2 +  8, ctx->M0_F4[i + 1], ctx->M0_E4[i + 1], ctx->M0_V[i + 1]);
            AFFINE_ENCODING(state1 + 12, state2 + 12, ctx->M1_F4[i + 1], ctx->M1_E4[i + 1], ctx->M1_V[i + 1]);
            
            AFFINE_ENCODING(state1 +  4, state2 +  4, ctx->C_F4[i + 1], ctx->C_E4[i + 1], ctx->C_V[i + 1]);
          
            tmp[0] = state2[0] ^ state2[8] ^ state2[12];
            tmp[1] = state2[1] ^ state2[9] ^ state2[13];
            tmp[2] = state2[2] ^ state2[10] ^ state2[14];
            tmp[3] = state2[3] ^ state2[11] ^ state2[15];

            tmp[0] = _mm256_i32gather_epi32((const int*)T0, _mm256_and_si256(mask, tmp[0]), 4)
                    ^ _mm256_i32gather_epi32((const int*)T1, _mm256_and_si256(mask, _mm256_srli_epi32(tmp[0], 8)), 4)
                    ^ _mm256_i32gather_epi32((const int*)T2, _mm256_and_si256(mask, _mm256_srli_epi32(tmp[0], 16)), 4) 
                    ^ _mm256_i32gather_epi32((const int*)T3, _mm256_and_si256(mask, _mm256_srli_epi32(tmp[0], 24)), 4);

            tmp[1] = _mm256_i32gather_epi32((const int*)T0, _mm256_and_si256(mask, tmp[1]), 4)
                    ^ _mm256_i32gather_epi32((const int*)T1, _mm256_and_si256(mask, _mm256_srli_epi32(tmp[1], 8)), 4)
                    ^ _mm256_i32gather_epi32((const int*)T2, _mm256_and_si256(mask, _mm256_srli_epi32(tmp[1], 16)), 4) 
                    ^ _mm256_i32gather_epi32((const int*)T3, _mm256_and_si256(mask, _mm256_srli_epi32(tmp[1], 24)), 4);

            tmp[2] = _mm256_i32gather_epi32((const int*)T0, _mm256_and_si256(mask, tmp[2]), 4)
                    ^ _mm256_i32gather_epi32((const int*)T1, _mm256_and_si256(mask, _mm256_srli_epi32(tmp[2], 8)), 4)
                    ^ _mm256_i32gather_epi32((const int*)T2, _mm256_and_si256(mask, _mm256_srli_epi32(tmp[2], 16)), 4) 
                    ^ _mm256_i32gather_epi32((const int*)T3, _mm256_and_si256(mask, _mm256_srli_epi32(tmp[2], 24)), 4);

            tmp[3] = _mm256_i32gather_epi32((const int*)T0, _mm256_and_si256(mask, tmp[3]), 4)
                    ^ _mm256_i32gather_epi32((const int*)T1, _mm256_and_si256(mask, _mm256_srli_epi32(tmp[3], 8)), 4)
                    ^ _mm256_i32gather_epi32((const int*)T2, _mm256_and_si256(mask, _mm256_srli_epi32(tmp[3], 16)), 4) 
                    ^ _mm256_i32gather_epi32((const int*)T3, _mm256_and_si256(mask, _mm256_srli_epi32(tmp[3], 24)), 4);
            
            AFFINE_ENCODING(tmp, tmp, ctx->D_F4[i + 1], ctx->D_E4[i + 1], ctx->D_V[i + 1]);

            state1[4] = state2[4] ^ tmp[0];
            state1[5] = state2[5] ^ tmp[1];
            state1[6] = state2[6] ^ tmp[2];
            state1[7] = state2[7] ^ tmp[3];
            
            //round3
            AFFINE_ENCODING(state1     , state2     , ctx->M1_F4[i + 2], ctx->M1_E4[i + 2], ctx->M1_V[i + 2]);
            AFFINE_ENCODING(state1 +  4, state2 +  4, ctx->M2_F4[i + 2], ctx->M2_E4[i + 2], ctx->M2_V[i + 2]);
            AFFINE_ENCODING(state1 + 12, state2 + 12, ctx->M0_F4[i + 2], ctx->M0_E4[i + 2], ctx->M0_V[i + 2]);
            
            AFFINE_ENCODING(state1 +  8, state2 +  8, ctx->C_F4[i + 2], ctx->C_E4[i + 2], ctx->C_V[i + 2]);
          
            tmp[0] = state2[0] ^ state2[4] ^ state2[12];
            tmp[1] = state2[1] ^ state2[5] ^ state2[13];
            tmp[2] = state2[2] ^ state2[6] ^ state2[14];
            tmp[3] = state2[3] ^ state2[7] ^ state2[15];

            tmp[0] = _mm256_i32gather_epi32((const int*)T0, _mm256_and_si256(mask, tmp[0]), 4)
                    ^ _mm256_i32gather_epi32((const int*)T1, _mm256_and_si256(mask, _mm256_srli_epi32(tmp[0], 8)), 4)
                    ^ _mm256_i32gather_epi32((const int*)T2, _mm256_and_si256(mask, _mm256_srli_epi32(tmp[0], 16)), 4) 
                    ^ _mm256_i32gather_epi32((const int*)T3, _mm256_and_si256(mask, _mm256_srli_epi32(tmp[0], 24)), 4);
                    
            tmp[1] = _mm256_i32gather_epi32((const int*)T0, _mm256_and_si256(mask, tmp[1]), 4)
                    ^ _mm256_i32gather_epi32((const int*)T1, _mm256_and_si256(mask, _mm256_srli_epi32(tmp[1], 8)), 4)
                    ^ _mm256_i32gather_epi32((const int*)T2, _mm256_and_si256(mask, _mm256_srli_epi32(tmp[1], 16)), 4) 
                    ^ _mm256_i32gather_epi32((const int*)T3, _mm256_and_si256(mask, _mm256_srli_epi32(tmp[1], 24)), 4);

            tmp[2] = _mm256_i32gather_epi32((const int*)T0, _mm256_and_si256(mask, tmp[2]), 4)
                    ^ _mm256_i32gather_epi32((const int*)T1, _mm256_and_si256(mask, _mm256_srli_epi32(tmp[2], 8)), 4)
                    ^ _mm256_i32gather_epi32((const int*)T2, _mm256_and_si256(mask, _mm256_srli_epi32(tmp[2], 16)), 4) 
                    ^ _mm256_i32gather_epi32((const int*)T3, _mm256_and_si256(mask, _mm256_srli_epi32(tmp[2], 24)), 4);

            tmp[3] = _mm256_i32gather_epi32((const int*)T0, _mm256_and_si256(mask, tmp[3]), 4)
                    ^ _mm256_i32gather_epi32((const int*)T1, _mm256_and_si256(mask, _mm256_srli_epi32(tmp[3], 8)), 4)
                    ^ _mm256_i32gather_epi32((const int*)T2, _mm256_and_si256(mask, _mm256_srli_epi32(tmp[3], 16)), 4) 
                    ^ _mm256_i32gather_epi32((const int*)T3, _mm256_and_si256(mask, _mm256_srli_epi32(tmp[3], 24)), 4);
            
            AFFINE_ENCODING(tmp, tmp, ctx->D_F4[i + 2], ctx->D_E4[i + 2], ctx->D_V[i + 2]);

            state1[8] = state2[8] ^ tmp[0];
            state1[9] = state2[9] ^ tmp[1];
            state1[10] = state2[10] ^ tmp[2];
            state1[11] = state2[11] ^ tmp[3];

            //round4
            AFFINE_ENCODING(state1     , state2     , ctx->M0_F4[i + 3], ctx->M0_E4[i + 3], ctx->M0_V[i + 3]);
            AFFINE_ENCODING(state1 +  4, state2 +  4, ctx->M1_F4[i + 3], ctx->M1_E4[i + 3], ctx->M1_V[i + 3]);
            AFFINE_ENCODING(state1 +  8, state2 +  8, ctx->M2_F4[i + 3], ctx->M2_E4[i + 3], ctx->M2_V[i + 3]);
            
            AFFINE_ENCODING(state1 + 12, state2 + 12, ctx->C_F4[i + 3], ctx->C_E4[i + 3], ctx->C_V[i + 3]);
          
            tmp[0] = state2[0] ^ state2[4] ^ state2[8];
            tmp[1] = state2[1] ^ state2[5] ^ state2[9];
            tmp[2] = state2[2] ^ state2[6] ^ state2[10];
            tmp[3] = state2[3] ^ state2[7] ^ state2[11];

            tmp[0] = _mm256_i32gather_epi32((const int*)T0, _mm256_and_si256(mask, tmp[0]), 4)
                    ^ _mm256_i32gather_epi32((const int*)T1, _mm256_and_si256(mask, _mm256_srli_epi32(tmp[0], 8)), 4)
                    ^ _mm256_i32gather_epi32((const int*)T2, _mm256_and_si256(mask, _mm256_srli_epi32(tmp[0], 16)), 4) 
                    ^ _mm256_i32gather_epi32((const int*)T3, _mm256_and_si256(mask, _mm256_srli_epi32(tmp[0], 24)), 4);
                    
            tmp[1] = _mm256_i32gather_epi32((const int*)T0, _mm256_and_si256(mask, tmp[1]), 4)
                    ^ _mm256_i32gather_epi32((const int*)T1, _mm256_and_si256(mask, _mm256_srli_epi32(tmp[1], 8)), 4)
                    ^ _mm256_i32gather_epi32((const int*)T2, _mm256_and_si256(mask, _mm256_srli_epi32(tmp[1], 16)), 4) 
                    ^ _mm256_i32gather_epi32((const int*)T3, _mm256_and_si256(mask, _mm256_srli_epi32(tmp[1], 24)), 4);

            tmp[2] = _mm256_i32gather_epi32((const int*)T0, _mm256_and_si256(mask, tmp[2]), 4)
                    ^ _mm256_i32gather_epi32((const int*)T1, _mm256_and_si256(mask, _mm256_srli_epi32(tmp[2], 8)), 4)
                    ^ _mm256_i32gather_epi32((const int*)T2, _mm256_and_si256(mask, _mm256_srli_epi32(tmp[2], 16)), 4) 
                    ^ _mm256_i32gather_epi32((const int*)T3, _mm256_and_si256(mask, _mm256_srli_epi32(tmp[2], 24)), 4);

            tmp[3] = _mm256_i32gather_epi32((const int*)T0, _mm256_and_si256(mask, tmp[3]), 4)
                    ^ _mm256_i32gather_epi32((const int*)T1, _mm256_and_si256(mask, _mm256_srli_epi32(tmp[3], 8)), 4)
                    ^ _mm256_i32gather_epi32((const int*)T2, _mm256_and_si256(mask, _mm256_srli_epi32(tmp[3], 16)), 4) 
                    ^ _mm256_i32gather_epi32((const int*)T3, _mm256_and_si256(mask, _mm256_srli_epi32(tmp[3], 24)), 4);
            
            AFFINE_ENCODING(tmp, tmp, ctx->D_F4[i + 3], ctx->D_E4[i + 3], ctx->D_V[i + 3]);

            state1[12] = state2[12] ^ tmp[0];
            state1[13] = state2[13] ^ tmp[1];
            state1[14] = state2[14] ^ tmp[2];
            state1[15] = state2[15] ^ tmp[3];
        }
        
        AFFINE_ENCODING(state1     , state1     , ctx->FE_F4[3], ctx->FE_E4[3], ctx->FE_V[3]);
        AFFINE_ENCODING(state1 +  4, state1 +  4, ctx->FE_F4[2], ctx->FE_E4[2], ctx->FE_V[2]);
        AFFINE_ENCODING(state1 +  8, state1 +  8, ctx->FE_F4[1], ctx->FE_E4[1], ctx->FE_V[1]);
        AFFINE_ENCODING(state1 + 12, state1 + 12, ctx->FE_F4[0], ctx->FE_E4[0], ctx->FE_V[0]);
        // dump_hex(state1,512);
        //TODO 将数据传出来然后归位 3.6 可以用_mm256_extract_epi8提取，取位的索引必须为常量，抽离成函数？ 
        // 提取数据和反序变换  
     
        DATA_PACK(state1);
        for (size_t i = 0; i < 16; i++)
        {
            state1[i] = _mm256_shuffle_epi32(state1[i],27);// 27 = 0b00011011 即反序变换
        }
        
        memcpy(OUT,state1,512);
        return 1;
    }

int wbsm4_se_avx2_ecb_encrypt(const uint8_t* inputb, uint8_t* outputb, size_t size, wbsm4_se_avx2_context *ctx)
    {
        uint8_t tmp_input[512] = {0};
        uint8_t tmp_output[512] = {0};
        memset(outputb,0,size);
        uint8_t* out = outputb;
        uint8_t* in = inputb;
        
        while(size > 0)
        {
            if(size < 512)
            {
                memset(tmp_input, 0, 512);
                memcpy(tmp_input, in, size);
        
                wbsm4_se_avx2_encrypt(tmp_input, tmp_output, ctx);

                memcpy(out, tmp_output, size);
                size = 0;
            }
            else
            {
                memcpy(tmp_input, in, 512);
    
                wbsm4_se_avx2_encrypt(tmp_input, tmp_output, ctx);

                memcpy(out, tmp_output, 512);

                size -= 512;
                out += 512;
                in += 512;
            }
        }
        return 1;
    }

int wbsm4_se_avx2_ctr_encrypt(const unsigned char *input, unsigned char *output, size_t len, unsigned char *iv, wbsm4_se_avx2_context *ctx)
    {
        uint8_t ctr[512] = {0};
        uint8_t output_space[512];
        __m128i iv_copy;
        __m128i count = _mm_setzero_si128();
        uint8_t op[16] = {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x01};
        __m128i cnt = _mm_loadu_si128((__m128i*)op);
        uint8_t *out = output;
    
        iv_copy = _mm_load_si128((__m128i *)iv);


        for(int j=0;len;j++)
        {
            int chunk = MIN(len, 512);
            int blocks = chunk / 16;//分组数

            for (int i = 0; i < blocks; i++)
            {
                _mm_storeu_si128((__m128i*)(ctr+i*16),iv_copy + count);
                count = _mm_add_epi64(count,cnt);
            }
        
            wbsm4_se_avx2_encrypt(ctr,output_space,ctx);

            memcpy(out,output_space,chunk);

            len -= chunk;
            out += chunk;

            for(int i = j*512; i < chunk+j*512; i++)
            {
                output[i] ^= input[i];
            }

        }
    }

int wbsm4_se_avx2_gcm_init(gcm_context *gcm_ctx, unsigned char *iv, unsigned char *aad, size_t alen,
                            unsigned char *tag, size_t tlen, wbsm4_se_avx2_context *ctx)
    {        
        uint8_t p_h[32],c_h[32];
        memset(p_h, 0, 32);//all 0
        memcpy(p_h+16, iv, 16);//iv||counter0
        memset(p_h+31, 1, 1);
        wbsm4_se_avx2_ecb_encrypt(p_h,c_h,32,ctx);
        computeTable(gcm_ctx->T, c_h);
        memcpy(gcm_ctx->H, c_h, 16);
        memcpy(gcm_ctx->Enc_y0, c_h+16, 16);
        gcm_ctx->iv = iv;// todo 不知道这样会不会iv丢失
        gcm_ctx->add = aad;
        gcm_ctx->alen = alen;
        gcm_ctx->tag = tag;
        gcm_ctx->tlen = tlen;
    }

int wbsm4_se_avx2_gcm_encrypt(gcm_context *gcm_ctx, const unsigned char *input, unsigned char *output, size_t ilen, wbsm4_se_avx2_context *ctx)
    {
        uint8_t ctr[512] = {0};
        uint8_t output_space[512];
        __m128i iv_copy;
        __m128i count = _mm_setzero_si128();
        uint8_t op[16] = {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x01};
        __m128i cnt = _mm_loadu_si128((__m128i*)op);
        int length = ilen;
        iv_copy = _mm_load_si128((__m128i *)gcm_ctx->iv);
        uint8_t *out = output;
       
        for(int j = 0; ilen; ++j)
        {
            int chunk = MIN(ilen, 512);
            int blocks = chunk / 16;

            count = _mm_add_epi64(count,cnt);
            
            for (int i = 0; i < blocks; i++)//gcm mode need more 1 block
            {
                //gcm mode iv from 0x02!
                count = _mm_add_epi64(count,cnt);
                _mm_storeu_si128((__m128i*)(ctr+i*16),iv_copy + count);
            }
          
            wbsm4_se_avx2_encrypt(ctr,output_space,ctx);
         
            memcpy(out,output_space,chunk);
              
            ilen -= chunk;
            out += chunk;
           
            for(int i = j*512; i < chunk+j*512; i++)
            {
                output[i] ^= input[i];
            }
        }
  
        //Auth tag test
        //compute tag
        ghash(gcm_ctx->T, gcm_ctx->add,gcm_ctx->alen, output, length, gcm_ctx->buff);
        // printf("\nbuff:\n");
        // dump_hex(gcm_ctx->buff,16);
        //uint8_t *ency1 = (uint8_t *) ctr + 16;
        for (int i = 0; i < gcm_ctx->tlen; i++)
        {
            gcm_ctx->tag[i] = gcm_ctx->buff[i] ^ gcm_ctx->Enc_y0[i];
        }
       
        //gcm_free(context);

    }
//int wbsm4_se_decrypt(const unsigned char *input, unsigned char *output, wbsm4_se_context *ctx) {
//    return wbsm4_se_encrypt(input, output, ctx);
//}