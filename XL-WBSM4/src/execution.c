#include "wbsm4.h"
static uint32_t get32(const uint8_t *input)
{
    return ((uint32_t)input[0] << 24) ^ ((uint32_t)input[1] << 16) ^
           ((uint32_t)input[2] << 8) ^ input[3];
}
static void put32(uint32_t value, uint8_t *output)
{
    output[0] = value >> 24; output[1] = value >> 16;
    output[2] = value >> 8; output[3] = value;
}
void sm4_wb_enc(const uint8_t input[16], uint8_t output[16], Sm4Whitebox *context)
{
    uint32_t x0 = get32(input), x1 = get32(input + 4), x2 = get32(input + 8), x3 = get32(input + 12);
    int round;
    x0 = affineU32(context->SE[0], x0); x1 = affineU32(context->SE[1], x1);
    x2 = affineU32(context->SE[2], x2); x3 = affineU32(context->SE[3], x3);
    for (round = 0; round < 32; ++round) {
        uint32_t a = affineU32(context->M[round][0], x1) ^
                     affineU32(context->M[round][1], x2) ^
                     affineU32(context->M[round][2], x3);
        uint32_t t = context->Table[round][0][(a >> 24) & 0xff] ^
                     context->Table[round][1][(a >> 16) & 0xff] ^
                     context->Table[round][2][(a >> 8) & 0xff] ^
                     context->Table[round][3][a & 0xff];
        uint32_t x4 = affineU32(context->C[round], x0) ^ affineU32(context->D[round], t);
        x0 = x1; x1 = x2; x2 = x3; x3 = x4;
    }
    {
        uint32_t y0 = affineU32(context->FE[3], x3);
        uint32_t y1 = affineU32(context->FE[2], x2);
        uint32_t y2 = affineU32(context->FE[1], x1);
        uint32_t y3 = affineU32(context->FE[0], x0);
        put32(y0, output); put32(y1, output + 4); put32(y2, output + 8); put32(y3, output + 12);
    }
}
