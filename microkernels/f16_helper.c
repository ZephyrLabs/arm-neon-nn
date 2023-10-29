#include "microkernels.h"

void __f16_f32_quant(const float32_t* input, float16_t* output, int n){
    int k = 0;

    // Process with 128 bit NEON registers (Quad-Word)
    for (; k + 4 <= n; k += 4) {
        float32x4_t fp32x4_a = vld1q_f32(input + k);
        float16x4_t fp16x4 = vcvt_f16_f32(fp32x4_a);
        vst1_f16(output + k, fp16x4);
    }

    // Handle remaining values
    for (; k < n; k++) {
        output[k] = (float16_t)input[k];
    }
}

void __f16_f32_dequant(const float16_t* input, float32_t* output, int n){
    int k = 0;

    // Process with 128 bit NEON registers (Quad-Word) (sorta)
    for (; k + 4 <= n; k += 4) {
        float16x4_t fp16x4 = vld1_f16(input + k);
        float32x4_t fp32x4 = vcvt_f32_f16(fp16x4);
        vst1q_f32(output + k, fp32x4);
    }

    // Handle remaining values
    for (; k < n; k++) {
        output[k] = (float32_t)input[k];
    }
}
