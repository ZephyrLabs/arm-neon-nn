#include "microkernels.h"

void __vadd_f16(const float16_t* a, const float16_t* b, float16_t* result, int n){
    int k = 0;

    // Process with 128 bit NEON regsiters (Quad-Word)
    for (; k + 8 <= n; k += 8) {
        float16x8_t a_vec = vld1q_f16(&a[k]);
        float16x8_t b_vec = vld1q_f16(&b[k]);
        float16x8_t result_vec = vaddq_f16(a_vec, b_vec);
        vst1q_f16(&result[k], result_vec);
    }

    // switch to using 64 bit NEON registers (Bi-Word)
    for (; k + 4 <= n; k += 4) {
        float16x4_t a_vec = vld1_f16(&a[k]);
        float16x4_t b_vec = vld1_f16(&b[k]);
        float16x4_t result_vec = vadd_f16(a_vec, b_vec);
        vst1_f16(&result[k], result_vec);
    }

    // Handle remaining
    for (; k < n; k++) {
        result[k] = a[k] + b[k];
    }
}

void __vsub_f16(const float16_t* a, const float16_t* b, float16_t* result, int n){
    int k = 0;

    // Process with 128 bit NEON regsiters (Quad-Word)
    for (; k + 8 <= n; k += 8) {
        float16x8_t a_vec = vld1q_f16(&a[k]);
        float16x8_t b_vec = vld1q_f16(&b[k]);
        float16x8_t result_vec = vsubq_f16(a_vec, b_vec);
        vst1q_f16(&result[k], result_vec);
    }

    // switch to using 64 bit NEON registers (Bi-Word)
    for (; k + 4 <= n; k += 4) {
        float16x4_t a_vec = vld1_f16(&a[k]);
        float16x4_t b_vec = vld1_f16(&b[k]);
        float16x4_t result_vec = vsub_f16(a_vec, b_vec);
        vst1_f16(&result[k], result_vec);
    }

    // Handle remaining
    for (; k < n; k++) {
        result[k] = a[k] - b[k];
    }
}

void __vmul_f16(const float16_t* a, const float16_t* b, float16_t* result, int n){
    int k = 0;

    // Process with 128 bit NEON regsiters (Quad-Word)
    for (; k + 8 <= n; k += 8) {
        float16x8_t a_vec = vld1q_f16(&a[k]);
        float16x8_t b_vec = vld1q_f16(&b[k]);
        float16x8_t result_vec = vsubq_f16(a_vec, b_vec);
        vst1q_f16(&result[k], result_vec);
    }

    // switch to using 64 bit NEON registers (Bi-Word)
    for (; k + 4 <= n; k += 4) {
        float16x4_t a_vec = vld1_f16(&a[k]);
        float16x4_t b_vec = vld1_f16(&b[k]);
        float16x4_t result_vec = vsub_f16(a_vec, b_vec);
        vst1_f16(&result[k], result_vec);
    }

    // Handle remaining
    for (; k < n; k++) {
        result[k] = a[k] - b[k];
    }
}

void __vdiv_f16(const float16_t* a, const float16_t* b, float16_t* result, int n){
    int k = 0;

    // Process with 128 bit NEON regsiters (Quad-Word)
    for (; k + 8 <= n; k += 8) {
        float16x8_t a_vec = vld1q_f16(&a[k]);
        float16x8_t b_vec = vld1q_f16(&b[k]);
        float16x8_t result_vec = vdivq_f16(a_vec, b_vec);
        vst1q_f16(&result[k], result_vec);
    }

    // switch to using 64 bit NEON registers (Bi-Word)
    for (; k + 4 <= n; k += 4) {
        float16x4_t a_vec = vld1_f16(&a[k]);
        float16x4_t b_vec = vld1_f16(&b[k]);
        float16x4_t result_vec = vdiv_f16(a_vec, b_vec);
        vst1_f16(&result[k], result_vec);
    }

    // Handle remaining
    for (; k < n; k++) {
        result[k] = a[k] / b[k];
    }
}

void __vrelu_f16(const float16_t* a, float16_t* result, int n) {
    int k = 0;

    // Process with 128 bit NEON regsiters (Quad-Word)
    for (; k + 8 <= n; k += 8) {
        float16x8_t a_vec = vld1q_f16(&a[k]);
        float16x8_t zero_vec = vdupq_n_f16(0.0f);
        float16x8_t result_vec = vmaxq_f16(a_vec, zero_vec);
        vst1q_f16(&result[k], result_vec);
    }

    // Switch to using 64-bit NEON registers (Bi-Word)
    for (; k + 4 <= n; k += 4) {
        float16x4_t a_vec = vld1_f16(&a[k]);
        float16x4_t zero_vec = vdup_n_f16(0.0f);
        float16x4_t result_vec = vmax_f16(a_vec, zero_vec);
        vst1_f16(&result[k], result_vec);
    }

    // Handle remaining
    for (; k < n; k++) {
        result[k] = (a[k] > 0.0f) ? a[k] : 0.0f;
    }
}

void __gemm_f16(const float16_t* a, const float16_t* b, const int l, const int m, const int n, float16_t* c){
    for(int i = 0; i < l; i++){
        for(int j = 0; j < n; j++) {
            int k = 0;

            // Process with 128 bit NEON regsiters (Quad-Word)
            float16x8_t acc_f16x8 = vdupq_n_f16(0.0);
            for (; k + 8 <= m; k += 8) {
                float16x8_t a_vec = vld1q_f16(&a[i * m + k]);
                float16x8_t b_vec = vld1q_f16(&b[k * n + j]);
                acc_f16x8 = vfmaq_f16(acc_f16x8, a_vec, b_vec);
            }
            float16_t fp16x8[8];
            vst1q_f16(fp16x8, acc_f16x8);

            // unrolled loop
            c[i * n + j] += fp16x8[0] + fp16x8[1] + fp16x8[2] + fp16x8[3] + fp16x8[4] + fp16x8[5] + fp16x8[6] + fp16x8[7];

            // switch to using 64 bit NEON registers (Bi-Word)
            float16x4_t acc_f16x4 = vdup_n_f16(0.0);
            for (; k + 4 <= m; k += 4) {
                float16x4_t a_vec = vld1_f16(&a[i * m + k]);
                float16x4_t b_vec = vld1_f16(&b[k * n + j]);
                acc_f16x4 = vfma_f16(acc_f16x4, a_vec, b_vec);
            }
            float16_t fp16x4[4];
            vst1_f16(fp16x4, acc_f16x4);

            // unrolled loop
            c[i * n + j] += fp16x4[0] + fp16x4[1] + fp16x4[2] + fp16x4[3];

            // Handle remaining
            for (; k < m; k++) {
                c[i * n + j] += a[i * m + k] * b[k * n + j];
            }
        }
    }
}
