#include "microkernels.h"

void __vadd_f32(const float* a, const float* b, float* result, int n) {
    int k = 0;

    // Process with 128-bit NEON registers (Quad-Word)
    for (; k + 4 <= n; k += 4) {
        float32x4_t a_vec = vld1q_f32(&a[k]);
        float32x4_t b_vec = vld1q_f32(&b[k]);
        float32x4_t result_vec = vaddq_f32(a_vec, b_vec);
        vst1q_f32(&result[k], result_vec);
    }

    // Use 64-bit NEON registers (Bi-Word) for the remaining elements
    for (; k + 2 <= n; k += 2) {
        float32x2_t a_vec = vld1_f32(&a[k]);
        float32x2_t b_vec = vld1_f32(&b[k]);
        float32x2_t result_vec = vadd_f32(a_vec, b_vec);
        vst1_f32(&result[k], result_vec);
    }

    // Handle remaining elements using scalar operations
    for (; k < n; k++) {
        result[k] = a[k] + b[k];
    }
}

void __vsub_f32(const float* a, const float* b, float* result, int n) {
    int k = 0;

    // Process with 128-bit NEON registers (Quad-Word)
    for (; k + 4 <= n; k += 4) {
        float32x4_t a_vec = vld1q_f32(&a[k]);
        float32x4_t b_vec = vld1q_f32(&b[k]);
        float32x4_t result_vec = vsubq_f32(a_vec, b_vec);
        vst1q_f32(&result[k], result_vec);
    }

    // Use 64-bit NEON registers (Bi-Word) for the remaining elements
    for (; k + 2 <= n; k += 2) {
        float32x2_t a_vec = vld1_f32(&a[k]);
        float32x2_t b_vec = vld1_f32(&b[k]);
        float32x2_t result_vec = vsub_f32(a_vec, b_vec);
        vst1_f32(&result[k], result_vec);
    }

    // Handle remaining elements using scalar operations
    for (; k < n; k++) {
        result[k] = a[k] - b[k];
    }
}

void __vmul_f32(const float* a, const float* b, float* result, int n) {
    int k = 0;

    // Process with 128-bit NEON registers (Quad-Word)
    for (; k + 4 <= n; k += 4) {
        float32x4_t a_vec = vld1q_f32(&a[k]);
        float32x4_t b_vec = vld1q_f32(&b[k]);
        float32x4_t result_vec = vmulq_f32(a_vec, b_vec);
        vst1q_f32(&result[k], result_vec);
    }

    // Use 64-bit NEON registers (Bi-Word) for the remaining elements
    for (; k + 2 <= n; k += 2) {
        float32x2_t a_vec = vld1_f32(&a[k]);
        float32x2_t b_vec = vld1_f32(&b[k]);
        float32x2_t result_vec = vmul_f32(a_vec, b_vec);
        vst1_f32(&result[k], result_vec);
    }

    // Handle remaining elements using scalar operations
    for (; k < n; k++) {
        result[k] = a[k] * b[k];
    }
}

void __vdiv_f32(const float* a, const float* b, float* result, int n) {
    int k = 0;

    // Process with 128-bit NEON registers (Quad-Word)
    for (; k + 4 <= n; k += 4) {
        float32x4_t a_vec = vld1q_f32(&a[k]);
        float32x4_t b_vec = vld1q_f32(&b[k]);
        float32x4_t result_vec = vdivq_f32(a_vec, b_vec);
        vst1q_f32(&result[k], result_vec);
    }

    // Use 64-bit NEON registers (Bi-Word) for the remaining elements
    for (; k + 2 <= n; k += 2) {
        float32x2_t a_vec = vld1_f32(&a[k]);
        float32x2_t b_vec = vld1_f32(&b[k]);
        float32x2_t result_vec = vdiv_f32(a_vec, b_vec);
        vst1_f32(&result[k], result_vec);
    }

    // Handle remaining elements using scalar operations
    for (; k < n; k++) {
        result[k] = a[k] / b[k];
    }
}

void __vrelu_f32(const float* a, float* result, int n) {
    int k = 0;

    // Process with 128-bit NEON registers (Quad-Word)
    for (; k + 4 <= n; k += 4) {
        float32x4_t a_vec = vld1q_f32(&a[k]);
        float32x4_t zero_vec = vdupq_n_f32(0.0f);
        float32x4_t result_vec = vmaxq_f32(a_vec, zero_vec);
        vst1q_f32(&result[k], result_vec);
    }

    // Use 64-bit NEON registers (Bi-Word) for the remaining elements
    for (; k + 2 <= n; k += 2) {
        float32x2_t a_vec = vld1_f32(&a[k]);
        float32x2_t zero_vec = vdup_n_f32(0.0f);
        float32x2_t result_vec = vmax_f32(a_vec, zero_vec);
        vst1_f32(&result[k], result_vec);
    }

    // Handle remaining elements using scalar operations
    for (; k < n; k++) {
        result[k] = (a[k] > 0.0f) ? a[k] : 0.0f;
    }
}

void __gemm_f32(const float* a, const float* b, const int l, const int m, const int n, float* c){
    for (int i = 0; i < l; i++) {
        for (int j = 0; j < n; j++) {
            int k = 0;

            // Process with 128-bit NEON registers (Quad-Word)
            float32x4_t acc_f32x4 = vdupq_n_f32(0.0);
            for (; k + 4 <= m; k += 4) {
                float32x4_t a_vec = vld1q_f32(&a[i * m + k]);
                float32x4_t b_vec = vld1q_f32(&b[k * n + j]);
                acc_f32x4 = vmlaq_f32(acc_f32x4, a_vec, b_vec);
            }
            float32_t fp32x4[4];
            vst1q_f32(fp32x4, acc_f32x4);

            // Unrolled loop
            c[i * n + j] += fp32x4[0] + fp32x4[1] + fp32x4[2] + fp32x4[3];

            // switch to using 64 bit NEON registers (Bi-Word)
            float32x2_t acc_f32x2 = vdup_n_f32(0.0);
            for (; k + 2 <= m; k += 2) {
                float32x2_t a_vec = vld1_f32(&a[i * m + k]);
                float32x2_t b_vec = vld1_f32(&b[k * n + j]);
                acc_f32x2 = vfma_f32(acc_f32x2, a_vec, b_vec);
            }
            float32_t fp32x2[2];
            vst1_f32(fp32x2, acc_f32x2);

            // unrolled loop
            c[i * n + j] += fp32x2[0] + fp32x2[1];

            // Handle remaining elements using scalar operations
            for (; k < m; k++) {
                c[i * n + j] += a[i * m + k] * b[k * n + j];
            }
        }
    }
}
