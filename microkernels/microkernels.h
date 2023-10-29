#include <arm_neon.h>

/* fp16 micro kernels */
void __vadd_f16(const float16_t* a, const float16_t* b, float16_t* result, int n);
void __vsub_f16(const float16_t* a, const float16_t* b, float16_t* result, int n);
void __vmul_f16(const float16_t* a, const float16_t* b, float16_t* result, int n);
void __vdiv_f16(const float16_t* a, const float16_t* b, float16_t* result, int n);
void __vrelu_f16(const float16_t* a, float16_t* result, int n);
void __gemm_f16(const float16_t* a, const float16_t* b, const int l, const int m, const int n, float16_t* c);

/* fp32 micro kernels */
void __vadd_f32(const float32_t* a, const float32_t* b, float32_t* result, int n);
void __vsub_f32(const float32_t* a, const float32_t* b, float32_t* result, int n);
void __vmul_f32(const float32_t* a, const float32_t* b, float32_t* result, int n);
void __vdiv_f32(const float32_t* a, const float32_t* b, float32_t* result, int n);
void __vrelu_f32(const float32_t* a, float32_t* result, int n);
void __gemm_f32(const float32_t* a, const float32_t* b, const int l, const int m, const int n, float32_t* c);

/* helper functions */
void __f16_f32_quant(const float32_t* input, float16_t* output, int n);
void __f16_f32_dequant(const float16_t* input, float32_t* output, int n);
