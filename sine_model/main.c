#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include <arm_neon.h>

#include "../microkernels/microkernels.h"
#include "main.h"

int main(void) {
    struct timespec startTime, endTime;

    // Load weights and biases
    size_t numElements;
    float32_t* layer1Weights = ldbin("bin/L1W.bin", &numElements);
    float32_t* layer2Weights = ldbin("bin/L2W.bin", &numElements);
    float32_t* layer3Weights = ldbin("bin/L3W.bin", &numElements);
    float32_t* layer1Bias = ldbin("bin/L1B.bin", &numElements);
    float32_t* layer2Bias = ldbin("bin/L2B.bin", &numElements);
    float32_t* layer3Bias = ldbin("bin/L3B.bin", &numElements);

    // Input tensor
    float32_t input[] = {M_PI / 3};
    float32_t l1Weights[32] = {0};
    float32_t l1Bias[32] = {0};
    float32_t l1Output[32] = {0};
    float32_t l2Weights[32] = {0};
    float32_t l2Bias[32] = {0};
    float32_t l2Output[32] = {0};
    float32_t l3Weights[1] = {0};
    float32_t l3Bias[1] = {0};

    printf("Input: %f\n", input[0]);

    clock_gettime(CLOCK_MONOTONIC, &startTime);

    // Layer 1
    __gemm_f32(layer1Weights, input, 32, 1, 1, l1Weights);
    __vadd_f32(layer1Bias, l1Weights, l1Bias, 32);
    __vrelu_f32(l1Bias, l1Output, 32);

    clock_gettime(CLOCK_MONOTONIC, &endTime);
    unsigned long long layer1Time = (endTime.tv_sec - startTime.tv_sec) * 1000000000 + (endTime.tv_nsec - startTime.tv_nsec);


    clock_gettime(CLOCK_MONOTONIC, &startTime);

    // Layer 2
    __gemm_f32(layer2Weights, l1Output, 32, 32, 1, l2Weights);
    __vadd_f32(layer2Bias, l2Weights, l2Bias, 32);
    __vrelu_f32(l2Bias, l2Output, 32);

    clock_gettime(CLOCK_MONOTONIC, &endTime);
    unsigned long long layer2Time = (endTime.tv_sec - startTime.tv_sec) * 1000000000 + (endTime.tv_nsec - startTime.tv_nsec);


    clock_gettime(CLOCK_MONOTONIC, &startTime);

    // Layer 3
    __gemm_f32(layer3Weights, l2Output, 1, 32, 1, l3Weights);
    __vadd_f32(layer3Bias, l3Weights, l3Bias, 1);

    clock_gettime(CLOCK_MONOTONIC, &endTime);
    unsigned long long layer3Time = (endTime.tv_sec - startTime.tv_sec) * 1000000000 + (endTime.tv_nsec - startTime.tv_nsec);

    printf("Output: %f\n", l3Bias[0]);

    // Layer wise execution profiling
    printf("\nExecution profiling\n");
    printf("Layer 1 Execution Time: %llu ns\n", layer1Time);
    printf("Layer 2 Execution Time: %llu ns\n", layer2Time);
    printf("Layer 3 Execution Time: %llu ns\n", layer3Time);
    printf("Total Time: %llu ns\n", layer1Time + layer2Time + layer3Time);

    return 0;
}
