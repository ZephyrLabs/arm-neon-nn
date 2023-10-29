#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <arm_neon.h>

float32_t* ldbin(const char* binary_filename, size_t* num_elements) {
    FILE* file = fopen(binary_filename, "rb");

    if(file == NULL) {
        perror("Error opening binary file");
        return NULL;
    }

    // Get the file size
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    // Calculate the number of elements (assuming 4 bytes per element for float32_t)
    *num_elements = file_size / sizeof(float32_t);

    // Allocate memory to store the data
    float32_t* weights = (float32_t*)malloc(file_size);

    if (weights == NULL) {
        perror("Memory allocation failed");
        return NULL;
    }

    // Read from the binary file
    size_t elements_read = fread(weights, sizeof(float32_t), *num_elements, file);

    // Handle error
    if (elements_read * sizeof(float) != file_size) {
        perror("Error reading binary file");
        free(weights);
        return NULL;
    }

    fclose(file);

    return weights;
}
