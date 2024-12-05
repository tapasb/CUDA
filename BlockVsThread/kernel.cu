#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <chrono> // Required for timing

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size, int blocks, int threadsPerBlock);

__global__ void addKernel(int* c, const int* a, const int* b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
}

int main() {
    const int arraySize = 1000; // Increased array size for better performance measurement
    int a[arraySize];
    int b[arraySize];
    int c[arraySize];

    // Initialize arrays
    for (int i = 0; i < arraySize; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    // --- Timing and execution with different configurations ---

    printf("| B     | T/B   | Time (ms) |\n");
    printf("|-------|-------|-----------|\n");

    // Configuration 1: 1 block, many threads
    for (int i = 0; i < 10; i++) { // Run 10 times
        auto start = std::chrono::high_resolution_clock::now();
        cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize, 1, arraySize);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "addWithCuda failed!");
            return 1;
        }
        printf("| 1 | %d | %f |\n", arraySize, duration.count() / 1000.0);
    }

    // Configuration 2: Many blocks, 1 thread per block
    for (int i = 0; i < 10; i++) { // Run 10 times
        auto start = std::chrono::high_resolution_clock::now();
        cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize, arraySize, 1);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "addWithCuda failed!");
            return 1;
        }
        printf("| %d | 1 | %f |\n", arraySize, duration.count() / 1000.0);
    }

    // --- End of timing and execution ---

    // Print the first 5 results for verification
    printf("\n{");
    for (int i = 0; i < 5; i++) {
        printf("%d, ", a[i]);
    }
    printf("} + {");
    for (int i = 0; i < 5; i++) {
        printf("%d, ", b[i]);
    }
    printf("} = {");
    for (int i = 0; i < 5; i++) {
        printf("%d, ", c[i]);
    }
    printf("}\n");

    cudaDeviceReset();
    return 0;
}

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size, int blocks, int threadsPerBlock) {
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch the kernel with the specified grid and block dimensions
    addKernel << <blocks, threadsPerBlock >> > (dev_c, dev_a, dev_b);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}


/*
-----------Output-----------

| B     | T/B   | Time (ms) |
|-------|-------|-----------|
| 1 | 1000 | 86.526000 |
| 1 | 1000 | 0.539000 |
| 1 | 1000 | 0.673000 |
| 1 | 1000 | 0.511000 |
| 1 | 1000 | 0.504000 |
| 1 | 1000 | 0.917000 |
| 1 | 1000 | 0.586000 |
| 1 | 1000 | 0.461000 |
| 1 | 1000 | 0.402000 |
| 1 | 1000 | 0.377000 |
| 1000 | 1 | 0.412000 |
| 1000 | 1 | 0.410000 |
| 1000 | 1 | 0.445000 |
| 1000 | 1 | 0.471000 |
| 1000 | 1 | 0.386000 |
| 1000 | 1 | 0.419000 |
| 1000 | 1 | 0.427000 |
| 1000 | 1 | 0.379000 |
| 1000 | 1 | 0.351000 |
| 1000 | 1 | 0.336000 |

{0, 1, 2, 3, 4, } + {0, 2, 4, 6, 8, } = {0, 3, 6, 9, 12, }


*/