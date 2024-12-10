#include <cuda_runtime.h>
#include <iostream>

// Example kernel: Add two arrays
__global__ void vectorAdd(const int* a, const int* b, int* c, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    // Problem size
    int size = 1000000;
    int* a, * b, * c; // Host vectors
    int* d_a, * d_b, * d_c; // Device vectors

    // Allocate host memory
    a = new int[size];
    b = new int[size];
    c = new int[size];

    // Initialize host data
    for (int i = 0; i < size; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    // Allocate device memory
    cudaMalloc(&d_a, size * sizeof(int));
    cudaMalloc(&d_b, size * sizeof(int));
    cudaMalloc(&d_c, size * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // Calculate block size 
    int blockSize = prop.maxThreadsPerBlock; // Start with maximum

    // Adjust block size based on shared memory or register usage if needed
    // For this simple example, we don't have those constraints

    // Calculate grid size
    int gridSize = (size + blockSize - 1) / blockSize;

    // Launch the kernel
    vectorAdd << <gridSize, blockSize >> > (d_a, d_b, d_c, size);

    // Copy results back to host
    cudaMemcpy(c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Verify results (optional)
    for (int i = 0; i < 5; i++) {
        std::cout << a[i] << " + " << b[i] << " = " << c[i] << std::endl;
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}