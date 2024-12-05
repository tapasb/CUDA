# Understanding Grids, Threads, and Blocks in CUDA

This README explains the fundamental concepts of grids, threads, and blocks in CUDA programming, using the provided `Kernel.cu` code as an example.

## Key Concepts

*   **Thread:** The basic unit of execution in CUDA. Each thread runs the same kernel code but operates on different data.

*   **Block:** A group of threads that can cooperate and share data. Threads within a block can synchronize their execution and access shared memory.

*   **Grid:** A collection of blocks that execute the same kernel. The grid defines the overall organization of threads and blocks on the GPU.

## Example: Kernel.cu

The `Kernel.cu` code demonstrates these concepts by performing parallel addition of two arrays using a CUDA kernel.

### Kernel Function

```cpp
__global__ void addKernel(int* c, const int* a, const int* b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
}
```

*   This `addKernel` function is executed on the GPU by multiple threads in parallel.

*   Each thread calculates its index (`i`) based on its block index (`blockIdx.x`), block dimension (`blockDim.x`), and thread index within the block (`threadIdx.x`).

*   Each thread then adds the corresponding elements of arrays `a` and `b` and stores the result in array `c`.

### Kernel Launch

```cpp
addKernel << <blocks, threadsPerBlock >> > (dev_c, dev_a, dev_b);
```

*   This line launches the `addKernel` with a specified number of `blocks` and `threadsPerBlock`.

*   The `<<<...>>>` syntax defines the grid and block dimensions.

*   By adjusting these parameters, you control how the workload is divided among threads and blocks on the GPU.

## Performance Considerations

The choice of grid and block dimensions significantly impacts performance. Factors to consider include:

*   **Workload size:** Larger workloads often benefit from more blocks to distribute the work.

*   **Thread cooperation:** If threads need to cooperate closely, using fewer blocks with more threads per block is preferable.

*   **Hardware limitations:** The maximum number of threads per block and blocks per grid is limited by the GPU's capabilities.

Experimenting with different configurations is crucial to find the best balance for your CUDA program's performance.