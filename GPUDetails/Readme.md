# CUDA API Tutorial: Understanding GPU Details

This tutorial explains the CUDA APIs used in the provided code to print GPU details and provides examples to clarify the terminology.

## CUDA APIs

The code uses the following CUDA runtime APIs:

*   `cudaGetDeviceCount()`: Retrieves the number of CUDA-capable GPUs in the system.
*   `cudaGetDeviceProperties()`: Retrieves properties of a specific GPU.

These APIs are declared in the `cuda_runtime.h` header file.

## Terminology

*   **Compute Capability:** A number that indicates the features and capabilities of a GPU. It consists of a major and minor revision number (e.g., 8.9). Higher compute capabilities generally indicate more powerful GPUs with newer features.
*   **Global Memory:** The main memory on the GPU, accessible by all threads. It's typically large (several gigabytes) and used to store data that needs to be shared between threads or persist across kernel launches.
*   **Shared Memory per Block:** A smaller, faster memory space that is shared by all threads within a block. It's useful for communication and data sharing within a block.
*   **Registers per Block:** The number of registers available to each block of threads. Registers are the fastest form of memory on the GPU and are used to store frequently accessed variables.
*   **Warp Size:** The number of threads that are executed simultaneously on a single processor core. It's typically 32 for NVIDIA GPUs.
*   **Max Threads per Block:** The maximum number of threads that can be launched in a single block. This limit depends on the GPU's compute capability.
*   **Max Block Dimensions:** The maximum dimensions of a block of threads in each dimension (x, y, z).
*   **Max Grid Dimensions:** The maximum dimensions of a grid of blocks in each dimension (x, y, z).

## Memory Types

CUDA provides several memory types with different characteristics and use cases:

*   **Global Memory:** Large, accessible by all threads, high latency.
    *   Example: Storing large arrays of input and output data.
*   **Shared Memory:** Smaller, faster than global memory, shared within a block.
    *   Example: Sharing intermediate results between threads in a block.
*   **Constant Memory:** Read-only, cached, accessible by all threads.
    *   Example: Storing constant values used by all threads (e.g., mathematical constants).
*   **Texture Memory:** Cached, optimized for spatial locality, accessible by all threads.
    *   Example: Storing image data for image processing applications.
*   **Local Memory:** Private to each thread, slower than registers.
    *   Example: Storing thread-specific temporary variables.
*   **Registers:** Fastest memory, private to each thread.
    *   Example: Storing loop counters and frequently accessed variables.

## Examples

*   **Compute Capability: 8.9** This indicates a GPU with major revision 8 and minor revision 9. It's a relatively high compute capability, suggesting a modern GPU with support for advanced features.
*   **Global Memory: 8187 MB** The GPU has 8187 MB of global memory available for storing data.
*   **Shared Memory per Block: 48 KB** Each block of threads can share 48 KB of memory for communication and data sharing.
*   **Registers per Block: 65536** Each block of threads has 65536 registers available for storing variables.
*   **Warp Size: 32** The GPU executes 32 threads simultaneously on each processor core.
*   **Max Threads per Block: 1024** A maximum of 1024 threads can be launched in a single block.
*   **Max Block Dimensions: (1024, 1024, 64)** A block of threads can have up to 1024 threads in the x and y dimensions and 64 threads in the z dimension.
*   **Max Grid Dimensions: (2147483647, 65535, 65535)** A grid of blocks can have a very large number of blocks in the x dimension and up to 65535 blocks in the y and z dimensions.

This tutorial provides a basic understanding of the CUDA APIs used to query GPU details, explains the associated terminology, and provides examples of different memory types and their use cases. You can further explore CUDA programming by learning about different memory types, thread organization, and more advanced CUDA libraries.