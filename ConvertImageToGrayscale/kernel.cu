#include <iostream>
#include <chrono> // For timing
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <chrono> // For timing
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// CUDA kernel function for grayscale conversion
__global__ void grayscaleKernel(uchar3* image, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int index = y * width + x;
        uchar3 pixel = image[index];
        unsigned char grayValue =
            0.2126 * pixel.x + 0.7152 * pixel.y + 0.0722 * pixel.z;
        image[index] = make_uchar3(grayValue, grayValue, grayValue);
    }
}

int main() {
    string imagePath = "C:/Users/tapas/CUDA/ConvertImageToGrayscale/image.jpeg";
    Mat image = imread(imagePath, IMREAD_COLOR);

    if (image.empty()) {
        cerr << "Error opening image file: " << imagePath << endl;
        return 1;
    }

    // Convert to uchar3 for easier CUDA handling
    Mat imageUchar3;
    image.convertTo(imageUchar3, CV_8UC3);

    // Get image dimensions
    int width = image.cols;
    int height = image.rows;

    // Allocate device memory
    uchar3* dev_image;
    cudaMalloc((void**)&dev_image, width * height * sizeof(uchar3));

    // Copy image data from host to device
    cudaMemcpy(dev_image, imageUchar3.data, width * height * sizeof(uchar3),
        cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    const int blockSize = 16;
    dim3 block(blockSize, blockSize);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // --- GPU processing ---
    auto startGPU = chrono::high_resolution_clock::now();
    grayscaleKernel << <grid, block >> > (dev_image, width, height);
    cudaDeviceSynchronize(); // Wait for kernel to finish
    auto endGPU = chrono::high_resolution_clock::now();
    auto durationGPU =
        chrono::duration_cast<chrono::microseconds>(endGPU - startGPU);

    // Copy grayscale image data from device to host
    cudaMemcpy(imageUchar3.data, dev_image, width * height * sizeof(uchar3),
        cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(dev_image);

    // --- Output performance metrics ---
    cout << "GPU Time (ms): " << durationGPU.count() / 1000.0 << endl;

    // Display the grayscale and color images side by side
    Mat combinedImage;
    hconcat(image, imageUchar3, combinedImage);
    imshow("Color (Left) vs. Grayscale (Right)", combinedImage);
    waitKey(0);

    return 0;
}