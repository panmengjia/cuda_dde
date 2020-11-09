//
// CUDA implementation of Laplacian Filter
//
#include "opencv2/opencv.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <stdio.h>
#include <cuda.h>
#include "cuda_runtime.h"
//#include <qdebug.h>

#define BLOCK_SIZE      8
#define FILTER_WIDTH    15
#define FILTER_HEIGHT   15

using namespace std;

const int inputSize = sizeof(uchar) * 1920 * 1080;
const int outputSize = sizeof(uchar) * 1920 * 1080;
const int kernelSize = sizeof(float) * FILTER_WIDTH * FILTER_HEIGHT;
unsigned char* d_input, * d_output;
float* d_kernel;


bool initialized = false;

// Run Laplacian Filter on GPU
__global__ void laplacianFilter(unsigned char* srcImage, unsigned char* dstImage, unsigned int width, unsigned int height, float* kernel0)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    //float kernel[3][3] = {-1, -1, -1, -1, 8, -1, -1, -1, -1};
    // only threads inside image will write results
    if ((x >= FILTER_WIDTH / 2) && (x < (width - FILTER_WIDTH / 2)) && (y >= FILTER_HEIGHT / 2) && (y < (height - FILTER_HEIGHT / 2)))
    {
        // Sum of pixel values
        float sum = 0;
        // Loop inside the filter to average pixel values
        for (int ky = -FILTER_HEIGHT / 2; ky <= FILTER_HEIGHT / 2; ky++) {
            for (int kx = -FILTER_WIDTH / 2; kx <= FILTER_WIDTH / 2; kx++) {
                float fl = srcImage[((y + ky) * width + (x + kx))];
                sum += fl * kernel0[(ky + FILTER_HEIGHT / 2) * FILTER_WIDTH + kx + FILTER_WIDTH / 2];
            }
        }
        dstImage[(y * width + x)] = sum;
    }
}


// The wrapper to call laplacian filter
extern "C" void laplacianFilter_GPU_wrapper(const cv::Mat & input, cv::Mat & output, const cv::Mat & kernel)
{
    // Use cuda event to catch time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    if (!initialized)
    {
        cout << "111" << endl;
        // Allocate device memory
        cudaMalloc<unsigned char>(&d_input, inputSize);
        cudaMalloc<unsigned char>(&d_output, outputSize);
        cudaMalloc<float>(&d_kernel, kernelSize);

        initialized = true;
    }

    // Copy data from OpenCV input image to device memory
    cudaMemcpy(d_input, input.ptr(), inputSize, cudaMemcpyHostToDevice);
    //        cout<<"22222"<<endl;
    cudaMemcpy(d_kernel, kernel.ptr(), kernelSize, cudaMemcpyHostToDevice);

    // Specify block size
    const dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    // Calculate grid size to cover the whole image
    const dim3 grid((output.cols + block.x - 1) / block.x, (output.rows + block.y - 1) / block.y);

    // Start time
    cudaEventRecord(start, 0);

    laplacianFilter << <grid, block >> > (d_input, d_output, output.cols, output.rows, d_kernel);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("Kernel time: %.2f ms\n", milliseconds);

    //Copy data from device memory to output image

    cudaMemcpy(output.ptr(), d_output, outputSize, cudaMemcpyDeviceToHost);

    //Free the device memory
//        cudaFree(d_input);
//        cudaFree(d_output);
//        cudaFree(d_kernel);


}


