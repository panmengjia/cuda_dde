//
// CUDA implementation of Laplacian Filter
//

#include "main.h"
#include "cuda_fp16.h"
//#include <qdebug.h>

using namespace std;

const int inputSize = sizeof(__half) * IMG_HEIGHT * IMG_WIDTH;
const int outputSize = sizeof(__half) * IMG_HEIGHT * IMG_WIDTH;
const int kernelSize = sizeof(__half) * FILTER_WIDTH * FILTER_HEIGHT;
__half* d_input, * d_output; //指针类型虽然不影响，内存分配大小，但是严谨使用，应该与目标数据的类型（fp16）一至
__half* d_kernel;

//cuda 官网 https://docs.nvidia.com/cuda/cuda-samples/index.html
//配置 sm compute https://stackoverflow.com/questions/35656294/cuda-how-to-use-arch-and-code-and-sm-vs-compute
//运运行 cuda 的sample代码  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\extras\demo_suite


bool initialized = false;

// Run Laplacian Filter on GPU
__global__ void laplacianFilter(__half* srcImage, __half* dstImage, unsigned int width, unsigned int height, __half* kernel0)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;  //block在grid中的序号 block在x维度的大小 线程在一个block中的序号
    int y = blockIdx.y * blockDim.y + threadIdx.y;   //线程的全局index

    //float kernel[3][3] = {-1, -1, -1, -1, 8, -1, -1, -1, -1};
    // only threads inside image will write results
    if ((x >= FILTER_WIDTH / 2) && (x < (width - FILTER_WIDTH / 2)) && (y >= FILTER_HEIGHT / 2) && (y < (height - FILTER_HEIGHT / 2)))
    {
        // Sum of pixel values
        __half sum = 0.0;
        // Loop inside the filter to average pixel values
        for (int ky = -FILTER_HEIGHT / 2; ky <= FILTER_HEIGHT / 2; ky++) {
            for (int kx = -FILTER_WIDTH / 2; kx <= FILTER_WIDTH / 2; kx++) {
                __half fl = srcImage[((y + ky) * width + (x + kx))];
                //sum += fl * kernel0[(ky + FILTER_HEIGHT / 2) * FILTER_WIDTH + kx + FILTER_WIDTH / 2];
                __half a = __hmul(fl, kernel0[(ky + FILTER_HEIGHT / 2) * FILTER_WIDTH + kx + FILTER_WIDTH / 2]);
                sum = __hadd(sum,a);
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
        //cudaMalloc<unsigned char>(&d_input, inputSize);
        //cudaMalloc<unsigned char>(&d_output, outputSize);
        //cudaMalloc<float>(&d_kernel, kernelSize);

        cudaMalloc((void**)&d_input, inputSize);
        cudaMalloc((void**)&d_output, outputSize);
        cudaMalloc((void**)&d_kernel, kernelSize);

        initialized = true;
    }

    // Copy data from OpenCV input image to device memory
    cudaMemcpy(d_input, input.ptr(), inputSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel.ptr(), kernelSize, cudaMemcpyHostToDevice);

    // Specify block size
    const dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    // Calculate grid size to cover the whole image
    const dim3 grid((output.cols + block.x - 1) / block.x, (output.rows + block.y - 1) / block.y);

    //Start time
    cudaEventRecord(start, 0);

    laplacianFilter <<<grid, block >>> ((__half*)d_input, (__half*)d_output, output.cols, output.rows, (__half*)d_kernel);  //网格中线程块的维度，一共有多少的线程块；线程块中线程的维度，每个线程块一共有多少个线程

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


