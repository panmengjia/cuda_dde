//
// CUDA implementation of Laplacian Filter
//
//#include "opencv2/opencv.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <stdio.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "cuda_fp16.h"
//#include <qdebug.h>
#include "main.h"



using namespace std;

const int inputSize = sizeof(__half)*1920 * 1080;
const int outputSize = sizeof(__half)*1920 * 1080;
const int kernelSize = sizeof(__half)*FILTER_WIDTH * FILTER_HEIGHT;
__half *d_input, *d_output;
__half *d_kernel;


bool initialized=false;

// Run Laplacian Filter on GPU
__global__ void laplacianFilter(__half*srcImage, __half*dstImage, unsigned int width, unsigned int height, __half* kernel0)
{
   int x = blockIdx.x*blockDim.x + threadIdx.x;
   int y = blockIdx.y*blockDim.y + threadIdx.y;

   //float kernel[3][3] = {-1, -1, -1, -1, 8, -1, -1, -1, -1};
   // only threads inside image will write results
   if((x>=FILTER_WIDTH/2) && (x<(width-FILTER_WIDTH/2)) && (y>=FILTER_HEIGHT/2) && (y<(height-FILTER_HEIGHT/2)))
   {
         // Sum of pixel values
         __half sum = 0.0;
         // Loop inside the filter to average pixel values
         for(int ky=-FILTER_HEIGHT/2; ky<=FILTER_HEIGHT/2; ky++) {
            for(int kx=-FILTER_WIDTH/2; kx<=FILTER_WIDTH/2; kx++) {
               __half fl = srcImage[((y+ky)*width + (x+kx))];
               __half a = __hmul(fl , kernel0[(ky + FILTER_HEIGHT / 2) * FILTER_WIDTH + kx + FILTER_WIDTH / 2]);
              /* sum += fl*kernel0[(ky+FILTER_HEIGHT/2)*FILTER_WIDTH + kx+FILTER_WIDTH/2];*/
               sum = __hadd(sum, a);
            }
         }
         dstImage[(y*width+x)] =  sum;
   }
}


// The wrapper to call laplacian filter
extern "C" void laplacianFilter_GPU_wrapper(const cv::Mat& input, cv::Mat& output, const cv::Mat& kernel)
{
        // Use cuda event to catch time
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        if (!initialized)
        {
            cout<<"111"<<endl;
            // Allocate device memory
            cudaMalloc(&d_input,inputSize);
            cudaMalloc(&d_output,outputSize);
            cudaMalloc(&d_kernel,kernelSize);

            initialized=true;
        }

        // Copy data from OpenCV input image to device memory
        cudaMemcpy(d_input,input.ptr(),inputSize,cudaMemcpyHostToDevice);
//        cout<<"22222"<<endl;
        cudaMemcpy(d_kernel,kernel.ptr(),kernelSize,cudaMemcpyHostToDevice);
//        cout <<"      "<<kernel<<endl;

        // Specify block size
        const dim3 block(BLOCK_SIZE,BLOCK_SIZE);

        // Calculate grid size to cover the whole image
        const dim3 grid((output.cols + block.x - 1)/block.x, (output.rows + block.y - 1)/block.y);

        // Start time
        cudaEventRecord(start,0);

        laplacianFilter<<<grid,block>>>((__half*)d_input, (__half*)d_output, output.cols, output.rows, (__half*)d_kernel);

        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        printf("Kernel time: %.4f ms\n", milliseconds);

        //Copy data from device memory to output image

        cudaMemcpy(output.ptr(),d_output,outputSize,cudaMemcpyDeviceToHost);
//        cout <<output<<endl;

        //Free the device memory
//        cudaFree(d_input);
//        cudaFree(d_output);
//        cudaFree(d_kernel);


}


/////////////////////////////////////////////////////////////////////copy////////////////////////////////////////////////////////////////////////


const int indataLength = sizeof(__half) * IMG_HEIGHT * IMG_WIDTH;
const int outdataLength = sizeof(__half) * IMG_HEIGHT * IMG_WIDTH;
const int kerneldataLength = sizeof(__half) * FILTER_WIDTH * FILTER_HEIGHT;
__half* d_indata, * d_outdata; //指针类型虽然不影响，内存分配大小，但是严谨使用，应该与目标数据的类型（fp16）一至
__half* d_kerneldata;

bool first = true;

__global__ void convfp16_kernel(const __half* srcImage, __half* dstImage, const __half* kernel,const unsigned int width, const unsigned int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;  //block在grid中的序号 block在x维度的大小 线程在一个block中的序号
    int y = blockIdx.y * blockDim.y + threadIdx.y;   //线程的全局index

       // only threads inside image will write results
    if ((x >= FILTER_WIDTH / 2) && (x < (width - FILTER_WIDTH / 2)) && (y >= FILTER_HEIGHT / 2) && (y < (height - FILTER_HEIGHT / 2)))
    {
        // Sum of pixel values
        __half sum = 0.0;
        // Loop inside the filter to average pixel values
        for (int ky = -FILTER_HEIGHT / 2; ky <= FILTER_HEIGHT / 2; ky++)
        {
            for (int kx = -FILTER_WIDTH / 2; kx <= FILTER_WIDTH / 2; kx++)
            {
                __half a = __hmul(srcImage[((y + ky) * width + (x + kx))],\
                    kernel[(ky + FILTER_HEIGHT / 2) * FILTER_WIDTH + kx + FILTER_WIDTH / 2]);
                sum = __hadd(sum, a);
            }
        }
        dstImage[(y * width + x)] = sum;
    }
//    else if (x < width && y < height)
//    {
//        //dstImage[(y * width + x)] = srcImage[(y * width + x)]*0.3;
//        dstImage[(y * width + x)] = __hmul(srcImage[(y * width + x)], 0.4); //cuda fp16一定要用自己的加减乘除
//    }

}

extern "C" void convfp16(const __half* indata, __half* outdata, const __half* kerneldata, const int width, const int height)
{

    // Use cuda event to catch time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    if (first)
    {
        // Allocate device memory
        //cudaMalloc<unsigned char>(&d_input, inputSize);
        //cudaMalloc<unsigned char>(&d_output, outputSize);
        //cudaMalloc<float>(&d_kernel, kernelSize);
        //反复在GPU上开辟内存会浪费大量时间
        cudaMalloc((void**)&d_indata, indataLength);
        cudaMalloc((void**)&d_outdata, outdataLength);
        cudaMalloc((void**)&d_kerneldata, kerneldataLength);
        first = false;
    }

    // Copy data from OpenCV input image to device memory
    cudaMemcpy(d_indata, indata, indataLength, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kerneldata, kerneldata, kerneldataLength, cudaMemcpyHostToDevice);

    // Specify block size
    const dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    // Calculate grid size to cover the whole image
    const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    //Start time
    cudaEventRecord(start, 0);

    convfp16_kernel << <grid, block >> > (d_indata, d_outdata, d_kerneldata, width, height); //右值直接赋值

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("Kernel time: %.2f ms\n", milliseconds);

    //Copy data from device memory to output image
    cudaMemcpy(outdata, d_outdata, outdataLength, cudaMemcpyDeviceToHost);

    //Mat indataMat =Mat(height, width, CV_16FC1, (uchar*)indata); //还识别不了__half*,const应该都可以做右值
    //cout << indataMat(Rect(10, 10, 10, 10));


    //Free the device memory
//        cudaFree(d_input);
//        cudaFree(d_output);
//        cudaFree(d_kernel);


}


/////////////////////////////////////////////////////////////////////copy////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////after vs debug cufft mulSpectrum////////////////////////////////////////////////////////////////////////

/*
核函数在访问内存时会自动结束函数
频谱相乘之后的逆变换的图像会被截取到不同的位置
*/



//Nvidia cufft https://docs.nvidia.com/cuda/cufft/index.html#cufft-setup
/// <summary>
/// 对核函数傅立叶变换，kernel  real (M,N)->complex (M,N/2+1)
///
/// 2D	C2C	  (N1,N2)     cufftComplex	(N1,N2)     cufftComplex
/// 2D	C2R	  (N1,N2/2+1) cufftComplex	(N1,N2)     cufftReal
/// 2D	R2C	  (N1,N2)     cufftReal	    (N1,N2/2+1) cufftComplex
///  N1,nx <=> h; N2,ny <=> w
///
/// </summary>
/// <param name="indata"></变换之前的实数数据，h*w*sizeof(cufftReal) bytes>
/// <param name="outdata"></变换之后的复数数据，h*(w/2+1)*sizeof(cufftComplex) bytes>
/// <param name="kh"></ height of image(or kernel),both have the same size>
/// <param name="kw"></weight of image(or kernel)>

void fftKernel(cufftReal* indata, cufftComplex* outdata, const unsigned int kh, const unsigned int kw)
{
    //imgh == kh
    //imgw == kw
    cufftReal* indata_dev;
    cufftComplex* outdata_dev;
    cudaMalloc((void**)&indata_dev, sizeof(cufftReal) * kh * kw);
    cudaMalloc((void**)&outdata_dev, sizeof(cufftComplex) * kh * (kw / 2 + 1));
    cudaMemcpy(indata_dev, indata, sizeof(cufftReal) * kw * kh, cudaMemcpyHostToDevice);

    cufftHandle planForward;
    cufftPlan2d(&planForward, kh, kw, CUFFT_R2C);
    cufftExecR2C(planForward, indata_dev, outdata_dev);
    cudaThreadSynchronize();

    cudaMemcpy(outdata, outdata_dev, sizeof(cufftComplex) * kh * (kw / 2 + 1), cudaMemcpyDeviceToHost);

    cufftDestroy(planForward);
    cudaFree(outdata_dev);
    cudaFree(indata_dev);
}


/// <summary>
/// 频谱相乘的核函数，卷积运算 互相关运算
/// </summary>
/// <param name="imgdata_dev"></在设备上的图像数据>
/// <param name="outdata_dev"></必须out-of-place存储计算后的数据，否则会有污染>
/// <param name="kdata_dev"></卷积核模板，可以始终放在设备上>
/// <param name="imgh"></height of image>
/// <param name="imgw"></width of image>
/// <returns></returns>

__global__ void kernel_convfft(const cufftComplex* imgdata_dev, cufftComplex* outdata_dev, const cufftComplex* kdata_dev, const unsigned int imgh, const unsigned int imgw)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    //    if(x< imgw && y < imgh/2 )
    //    {
    //        imgdata_dev[y*imgw + x].x = imgdata_dev[y*imgw + x].x * kdata_dev[y*imgw + x].x;
    //        imgdata_dev[y*imgw + x].y = imgdata_dev[y*imgw + x].y * kdata_dev[y*imgw + x].y;
    //    }
    //    else if(y >= imgh/2 && x <imgw && y <imgh)
    //    {
    //        imgdata_dev[(imgh-1-y)*imgw + (imgw-1-x)].x = imgdata_dev[(imgh-1-y)*imgw + (imgw-1-x)].x * kdata_dev[(imgh-1-y)*imgw + (imgw-1-x)].x;
    //        imgdata_dev[(imgh-1-y)*imgw + (imgw-1-x)].y = imgdata_dev[(imgh-1-y)*imgw + (imgw-1-x)].y * kdata_dev[(imgh-1-y)*imgw + (imgw-1-x)].y;
    //    }
    //if (x == 0 || y == 0)
    //{
    //	imgdata_dev[y * imgw + x].x = imgdata_dev[y * imgw + x].x * kdata_dev[y * imgw + x].x - imgdata_dev[y * imgw + x].y * kdata_dev[y * imgw + x].y;
    //	imgdata_dev[y * imgw + x].y = imgdata_dev[y * imgw + x].y * kdata_dev[y * imgw + x].x + imgdata_dev[y * imgw + x].x * kdata_dev[y * imgw + x].y;
    //}
    //else if (y > 0 && x > 0 && x < (imgw / 2 + 1))
    //{
    //	imgdata_dev[y * imgw + x].x = imgdata_dev[y * imgw + x].x * kdata_dev[y * imgw + x].x - imgdata_dev[y * imgw + x].y * kdata_dev[y * imgw + x].y;
    //	imgdata_dev[y * imgw + x].y = imgdata_dev[y * imgw + x].y * kdata_dev[y * imgw + x].x + imgdata_dev[y * imgw + x].x * kdata_dev[y * imgw + x].y;
    //}
    unsigned int h = imgh;
    unsigned int w = imgw / 2 + 1;
    if (x < w && y < h) //设备上访问非法内存的线程会被停止
    {
        //float ri = imgdata_dev[y * w + x].x;
        //float ii = imgdata_dev[y * w + x].y;
        //float rk = kdata_dev[y * w + x].x;
        //float ik = kdata_dev[y * w + x].y;  //核的虚数部位

        //if ()
        //{
            outdata_dev[y * w + x].x = imgdata_dev[y * w + x].x * kdata_dev[y * w + x].x - imgdata_dev[y * w + x].y * kdata_dev[y * w + x].y; //原来写成imgw
            outdata_dev[y * w + x].y = imgdata_dev[y * w + x].y * kdata_dev[y * w + x].x + imgdata_dev[y * w + x].x * kdata_dev[y * w + x].y;
        //}
        //else if (x == 0 && y == 0)
        //{
        //	imgdata_dev[y * w + x].x = imgdata_dev[y * w + x].x * kdata_dev[y * w + x].x;
        //}
    }
}

bool first1 = true;
cufftReal* imgdata_dev;
cufftReal* outdata_dev;
cufftComplex* outdataComplex_dev;
cufftComplex* imgdataComplex_dev;
cufftComplex* kdata_dev;
cufftHandle planForward, planInverse;

void fftImgKernel(const cufftReal* imgdata, cufftReal* outdata, const cufftComplex* kdata, const unsigned int imgh, const unsigned int imgw)
{
    if (first1)
    {
        //img
        cudaMalloc((void**)&imgdata_dev, sizeof(cufftReal) * imgh * imgw);
        cudaMalloc((void**)&outdata_dev, sizeof(cufftReal) * imgh * imgw);
        //img complex
        cudaMalloc((void**)&imgdataComplex_dev, sizeof(cufftComplex) * imgh * (imgw / 2 + 1));//数据要对齐，前面可能就是数据错误
        cudaMalloc((void**)&outdataComplex_dev, sizeof(cufftComplex) * imgh * (imgw / 2 + 1));
        //kernel
        cudaMalloc((void**)&kdata_dev, sizeof(cufftComplex) * imgh * (imgw / 2 + 1));
        //handle
        cufftPlan2d(&planForward, imgh, imgw, CUFFT_R2C);
        cufftPlan2d(&planInverse, imgh, imgw, CUFFT_C2R);
        first1 = false;
    }
    //memory copy kernel and image from host to device
    cudaMemcpy(kdata_dev, kdata, sizeof(cufftComplex) * imgh * (imgw / 2 + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(imgdata_dev, imgdata, sizeof(cufftReal) * imgh * imgw , cudaMemcpyHostToDevice);
    //fft forward transform
    cufftExecR2C(planForward, imgdata_dev, imgdataComplex_dev);
    cudaThreadSynchronize();
    //multiply kernel and image
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((imgw/2 + 1 + block.x - 1) / block.x, (imgh + block.y - 1) / block.y);
    kernel_convfft << <grid, block >> > (imgdataComplex_dev, outdataComplex_dev, kdata_dev, imgh, imgw);
    //fft inverse transform
    cufftExecC2R(planInverse, outdataComplex_dev, outdata_dev);
    cudaThreadSynchronize();
    cudaMemcpy(outdata, outdata_dev, sizeof(cufftReal) * imgh * imgw, cudaMemcpyDeviceToHost);
    //Mat outImg = Mat(imgh, imgw, CV_32F, outdata);
    //cout << outImg(Rect(0, 0, 15, 15)) << endl;
    //    cudaFree(imgdata_dev);
    //    cudaFree(imgdataComplex_dev);
    //    cudaFree(kdata_dev);
    //    cufftDestroy(planForward);
    //    cufftDestroy(planInverse);
}

/////////////////////////////////////////////////////////////////////after vs debug cufft mulSpectrum///////////////////////////////////////////////////////
