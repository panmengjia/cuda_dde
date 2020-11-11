#include "main.h"

//cufft_examples/src/fp16_common.hpp
#include "string"





void fftKernel(void* indata,void* outdata,const unsigned int kh,const unsigned int kw)
{
    cufftReal* indata_dev;
    cufftComplex* outdata_dev;
    cudaMalloc((void**)&indata_dev,sizeof(cufftReal)*kh*kw);
    cudaMalloc((void**)&outdata_dev,sizeof(cufftComplex)*kh*(kw/2+1));
    cudaMemcpy(indata_dev,indata,sizeof(cufftReal)*kw*kh,cudaMemcpyHostToDevice);

    cufftHandle planForward;
    cufftPlan2d(&planForward,kh,kw,CUFFT_R2C);
    cufftExecR2C(planForward,indata_dev,outdata_dev);
    cudaThreadSynchronize();

    cudaMemcpy(outdata,outdata_dev,sizeof(cufftReal)*kh*kw,cudaMemcpyDeviceToHost);

    cufftDestroy(planForward);
    cudaFree(outdata_dev);
    cudaFree(indata_dev);
}




void fftQt1(void* indata,void* outdata,const unsigned int heigth,const unsigned int width)
{
    cufftComplex* dataComplex_dev;
    cufftReal* indata_dev;
    cufftReal* outdata_dev;
    cudaMalloc((void**)&dataComplex_dev,sizeof(cufftComplex)*heigth*(width/2+1));
    cudaMalloc((void**)&indata_dev,sizeof(cufftReal)*heigth*width);
    cudaMalloc((void**)&outdata_dev,sizeof(cufftReal)*heigth*width);
    cudaMemcpy(indata_dev,indata,sizeof(cufftReal)*heigth*width,cudaMemcpyHostToDevice);
    cufftHandle planForward,planInverse;

    cufftPlan2d(&planForward,heigth,width,CUFFT_R2C);
    cufftPlan2d(&planInverse,heigth,width,CUFFT_C2R);

    cufftExecR2C(planForward,indata_dev,dataComplex_dev);
    cudaThreadSynchronize();
    cufftExecC2R(planInverse,dataComplex_dev,outdata_dev);
    cudaThreadSynchronize();
    cudaMemcpy(outdata,outdata_dev,sizeof(cufftReal)*heigth*width,cudaMemcpyDeviceToHost);

    cufftDestroy(planForward);
    cufftDestroy(planInverse);
    cudaFree(dataComplex_dev);
    cudaFree(indata_dev);
}

void mainfftQT1()
{
    Mat img = imread("/home/nvidia/Downloads/8700e1c96cbbdbfc8bb32a700fd8fc85.jpg",IMREAD_GRAYSCALE);
    imshow("img",img);
    if( img.empty() )
    {
        std::cout <<"Cannot read image file"<<std::endl;
    }

    int M = getOptimalDFTSize( img.rows );                               // 获得最佳DFT尺寸，为2的次方
    int N = getOptimalDFTSize( img.cols );
    Mat padded;
    copyMakeBorder(img, padded, 0, M - img.rows, 0, N - img.cols, BORDER_CONSTANT, Scalar::all(0));   // opencv中的边界扩展函数，提供多种方式扩展

    Mat planes = Mat_<float>(padded);

    Mat paddedInverse = Mat::zeros(planes.size(),CV_32FC1);
    fftQt1(planes.data,paddedInverse.data,planes.rows,planes.cols);
//   for(int i=0;i<planes.rows;++i)
//   {
//       for(int j=0;j<planes.cols;++j)
//       {
//          paddedInverse.at<float>(i,j) = paddedInverse.at<float>(i,j)/(planes.rows*planes.cols);
//       }
//   }
   paddedInverse.convertTo(paddedInverse,CV_8UC1,1.0/(float)(planes.rows*planes.cols));
   imshow("paddedInverse",paddedInverse);




}



#define BLOCKIZE 32
//@

__global__ void kernel_convfft(cufftComplex* imgdata_dev,cufftComplex* kdata_dev,const unsigned int imgh,const unsigned int imgw)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x< imgw && y < imgh/2 )
    {
        imgdata_dev[y*imgw + x].x = imgdata_dev[y*imgw + x].x * kdata_dev[y*imgw + x].x;
        imgdata_dev[y*imgw + x].y = imgdata_dev[y*imgw + x].y * kdata_dev[y*imgw + x].y;
    }
    else if(y >= imgh/2 && x <imgw && y <imgh)
    {
        imgdata_dev[(imgh-1-y)*imgw + (imgw-1-x)].x = imgdata_dev[(imgh-1-y)*imgw + (imgw-1-x)].x * kdata_dev[(imgh-1-y)*imgw + (imgw-1-x)].x;
        imgdata_dev[(imgh-1-y)*imgw + (imgw-1-x)].y = imgdata_dev[(imgh-1-y)*imgw + (imgw-1-x)].y * kdata_dev[(imgh-1-y)*imgw + (imgw-1-x)].y;
    }

}

bool first =true;
cufftReal* imgdata_dev;
cufftComplex* imgdataComplex_dev;
cufftComplex* kdata_dev;
cufftHandle planForward,planInverse;

void fftImgKernel(void* imgdata,void* kdata,const unsigned int imgh,const unsigned int imgw)
{
    if(first)
    {
        //img

        cudaMalloc((void**)&imgdata_dev,sizeof(cufftReal)*imgh*imgw);


        //img complex

        cudaMalloc((void**)&imgdataComplex_dev,sizeof(cufftComplex)*imgh*imgw);

        //kernel

        cudaMalloc((void**)&kdata_dev,sizeof(cufftComplex)*imgh*imgw);



        //handle

        cufftPlan2d(&planForward,imgh,imgw,CUFFT_R2C);
        cufftPlan2d(&planInverse,imgh,imgw,CUFFT_C2R);
        first =false;
    }

    cudaMemcpy(kdata_dev,kdata,sizeof(cufftComplex)*imgh*imgw,cudaMemcpyHostToDevice);
    cudaMemcpy(imgdata_dev,imgdata,sizeof(cufftReal)*imgh*imgw,cudaMemcpyHostToDevice);


    //fft forward exec
    cufftExecR2C(planForward,imgdata_dev,imgdataComplex_dev);
    cudaThreadSynchronize();

    //multiply kernel and image
    dim3 block(BLOCKIZE,BLOCKIZE);
    dim3 grid((imgw + block.x - 1) / block.x, (imgh + block.y - 1) / block.y);
    kernel_convfft<<<grid,block>>>(imgdataComplex_dev,kdata_dev,imgh,imgw);

    cufftExecC2R(planInverse,imgdataComplex_dev,imgdata_dev);
    cudaMemcpy(imgdata,imgdata_dev,sizeof(cufftReal)*imgh*imgw,cudaMemcpyDeviceToHost);

//    cudaFree(imgdata_dev);
//    cudaFree(imgdataComplex_dev);
//    cudaFree(kdata_dev);
//    cufftDestroy(planForward);
//    cufftDestroy(planInverse);
}
