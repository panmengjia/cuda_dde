#include "main.h"

//cufft_examples/src/fp16_common.hpp


void fftQt1(uchar* indata,uchar* outdata,const unsigned int heigth,const unsigned int width)
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



