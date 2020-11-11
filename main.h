#ifndef MAIN_H__
#define MAIN_H__

#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>

#include <fstream>
#include <string>
#include <stdio.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include <cufft.h>  //要添加cufft.lib cufftw.lib库文件，否则报错无法解析cudafft错误

using namespace cv;
using namespace std;


#define KH (15)  //heigth of kernel
#define KW (15)

extern int opencvdft(int argc, const char ** argv);
extern void mainfftQT1();
extern void maindde();
extern void fftImgKernel(void* imgdata,void* kdata,const unsigned int imgh,const unsigned int imgw);
extern void fftKernel(void* indata,void* outdata,const unsigned int kh,const unsigned int kw);
extern void fftQt1(void* indata,void* outdata,const unsigned int heigth,const unsigned int width);
extern void fftImgKernel(void* imgdata,void* kdata,const unsigned int imgh,const unsigned int imgw);



#endif
