#ifndef MAIN_H__
#define MAIN_H__

#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>

#include <string>
#include <stdio.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include <cufft.h>  //要添加cufft.lib cufftw.lib库文件，否则报错无法解析cudafft错误

using namespace cv;
using namespace std;


extern int opencvdft(int argc, const char ** argv);
extern void mainfftQT1();



#endif
