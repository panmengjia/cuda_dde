#pragma once


#ifndef MAIN_H__
#define MAIN_H__


#include <opencv.hpp>
#include <iostream>
//#include <unistd.h>
#include <fstream>

#include "opencv2/opencv.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <stdio.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include <cufft.h>  //要添加cufft.lib cufftw.lib库文件，否则报错无法解析cudafft错误

using namespace cv;
using namespace std;


extern int maindde();
extern "C" void mainfft();



#endif