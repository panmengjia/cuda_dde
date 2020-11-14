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
#include <cufft.h>  //Ҫ���cufft.lib cufftw.lib���ļ������򱨴��޷�����cudafft����


#define BLOCK_SIZE      (8)
#define FILTER_WIDTH    (15)
#define FILTER_HEIGHT   (15)  //85 vs 2.2seconds  ��Խ���ɫ���߿�Խ��

#define IMG_HEIGHT  (1080)
#define IMG_WIDTH   (1920)


using namespace cv;
using namespace std;


extern int maindde();
extern "C" void mainfft();
extern "C" void fft1d();
extern "C" void fft3d();
extern  void fft2main();



#endif