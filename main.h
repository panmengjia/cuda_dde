#ifndef MAIN_H__
#define MAIN_H__


#define VIDEO_DIR ("130(3).avi")
#define TXT_DIR ("15/")


#include "cuda.h"
#include "cuda_runtime.h"
#include "cufft.h"
#include<fstream>

#define BLOCK_SIZE      (16)
#define FILTER_WIDTH    (15)
#define FILTER_HEIGHT   (15)  //85 vs 2.2seconds  核越大黑色，边框越大

#define IMG_HEIGHT  (1080)
#define IMG_WIDTH   (1920)


extern "C" void fftKernel(cufftReal* indata, cufftComplex* outdata, const unsigned int kh, const unsigned int kw);
extern "C" void fftImgKernel(const cufftReal* imgdata, cufftReal* outdata, const cufftComplex* kdata, const unsigned int imgh, const unsigned int imgw);

#endif

