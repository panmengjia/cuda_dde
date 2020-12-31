#ifndef MAIN_H__
#define MAIN_H__


#define VIDEO_DIR ("/home/nvidia/Desktop/dde1448/130(3).avi")
//#define TXT_DIR ("/home/nvidia/Desktop/dde1448/15/")
#define TXT_DIR ("/home/nvidia/Desktop/dde1448/txt129/")



#include "cuda.h"
#include "cuda_runtime.h"
#include "cufft.h"
#include<fstream>

#define BLOCK_SIZE      (32)

#define FILTER_NUM           (60)
#define FILTER_WIDTH_SRC    (129)
#define FILTER_HEIGHT_SRC   (129)

#define FILTER_WIDTH    (9)
#define FILTER_HEIGHT   (9)

#define IMG_HEIGHT  (1080)
#define IMG_WIDTH   (1920)


extern "C" void fftKernel(cufftReal* indata, cufftComplex* outdata, const unsigned int kh, const unsigned int kw);
extern "C" void fftImgKernel(const cufftReal* imgdata, cufftReal* outdata, const cufftComplex* kdata, const unsigned int imgh, const unsigned int imgw);

#endif

