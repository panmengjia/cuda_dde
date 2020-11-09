//#include "stdafx.h"
#include <windows.h>
#include <windowsx.h>
//#include <onnxruntime_cxx_api.h>
//#include <cuda_provider_factory.h>
//#include <onnxruntime_c_api.h>
//#include <tensorrt_provider_factory.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <stdlib.h> 
#include <iostream> 

using namespace cv;
using namespace std;
using namespace cv::cuda;

int main()
{
	cuda::printCudaDeviceInfo(cuda::getDevice());
	int count = cuda::getCudaEnabledDeviceCount();
	printf("GPU Device Count : %d \n", count);

	return 0;
}

