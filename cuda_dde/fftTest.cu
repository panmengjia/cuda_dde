#include "main.h"


void fft1d()
{
	const int signalLength = 10;
	cufftReal* signal = (cufftReal*)malloc(sizeof(cufftReal) * signalLength);
	for (int i = 0; i < 10; i++)
	{
		*signal++ = i + 1;
	}
	cufftReal* signalDev;
	cudaMalloc((void**)&signalDev, sizeof(cufftReal) * signalLength);
	cudaMemcpy(signalDev, signal, sizeof(cufftReal) * signalLength, cudaMemcpyDeviceToHost);

	cufftHandle 




}