#include "main.h"

//c2c c2r r2c https://blog.csdn.net/qq_17239003/article/details/79090803
//很好的博客文章：https://zhuanlan.zhihu.com/p/34587739


__global__ void kernel_normalize1d(cufftComplex* signal,const int signalLength)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < signalLength) //是避免访问非法内存吗
	{
		(signal + idx)->x /= signalLength;
		(signal + idx)->y /= signalLength;
	}
}

void fft1d()
{
	// Use cuda event to catch time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	const int signalLength = 15;
	cufftComplex* signal = (cufftComplex*)malloc(sizeof(cufftComplex) * signalLength);
	for (int i = 0; i < signalLength; i++)
	{
		(signal+i)->x = (float)(i + 1);
		(signal+i)->y = 0;
	}
	cufftComplex* signalDev;
	cudaMalloc((void**)&signalDev, sizeof(cufftComplex) * signalLength);
	cudaMemcpy(signalDev, signal, sizeof(cufftComplex) * signalLength, cudaMemcpyHostToDevice);

	cufftHandle plan;
	cufftPlan1d(&plan, signalLength, CUFFT_C2C,1);
	cufftExecC2C(plan, (cufftComplex*)signalDev, (cufftComplex*)signalDev, CUFFT_FORWARD);
	cudaDeviceSynchronize();//wait to be done
	cudaMemcpy(signal, signalDev, signalLength * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

	for (int i = 0; i < signalLength; i++)
	{
		cout << (signal+i)->x << " " << (signal+i)->y << endl;
	}

	//逆变换
	cufftComplex* signalInverse = (cufftComplex*)malloc(sizeof(cufftComplex) * signalLength);
	cufftComplex* signalInvDev;
	cudaMalloc((void**)&signalInvDev, sizeof(cufftComplex) * signalLength);
	for (int i = 0; i < signalLength; i++)  //全部初始化为0，否则y in-place运算会有错吴数据，也可以不考虑y中的数据
	{
		(signal + i)->x = 0;
		(signal + i)->y = 0;
	}
	for (int i = 0; i < signalLength; i++)
	{
		cout << (signalInverse + i)->x << " " << (signalInverse + i)->y << endl;
	}


	cufftHandle planInv;
	cufftPlan1d(&plan, signalLength, CUFFT_C2C, 1);
	cufftExecC2C(plan, (cufftComplex*)signalDev, signalInvDev, CUFFT_INVERSE);
	cudaDeviceSynchronize();
	cudaMemcpy(signalInverse, signalInvDev, signalLength * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
	for (int i = 0; i < signalLength; i++)
	{
		cout << (signalInverse + i)->x << " " << (signalInverse + i)->y << endl;
	}
	//核归一化函数
	//可以测试下托管方式
	const unsigned int blockX = 32;
	// Specify block size
	const dim3 block(blockX, 1);
	const dim3 grid(1);


	// Start time
	cudaEventRecord(start, 0);  //暂时不用
	kernel_normalize1d <<< grid, block >>> (signalInvDev,signalLength);//可能实际启动的线程数是32
	cudaMemcpy(signalInverse, signalInvDev, signalLength * sizeof(cufftComplex),cudaMemcpyDeviceToHost);


	for (int i = 0; i < signalLength; i++)
	{
		cout << (signalInverse + i)->x << " " << (signalInverse + i)->y << endl;
	}

	cufftDestroy(plan);
	free(signal);
	cudaFree(signalDev);

}

void fft2d()
{


}