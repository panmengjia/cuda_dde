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


vector<vector<float>> extractConvMatfft()
{
	const string& str = "C:/Users/b515/Desktop/pmj/cuda_dde/fenglin/15/";
	vector<vector<float>> HVSFT;
	HVSFT.resize(15);
	//    unsigned int counter = 0;
	for (int i = 1; i < 16; i++)
	{
		ifstream dataFile(str + to_string(i) + ".txt");
		float dataElement;
		while (dataFile >> dataElement)
		{
			HVSFT[i - 1].push_back(dataElement);
		}
		dataFile.close();
	}
	return HVSFT;
}

__global__ void kernel_normalize2d()
{

}

//该过程对两个二维数组做运算
//二级指针测试不行
void fft2d()
{
	vector<vector<float>> dataArray = extractConvMatfft();
	//for (int i = 0;  i < 15 * 15; i++)
	//{
	//	cout << *(dataArray[0].data() +i) << "  ";  //dataArray[0][i]  *(dataArray[0].data() +i)居然不需要强制类型转换
	//}
	const unsigned int matSize = 20;
	cufftReal (*mat)[5];
	mat = (cufftReal(*)[5])malloc(sizeof(cufftReal) * matSize);  //4row 5col
	for (int i = 0; i < 20; i++)
	{
		//*(*(mat + i/5)+i%5) = i + 1;
		*(*mat + i) = i + 1; //行指针变量肯定存的是行地址，指针变量的值就是地址
		//cout << mat[i / 5][i % 5] << endl;
	}
	cufftReal(*mat_dev)[5];
	mat_dev = NULL;
	cudaMalloc((void**)mat_dev, sizeof(cufftReal [5]) * matSize/5); //cufftReal
	cudaMemcpy(*mat_dev, *mat, sizeof(cufftReal) * matSize, cudaMemcpyHostToDevice);
	cufftComplex(*matForward)[5];
	matForward = (cufftComplex(*)[5])malloc(sizeof(cufftReal [5]) * matSize/5);  //对地址进行强制类型转换，应该都是对右值强制类型转换
	cufftComplex(*matForward_dev)[5];
	matForward_dev = NULL;
	cudaMalloc((void**)matForward_dev, sizeof(cufftComplex) * matSize);
	//matForward_dev = (cufftComplex(*)[5])malloc(sizeof(cufftComplex) * matSize);  //对地址进行强制类型转换，应该都是对右值强制类型转换


	cufftHandle plan;
	/*
	二维FFT算法实现中，同一维FFT不同的是：
    (1) 输入参数：没有BATCH，增加了NY。NX为行数，NY为列数；
    (2) FFT的正变换的输入位置和输出位置不同。*/
	cufftPlan2d(&plan, 4, 5, CUFFT_R2C);  //  行 列
	cufftExecR2C(plan,(cufftReal*)*mat_dev, (cufftComplex*)*matForward_dev);
	cudaMemcpy((cufftComplex*)*matForward, (cufftComplex*)*matForward_dev, sizeof(cufftComplex) * matSize, cudaMemcpyDeviceToHost);
	for (int i = 0; i < matSize; ++i)
	{
		cout << (*matForward + i)->x << " " << (*matForward + i)->y << endl;
	}


	cudaFree(*mat_dev);
	cudaFree(*matForward_dev);
	free(*mat);
	
}

void fft3d()
{
	vector<vector<float>> dataArray = extractConvMatfft();
	//for (int i = 0;  i < 15 * 15; i++)
	//{
	//	cout << *(dataArray[0].data() +i) << "  ";  //dataArray[0][i]  *(dataArray[0].data() +i)居然不需要强制类型转换
	//}
	const unsigned int matSize = 16;
	cufftComplex* mat;
	mat = (cufftComplex*)malloc(sizeof(cufftComplex) * matSize);  //5row 4col
	for (int i = 0; i < matSize; i++)
	{
		(*(mat + i)).x = (i + 1); //行指针变量肯定存的是行地址，指针变量的值就是地址
		(mat + i)->y = 0;
		cout << mat[i].x << endl;
	}
	cufftComplex*  mat_dev;  //N
	mat_dev = NULL;
	cudaMalloc((void**)&mat_dev, sizeof(cufftComplex) * matSize);  // //cufftComplex
	cudaMemcpy(mat_dev, mat, sizeof(cufftComplex) * matSize, cudaMemcpyHostToDevice); 
	cufftComplex* matForward; //R2C N->N/2+1
	matForward = (cufftComplex*)malloc(sizeof(cufftComplex) * matSize);  //对地址进行强制类型转换，应该都是对右值强制类型转换
	cufftComplex* matForward_dev; //R2C N->N/2+1
	matForward_dev = NULL;
	cudaMalloc((void**)&matForward_dev, sizeof(cufftComplex) * matSize); //R2C N->N/2+1


	cufftHandle plan;
	/*
	二维FFT算法实现中，同一维FFT不同的是：
	(1) 输入参数：没有BATCH，增加了NY。NX为行数，NY为列数；
	(2) FFT的正变换的输入位置和输出位置不同。*/
	cufftPlan2d(&plan, 4, 4, CUFFT_C2C);  //  行 列
	cufftExecC2C(plan, (cufftComplex*)mat_dev, (cufftComplex*)matForward_dev,CUFFT_FORWARD);
	cudaMemcpy((cufftComplex*)matForward, (cufftComplex*)matForward_dev, sizeof(cufftComplex) * matSize, cudaMemcpyDeviceToHost);
	for (int i = 0; i < matSize; ++i)
	{
		//if (i == 0) cout << endl;
		//if (i % 4== 0) cout << endl;
		cout << (matForward + i)->x << " " << (matForward + i)->y << endl;
	}

	//逆变换
	//cufftHandle planInv;
	//cufftPlan1d(&planInv,11, CUFFT_C2R,1);

	cudaFree(mat_dev);
	cudaFree(matForward_dev);
	free(mat);
	free(matForward);

}