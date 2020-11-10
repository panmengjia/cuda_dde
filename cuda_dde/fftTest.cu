#include "main.h"

//c2c c2r r2c https://blog.csdn.net/qq_17239003/article/details/79090803
//�ܺõĲ������£�https://zhuanlan.zhihu.com/p/34587739


__global__ void kernel_normalize1d(cufftComplex* signal,const int signalLength)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < signalLength) //�Ǳ�����ʷǷ��ڴ���
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

	//��任
	cufftComplex* signalInverse = (cufftComplex*)malloc(sizeof(cufftComplex) * signalLength);
	cufftComplex* signalInvDev;
	cudaMalloc((void**)&signalInvDev, sizeof(cufftComplex) * signalLength);
	for (int i = 0; i < signalLength; i++)  //ȫ����ʼ��Ϊ0������y in-place������д������ݣ�Ҳ���Բ�����y�е�����
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
	//�˹�һ������
	//���Բ������йܷ�ʽ
	const unsigned int blockX = 32;
	// Specify block size
	const dim3 block(blockX, 1);
	const dim3 grid(1);


	// Start time
	cudaEventRecord(start, 0);  //��ʱ����
	kernel_normalize1d <<< grid, block >>> (signalInvDev,signalLength);//����ʵ���������߳�����32
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

//�ù��̶�������ά����������
//����ָ����Բ���
void fft2d()
{
	vector<vector<float>> dataArray = extractConvMatfft();
	//for (int i = 0;  i < 15 * 15; i++)
	//{
	//	cout << *(dataArray[0].data() +i) << "  ";  //dataArray[0][i]  *(dataArray[0].data() +i)��Ȼ����Ҫǿ������ת��
	//}
	const unsigned int matSize = 20;
	cufftReal (*mat)[5];
	mat = (cufftReal(*)[5])malloc(sizeof(cufftReal) * matSize);  //4row 5col
	for (int i = 0; i < 20; i++)
	{
		//*(*(mat + i/5)+i%5) = i + 1;
		*(*mat + i) = i + 1; //��ָ������϶�������е�ַ��ָ�������ֵ���ǵ�ַ
		//cout << mat[i / 5][i % 5] << endl;
	}
	cufftReal(*mat_dev)[5];
	mat_dev = NULL;
	cudaMalloc((void**)mat_dev, sizeof(cufftReal [5]) * matSize/5); //cufftReal
	cudaMemcpy(*mat_dev, *mat, sizeof(cufftReal) * matSize, cudaMemcpyHostToDevice);
	cufftComplex(*matForward)[5];
	matForward = (cufftComplex(*)[5])malloc(sizeof(cufftReal [5]) * matSize/5);  //�Ե�ַ����ǿ������ת����Ӧ�ö��Ƕ���ֵǿ������ת��
	cufftComplex(*matForward_dev)[5];
	matForward_dev = NULL;
	cudaMalloc((void**)matForward_dev, sizeof(cufftComplex) * matSize);
	//matForward_dev = (cufftComplex(*)[5])malloc(sizeof(cufftComplex) * matSize);  //�Ե�ַ����ǿ������ת����Ӧ�ö��Ƕ���ֵǿ������ת��


	cufftHandle plan;
	/*
	��άFFT�㷨ʵ���У�ͬһάFFT��ͬ���ǣ�
    (1) ���������û��BATCH��������NY��NXΪ������NYΪ������
    (2) FFT�����任������λ�ú����λ�ò�ͬ��*/
	cufftPlan2d(&plan, 4, 5, CUFFT_R2C);  //  �� ��
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
	//	cout << *(dataArray[0].data() +i) << "  ";  //dataArray[0][i]  *(dataArray[0].data() +i)��Ȼ����Ҫǿ������ת��
	//}
	const unsigned int matSize = 16;
	cufftComplex* mat;
	mat = (cufftComplex*)malloc(sizeof(cufftComplex) * matSize);  //5row 4col
	for (int i = 0; i < matSize; i++)
	{
		(*(mat + i)).x = (i + 1); //��ָ������϶�������е�ַ��ָ�������ֵ���ǵ�ַ
		(mat + i)->y = 0;
		cout << mat[i].x << endl;
	}
	cufftComplex*  mat_dev;  //N
	mat_dev = NULL;
	cudaMalloc((void**)&mat_dev, sizeof(cufftComplex) * matSize);  // //cufftComplex
	cudaMemcpy(mat_dev, mat, sizeof(cufftComplex) * matSize, cudaMemcpyHostToDevice); 
	cufftComplex* matForward; //R2C N->N/2+1
	matForward = (cufftComplex*)malloc(sizeof(cufftComplex) * matSize);  //�Ե�ַ����ǿ������ת����Ӧ�ö��Ƕ���ֵǿ������ת��
	cufftComplex* matForward_dev; //R2C N->N/2+1
	matForward_dev = NULL;
	cudaMalloc((void**)&matForward_dev, sizeof(cufftComplex) * matSize); //R2C N->N/2+1


	cufftHandle plan;
	/*
	��άFFT�㷨ʵ���У�ͬһάFFT��ͬ���ǣ�
	(1) ���������û��BATCH��������NY��NXΪ������NYΪ������
	(2) FFT�����任������λ�ú����λ�ò�ͬ��*/
	cufftPlan2d(&plan, 4, 4, CUFFT_C2C);  //  �� ��
	cufftExecC2C(plan, (cufftComplex*)mat_dev, (cufftComplex*)matForward_dev,CUFFT_FORWARD);
	cudaMemcpy((cufftComplex*)matForward, (cufftComplex*)matForward_dev, sizeof(cufftComplex) * matSize, cudaMemcpyDeviceToHost);
	for (int i = 0; i < matSize; ++i)
	{
		//if (i == 0) cout << endl;
		//if (i % 4== 0) cout << endl;
		cout << (matForward + i)->x << " " << (matForward + i)->y << endl;
	}

	//��任
	//cufftHandle planInv;
	//cufftPlan1d(&planInv,11, CUFFT_C2R,1);

	cudaFree(mat_dev);
	cudaFree(matForward_dev);
	free(mat);
	free(matForward);

}