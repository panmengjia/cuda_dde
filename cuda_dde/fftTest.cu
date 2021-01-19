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

////////////////////////////////////////////////////////cuda dft mulSpectrum////////////////////////////////////////////////////////////////////

/*
核函数在访问内存时会自动结束函数
频谱相乘之后的逆变换的图像会被截取到不同的位置

*/



//Nvidia cufft https://docs.nvidia.com/cuda/cufft/index.html#cufft-setup
/// <summary>
/// 对核函数傅立叶变换，kernel  real (M,N)->complex (M,N/2+1)
/// 
/// 2D	C2C	  (N1,N2)     cufftComplex	(N1,N2)     cufftComplex
/// 2D	C2R	  (N1,N2/2+1) cufftComplex	(N1,N2)     cufftReal
/// 2D	R2C	  (N1,N2)     cufftReal	    (N1,N2/2+1) cufftComplex
///  N1,nx <=> h; N2,ny <=> w
/// 
/// </summary>
/// <param name="indata"></变换之前的实数数据，h*w*sizeof(cufftReal) bytes>
/// <param name="outdata"></变换之后的复数数据，h*(w/2+1)*sizeof(cufftComplex) bytes>
/// <param name="kh"></ height of image(or kernel),both have the same size>
/// <param name="kw"></weight of image(or kernel)>

void fftKernel(cufftReal* indata, cufftComplex* outdata, const unsigned int kh, const unsigned int kw)
{
	//imgh == kh
	//imgw == kw
	cufftReal* indata_dev;
	cufftComplex* outdata_dev;
	cudaMalloc((void**)&indata_dev, sizeof(cufftReal) * kh * kw);
	cudaMalloc((void**)&outdata_dev, sizeof(cufftComplex) * kh * (kw / 2 + 1));
	cudaMemcpy(indata_dev, indata, sizeof(cufftReal) * kw * kh, cudaMemcpyHostToDevice);

	cufftHandle planForward;
	cufftPlan2d(&planForward, kh, kw, CUFFT_R2C);
	cufftExecR2C(planForward, indata_dev, outdata_dev);
	cudaThreadSynchronize();

	cudaMemcpy(outdata, outdata_dev, sizeof(cufftComplex) * kh * (kw / 2 + 1), cudaMemcpyDeviceToHost);

	cufftDestroy(planForward);
	cudaFree(outdata_dev);
	cudaFree(indata_dev);
}


/// <summary>
/// 频谱相乘的核函数，卷积运算 互相关运算
/// </summary>
/// <param name="imgdata_dev"></在设备上的图像数据>
/// <param name="outdata_dev"></必须out-of-place存储计算后的数据，否则会有污染>
/// <param name="kdata_dev"></卷积核模板，可以始终放在设备上>
/// <param name="imgh"></height of image>
/// <param name="imgw"></width of image>
/// <returns></returns>

__global__ void kernel_convfft(const cufftComplex* imgdata_dev, cufftComplex* outdata_dev, const cufftComplex* kdata_dev, const unsigned int imgh, const unsigned int imgw)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	//    if(x< imgw && y < imgh/2 )
	//    {
	//        imgdata_dev[y*imgw + x].x = imgdata_dev[y*imgw + x].x * kdata_dev[y*imgw + x].x;
	//        imgdata_dev[y*imgw + x].y = imgdata_dev[y*imgw + x].y * kdata_dev[y*imgw + x].y;
	//    }
	//    else if(y >= imgh/2 && x <imgw && y <imgh)
	//    {
	//        imgdata_dev[(imgh-1-y)*imgw + (imgw-1-x)].x = imgdata_dev[(imgh-1-y)*imgw + (imgw-1-x)].x * kdata_dev[(imgh-1-y)*imgw + (imgw-1-x)].x;
	//        imgdata_dev[(imgh-1-y)*imgw + (imgw-1-x)].y = imgdata_dev[(imgh-1-y)*imgw + (imgw-1-x)].y * kdata_dev[(imgh-1-y)*imgw + (imgw-1-x)].y;
	//    }
	//if (x == 0 || y == 0)
	//{
	//	imgdata_dev[y * imgw + x].x = imgdata_dev[y * imgw + x].x * kdata_dev[y * imgw + x].x - imgdata_dev[y * imgw + x].y * kdata_dev[y * imgw + x].y;
	//	imgdata_dev[y * imgw + x].y = imgdata_dev[y * imgw + x].y * kdata_dev[y * imgw + x].x + imgdata_dev[y * imgw + x].x * kdata_dev[y * imgw + x].y;
	//}
	//else if (y > 0 && x > 0 && x < (imgw / 2 + 1))
	//{
	//	imgdata_dev[y * imgw + x].x = imgdata_dev[y * imgw + x].x * kdata_dev[y * imgw + x].x - imgdata_dev[y * imgw + x].y * kdata_dev[y * imgw + x].y;
	//	imgdata_dev[y * imgw + x].y = imgdata_dev[y * imgw + x].y * kdata_dev[y * imgw + x].x + imgdata_dev[y * imgw + x].x * kdata_dev[y * imgw + x].y;
	//}
	unsigned int h = imgh;
	unsigned int w = imgw / 2 + 1;
	if (x < w && y < h) //设备上访问非法内存的线程会被停止
	{
		//float ri = imgdata_dev[y * w + x].x;
		//float ii = imgdata_dev[y * w + x].y;
		//float rk = kdata_dev[y * w + x].x;
		//float ik = kdata_dev[y * w + x].y;  //核的虚数部位

		//if ()
		//{
			outdata_dev[y * w + x].x = imgdata_dev[y * w + x].x * kdata_dev[y * w + x].x - imgdata_dev[y * w + x].y * kdata_dev[y * w + x].y; //原来写成imgw
			outdata_dev[y * w + x].y = imgdata_dev[y * w + x].y * kdata_dev[y * w + x].x + imgdata_dev[y * w + x].x * kdata_dev[y * w + x].y;
		//}
		//else if (x == 0 && y == 0)
		//{
		//	imgdata_dev[y * w + x].x = imgdata_dev[y * w + x].x * kdata_dev[y * w + x].x;
		//}
	}
}

bool first1 = true;
cufftReal* imgdata_dev;
cufftReal* outdata_dev;
cufftComplex* outdataComplex_dev;
cufftComplex* imgdataComplex_dev;
cufftComplex* kdata_dev;
cufftHandle planForward, planInverse;

void fftImgKernel(const cufftReal* imgdata, cufftReal* outdata, const cufftComplex* kdata, const unsigned int imgh, const unsigned int imgw)
{
	if (first1)
	{
		//img
		cudaMalloc((void**)&imgdata_dev, sizeof(cufftReal) * imgh * imgw);
		cudaMalloc((void**)&outdata_dev, sizeof(cufftReal) * imgh * imgw);
		//img complex
		cudaMalloc((void**)&imgdataComplex_dev, sizeof(cufftComplex) * imgh * (imgw / 2 + 1));//数据要对齐，前面可能就是数据错误
		cudaMalloc((void**)&outdataComplex_dev, sizeof(cufftComplex) * imgh * (imgw / 2 + 1));
		//kernel
		cudaMalloc((void**)&kdata_dev, sizeof(cufftComplex) * imgh * (imgw / 2 + 1));
		//handle
		cufftPlan2d(&planForward, imgh, imgw, CUFFT_R2C);
		cufftPlan2d(&planInverse, imgh, imgw, CUFFT_C2R);
		first1 = false;
	}
    //memory copy kernel and image from host to device
	cudaMemcpy(kdata_dev, kdata, sizeof(cufftComplex) * imgh * (imgw / 2 + 1), cudaMemcpyHostToDevice);
	cudaMemcpy(imgdata_dev, imgdata, sizeof(cufftReal) * imgh * imgw , cudaMemcpyHostToDevice);
	//fft forward transform
	cufftExecR2C(planForward, imgdata_dev, imgdataComplex_dev);
	cudaThreadSynchronize();
	//multiply kernel and image
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((imgw/2 + 1 + block.x - 1) / block.x, (imgh + block.y - 1) / block.y);
	kernel_convfft << <grid, block >> > (imgdataComplex_dev, outdataComplex_dev, kdata_dev, imgh, imgw);
	//fft inverse transform
	cufftExecC2R(planInverse, outdataComplex_dev, outdata_dev);
	cudaThreadSynchronize();
	cudaMemcpy(outdata, outdata_dev, sizeof(cufftReal) * imgh * imgw, cudaMemcpyDeviceToHost);
	//Mat outImg = Mat(imgh, imgw, CV_32F, outdata);
	//cout << outImg(Rect(0, 0, 15, 15)) << endl;
	//    cudaFree(imgdata_dev);
	//    cudaFree(imgdataComplex_dev);
	//    cudaFree(kdata_dev);
	//    cufftDestroy(planForward);
	//    cufftDestroy(planInverse);
}


void ycbcrUpdate1119(const Mat& IM_result_cbcr, const Mat& IM_bri_T, Mat& IM_result_cbcr_re);

vector<vector<float>> extractConvMat01119();

int testcudamulSpectrum()
{
	VideoCapture* cap = new VideoCapture("C:\\Users\\b515\\Desktop\\pmj\\cuda_dde\\fenglin\\130 (3).avi");
	if (!cap->isOpened())
	{
		cout << "video is empty!!" << endl;
	}
	Mat frame; //1080 1920
	//discard the first 25 frames
	while (cap->read(frame))
	{
		static unsigned int counter = 0;
		if (++counter == 25)
		{
			break;
		}
	}
	Mat meanFrame, stdDevFrame, singleChannel/*IM_bri*/;
	double mean;
	vector<Mat> splitFrame2bgr3;

	Mat meanBri, stdDevBri;
	double thresholdRate;

	Mat IM_bri_T0;
	//
	vector<vector<float>> temp = extractConvMat01119();
	
	for (int i = 0; i < 15 * 15; i++)
	{
		if (i % 15 == 0) cout << endl;
		cout << temp[6][i] << " ";
	}

	vector<Mat> hsfatMat;
	for (int i = 0; i < 15; i++)
	{
		hsfatMat.push_back(Mat(FILTER_HEIGHT, FILTER_WIDTH, CV_32FC1, temp[i].data()));
	}
	cout << endl;
	cout << hsfatMat[6] << endl;
	cap->read(frame);
	int fM = getOptimalDFTSize(frame.rows);                               // 获得最佳DFT尺寸，为2的次方
	int fN = getOptimalDFTSize(frame.cols);

	cufftComplex* kernelComplex[15];
	Mat kernelpadded[15];
	for (int i = 0; i < 15; ++i)
	{
		kernelpadded[i] = Mat(fM, fN, CV_32FC1);
		kernelComplex[i] = (cufftComplex*)malloc(sizeof(cufftComplex) * fM * (fN / 2 + 1));
		//        copyMakeBorder(hsfatMat[i],kernelpadded[i], fM/2 - hsfatMat[i].rows/2-1, fM/2 - hsfatMat[i].rows/2, fN/2 - hsfatMat[i].cols/2-1, fN/2 - hsfatMat[i].cols/2, BORDER_CONSTANT, Scalar::all(0));   // opencv中的边界扩展函数，提供多种方式扩展
		copyMakeBorder(hsfatMat[i], kernelpadded[i], 0, fM - hsfatMat[i].rows, 0, fN - hsfatMat[i].cols, BORDER_CONSTANT, Scalar::all(0));   // opencv中的边界扩展函数，提供多种方式扩展
		fftKernel((cufftReal*)kernelpadded[i].data, kernelComplex[i], kernelpadded[i].rows, kernelpadded[i].cols);
	}

	cufftComplex* kernelComplex6 = (cufftComplex*)malloc(sizeof(cufftComplex) * fM * (fN / 2 + 1));
	Mat kernelpadded6;
	copyMakeBorder(hsfatMat[6], kernelpadded6, 0, fM - hsfatMat[6].rows, 0, fN - hsfatMat[6].cols, BORDER_CONSTANT, Scalar::all(0));   // opencv中的边界扩展函数，提供多种方式扩展
	fftKernel((cufftReal*)kernelpadded6.data, kernelComplex6, kernelpadded6.rows, kernelpadded6.cols);
	//cout << kernelpadded6 << endl;
	cout << hsfatMat[6] << endl;

	//for (int i = 0; i < fM; i++)
	//{
	//	for (int j = 0; j < (fN / 2 + 1); j++)
	//	{
	//		cout <<kernelComplex6[i * (fN / 2 + 1) + j].x << endl;
	//	}
	//}
	Mat kernelComplex6Mat = Mat(fM, (fN / 2 + 1), CV_32FC2, kernelComplex6);
	cout << kernelComplex6Mat(Rect(0,0,15,15)) << endl;

	Mat planes[] = { kernelpadded6,Mat::zeros(kernelpadded6.size(),CV_32F) };
	Mat complexImg;
	merge(planes, 2, complexImg);
	dft(complexImg, complexImg);



	while (cap->read(frame))  //type CV_8UC3
	{
		double time = (double)getTickCount();

		meanStdDev(frame, meanFrame, stdDevFrame);
		mean /*results_de_mean*/ = (meanFrame.at<Vec3d>(0, 0)[0] + meanFrame.at<Vec3d>(0, 0)[1] + meanFrame.at<Vec3d>(0, 0)[2]) / 3;
		split(frame, splitFrame2bgr3);
		singleChannel = 0.257 * splitFrame2bgr3[2] + 0.564 * splitFrame2bgr3[1] + 0.098 * splitFrame2bgr3[0] + 0.0;
		meanStdDev(singleChannel, meanBri, stdDevBri);
		thresholdRate = meanBri.at<double>(0, 0) / stdDevBri.at<double>(0, 0) * meanBri.at<double>(0, 0) / 80;
		thresholdRate += 0.2;
		if (thresholdRate > 0.6)
		{
			thresholdRate = 0.6;
		}
		cout << "thresholdRate*10:   " << thresholdRate * 10 << endl;
		Mat H_S_f_A_T = hsfatMat[(unsigned int)(thresholdRate * 10)];

		IM_bri_T0 = singleChannel;

		Mat IM_bri_T = Mat::zeros(IM_bri_T0.size(), IM_bri_T0.type());
		//        imshow("IM_bri_T",IM_bri_T0);

		int M = getOptimalDFTSize(IM_bri_T.rows);                               // 获得最佳DFT尺寸，为2的次方
		int N = getOptimalDFTSize(IM_bri_T.cols);

		Mat padded;
		copyMakeBorder(singleChannel, padded, 0, M - IM_bri_T0.rows, 0, N - IM_bri_T0.cols, BORDER_CONSTANT, Scalar::all(0));   // opencv中的边界扩展函数，提供多种方式扩展

//        Mat planes = Mat_<float>(padded);

//        imshow("planes",planes);
		padded.convertTo(padded, CV_32FC1, 1.0 / 255.0);
		//        imshow("padded",padded);

		Mat paddedInverse = Mat::zeros(padded.size(), CV_32FC1);
		//        fftQt1(planes.data,paddedInverse.data,planes.rows,planes.cols);
		fftImgKernel((cufftReal*)padded.data, (cufftReal*)paddedInverse.data, kernelComplex6/*[(unsigned int)(thresholdRate * 10)]*/, padded.rows, padded.cols);
		//imshow("paddedInverse", paddedInverse);
		//paddedInverse = padded;
		imshow("paddedInverse", paddedInverse);
		paddedInverse.convertTo(paddedInverse, CV_8UC1, 255/1920.0/1080.0);

		imshow("paddedInverse1", paddedInverse);

		//      laplacianFilter_GPU_wrapper(IM_bri_T0, IM_bri_T, H_S_f_A_T);

		//      filter2D(IM_bri_T0,IM_bri_T,CV_64FC1,H_S_f_A_T);

		//        Mat meanOfIM_bri_T, stdDevOfIM_bri_T;
		//        meanStdDev(IM_bri_T, meanOfIM_bri_T, stdDevOfIM_bri_T);
		//        IM_bri_T = IM_bri_T / meanOfIM_bri_T.at<double>(0, 0) * meanBri.at<double>(0, 0) / 1;

		//        Mat IM_result_cbcr;
		//        cvtColor(frame, IM_result_cbcr, COLOR_BGR2YCrCb);
		//        Mat IM_result_cbcr_re;
		//        ycbcrUpdate(IM_result_cbcr, IM_bri_T, IM_result_cbcr_re);

		Mat meanOfIM_bri_T, stdDevOfIM_bri_T;
		meanStdDev(paddedInverse, meanOfIM_bri_T, stdDevOfIM_bri_T);
		paddedInverse = paddedInverse / meanOfIM_bri_T.at<double>(0, 0) * meanBri.at<double>(0, 0) / 1;

		Mat IM_result_cbcr;
		cvtColor(frame, IM_result_cbcr, COLOR_BGR2YCrCb);
		Mat IM_result_cbcr_re;
		ycbcrUpdate1119(IM_result_cbcr, paddedInverse, IM_result_cbcr_re);

		imshow("frame", frame);
		imshow("IM_result_cbcr_re", IM_result_cbcr_re);
		waitKey(1);
		double fps = getTickFrequency() / (getTickCount() - time);
		cout << "fps" << fps << endl;
	}

	delete cap;
	delete[] kernelComplex;
	return 0;
}




void ycbcrUpdate1119(const Mat& IM_result_cbcr, const Mat& IM_bri_T, Mat& IM_result_cbcr_re)
{
	vector<Mat> channelsOfIM;
	split(IM_result_cbcr, channelsOfIM);
	Mat IM_bri_T_8U;
	IM_bri_T.convertTo(IM_bri_T_8U, CV_8UC1);
	channelsOfIM[0] = IM_bri_T_8U;
	merge(channelsOfIM, IM_result_cbcr_re);
	cvtColor(IM_result_cbcr_re, IM_result_cbcr_re, COLOR_YCrCb2BGR);
}

vector<vector<float>> extractConvMat01119()
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
////////////////////////////////////////////////////////cuda dft mulSpectrum////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////cuda dft mulSpectrum副本////////////////////////////////////////////////////////////////////
/*
* cuda dft mulSpectrum
* 2020.11.19继承前面使用cuda核函数写的频谱相乘，修改错误，规范代码
* 
* 
* 
*/

#define CVTCOLOR (true)

void cudaFFTmulSpectrum1119float()
{
	/////////////////////////////////////////////读取txt////////////////////////////////////////////
	const string str = "C:/Users/b515/Desktop/pmj/cuda_dde/fenglin/" + to_string(FILTER_WIDTH) + "/";
	vector<vector<float>> dataAllFile;
	vector<float> dataPerFile;
	float dataElement;
	//vector<Mat> kernelMat;
	for (int i = 0; i < 15; ++i)
	{
		//vector<float> dataPerFile;
		ifstream dataFile(str + to_string(i + 1) + ".txt");
		while (dataFile >> dataElement)
		{
			dataPerFile.push_back(dataElement);
		}
		dataFile.close();
		dataAllFile.push_back(dataPerFile);
		dataPerFile.clear();
	}

	vector<Mat> kernelMat;
	cufftComplex* kernelComplex[15];
	Mat kernelPadded;

	for (int i = 0; i < 15; ++i) //在外面只能得到最后一次的数据结果，之前的都被覆盖了
	{
		kernelComplex[i] = (cufftComplex*)malloc(sizeof(cufftComplex) * IMG_HEIGHT * (IMG_WIDTH / 2 + 1));
		kernelMat.push_back(Mat(FILTER_HEIGHT, FILTER_WIDTH, CV_32FC1, dataAllFile[i].data()));
		copyMakeBorder(kernelMat[i], kernelPadded, 0, IMG_HEIGHT - kernelMat[i].rows, 0, IMG_WIDTH - kernelMat[i].cols, BORDER_CONSTANT, Scalar(0)); //Scalar::all(0)
		fftKernel((cufftReal*)kernelPadded.data, kernelComplex[i], kernelPadded.rows, kernelPadded.cols);/*IMG_HEIGHT, IMG_WIDTH);*/
		//kernelMat[i].convertTo(kernelMat[i], CV_16FC1);
	}

	cout << kernelMat[6] << endl;
	//测试代码
	//测试fftKernel与dft结果的异同，是完全一样的
	//opencv dft的complex形式与cufft的cudaComplex的数据结构是一样的 
	//行优先存储，连续
	cufftComplex* kernelComplex6 = (cufftComplex*)malloc(sizeof(cufftComplex) * IMG_HEIGHT * (IMG_WIDTH / 2 + 1));
	Mat kernelPadded6;
	copyMakeBorder(kernelMat[6], kernelPadded6, 0, IMG_HEIGHT - kernelMat[6].rows, 0, IMG_WIDTH - kernelMat[6].cols, BORDER_CONSTANT, Scalar::all(0));   // opencv中的边界扩展函数，提供多种方式扩展
	fftKernel((cufftReal*)kernelPadded6.data, kernelComplex6, kernelPadded6.rows, kernelPadded6.cols);
	//cout << kernelpadded6 << endl;
	cout << kernelMat[6] << endl;

	//通道连续存储，两者可以完美对应
	Mat kernelComplex6Mat = Mat(IMG_HEIGHT, (IMG_WIDTH / 2 + 1), CV_32FC2, kernelComplex6);
	cout << kernelComplex6Mat(Rect(0, 0, 15, 15)) << endl;

	Mat planes[] = { kernelPadded6,Mat::zeros(kernelPadded6.size(),CV_32F) };
	Mat complexImg;
	merge(planes, 2, complexImg);
	dft(complexImg, complexImg);
	/////////////////////////////////////////////读取txt//////////////////////////////////////////// 

	/////////////////////////////////////////////DDEfilter////////////////////////////////////////////
	VideoCapture* cap = new VideoCapture("C:\\Users\\b515\\Desktop\\pmj\\cuda_dde\\fenglin\\130 (3).avi");
	if (!cap->isOpened())
	{
		cout << "video is empty!!" << endl;
	}
	Mat frame; //1080 1920
	Mat outFrame, singleFrame, meanSF, stdDevSF, selectedKernelMat, SFfp32, SFfp32out, frameYUV;
	vector<Mat> frame3channels, frame3YUV;
	double threshRate;

	while (1)
	{
		double time = (double)getTickCount();
		double time2 = (double)getTickCount();
		cap->read(frame);
#if CVTCOLOR
		cvtColor(frame, singleFrame, COLOR_BGR2GRAY);
#else   
		split(frame, frame3channels);
		singleFrame = 0.257 * frame3channels[2] + 0.564 * frame3channels[1] + 0.098 * frame3channels[0];
#endif
		meanStdDev(singleFrame, meanSF, stdDevSF);

		threshRate = meanSF.at<double>(0, 0) / stdDevSF.at<double>(0, 0) * meanSF.at<double>(0, 0) / 80 + 0.2;
		if (threshRate > 0.6)
		{
			threshRate = 0.6;
		}
		cout << "threshRate*10: " << threshRate * 10 << endl;
		selectedKernelMat = kernelMat[(unsigned __int64)(threshRate * 10)];
		time2 = (double)(getTickCount() - time2) * 1000 / getTickFrequency();
		cout << "   time while pre   =  " << time2 << "  ms" << endl;

		time2 = (double)getTickCount();
		//singleFrame.convertTo(SFfp16, CV_16FC1);
		//SFfp16out = Mat(SFfp16.size(), CV_16FC1); //已开辟内存，但是随机数，都是-23.0
		singleFrame.convertTo(SFfp32, CV_32FC1);
		SFfp32out = Mat(SFfp32.size(), CV_32FC1); //已开辟内存，但是随机数，都是-23.0

		time2 = (double)(getTickCount() - time2) * 1000 / getTickFrequency();
		cout << "   time creat SFfp32out convertTo SFfp32Frame   =  " << time2 << "  ms" << endl;

		time2 = (double)getTickCount();
		/* laplacianFilter_GPU_wrapper(SFfp16, SFfp16out, selectedKernelMat);*/
		 //convfp16((__half*)SFfp16.data, (__half*)SFfp16out.data, (__half*)selectedKernelMat.data, SFfp16.cols, SFfp16.rows); //.ptr() .data返回的都是一级指针（列指针）

		//timeCall = (double)getTickCount();
		//convolveDFT(SFfp32, selectedKernelMat, SFfp32out);

		fftImgKernel((cufftReal*)SFfp32.data, (cufftReal*)SFfp32out.data, kernelComplex[(unsigned int)(threshRate * 10)], SFfp32.rows, SFfp32.cols);

		time2 = (double)(getTickCount() - time2) * 1000 / getTickFrequency();
		cout << "   time convDFT   =  " << time2 << "  ms" << endl;

		time2 = (double)getTickCount();
		//SFfp32out.convertTo(SFfp32out, CV_8UC1);
		SFfp32out.convertTo(SFfp32out, CV_8UC1,1/((float)IMG_HEIGHT*(float)IMG_WIDTH)); //真是不懂到底要不要乘以255
		time2 = (double)(getTickCount() - time2) * 1000 / getTickFrequency();
		cout << "   time convertTo SFfp32out fp32->8u  =  " << time2 << "  ms" << endl;

		time2 = (double)getTickCount();
		Scalar meanSFfp32out = mean(SFfp32out);
		SFfp32out = SFfp32out / meanSFfp32out[0] * meanSF.at<double>(0, 0) / 1;
		time2 = (double)(getTickCount() - time2) * 1000 / getTickFrequency();
		cout << "   time mean of SF32fp32out  =  " << time2 << "  ms" << endl;

		time2 = (double)getTickCount();
		cvtColor(frame, frameYUV, COLOR_BGR2YCrCb);
		split(frameYUV, frame3YUV);
		frame3YUV[0] = SFfp32out;
		merge(frame3YUV, frameYUV);
		cvtColor(frameYUV, outFrame, COLOR_YCrCb2BGR);
		time2 = (double)(getTickCount() - time2) * 1000 / getTickFrequency();
		cout << "   time cvtColor yuv  =  " << time2 << "  ms" << endl;

		time2 = (double)getTickCount();
		imshow("frame", frame);
		imshow("outFrame", outFrame);
		time2 = (double)(getTickCount() - time2) * 1000 / getTickFrequency();
		cout << "   time imshow  =  " << time2 << "  ms" << endl;


		double fps = getTickFrequency() / (getTickCount() - time);
		time = (double)(getTickCount() - time) * 1000 / getTickFrequency();
		cout << "  time total =  " << time << "  ms" << endl;
		cout << "fps" << fps << endl;
		waitKey(1);
	}

	delete cap;
	/////////////////////////////////////////////DDEfilter////////////////////////////////////////////
}

////////////////////////////////////////////////////////cuda dft mulSpectrum副本////////////////////////////////////////////////////////////////////