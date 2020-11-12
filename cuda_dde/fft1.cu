#include "main.h"


void fft1test(uchar* indata, uchar* outdata, const int height, const int width)
{
	//利用cuda进行fft变换时，会有一些参数设置的规则，一下举例进行说明：
	float* h_Data = (float*)indata; //"h_"： host，表示CPU内存
	float* d_Data; //"d_"：device，表示GPU内存

	cufftComplex* d_DataSpectrum; //cufftComplex：为float复数形式，x为实数，y为复数
	cufftHandle fftPlanFwd, fftPlanInv;   //Fwd表示正变换，Inv变换反变换

	const int  dataH = height; //二维图像的高度
	const int  dataW = width;  //二维图像的宽度
	int fftH = dataH; //若dataH 不为2的幂次数，需要进行图像扩展
	int fftW = dataW; //若dataW 不为2的幂次数，需要进行图像扩展
	h_Data = (float*)malloc(dataH * dataW * sizeof(float)); //CPU端内存分配方式
	cudaMalloc((void**)&d_Data, dataH * dataW * sizeof(float)); //GPU端内存开盘方式
	cudaMalloc((void**)&d_DataSpectrum, fftH * (fftW / 2 + 1) * sizeof(cufftComplex)); //cufft中R2C变换要求输入为H*W，输出为H*(W/2+1)，W必须保证为2的倍数


	//对cpu中的内存块h_Data赋值，这里仅为说明，因此用getRand()函数进行随机数产生

	cufftPlan2d(&fftPlanFwd, fftH, fftW, CUFFT_R2C);
	//设置fft变换的句柄，这里需要注意，传入的参数先为fftH，fftW，即先传高度值，再传宽度值，CUFFT_R2C表示从real实数变换到complex复数
	cufftPlan2d(&fftPlanInv, fftH, fftW, CUFFT_C2R);
	//设置fft反变换的句柄，雷同正变换，传入的参数先为fftH，fftW，即先传高度值，再传宽度值，CUFFT_C2R表示从complex复数变换到real实数

	cudaMemcpy(d_Data, h_Data, dataH * dataW * sizeof(float), cudaMemcpyHostToDevice);
	//将CPU内存的数据块copy到GPU的显存中，cudaMemcpyHostToDevice表示从host端copy到device端，cudaMemcpyDeviceToHost表示从device端copy到host端

	cufftExecR2C(fftPlanFwd, (cufftReal*)d_Data, (cufftComplex*)d_DataSpectrum);
	//按照FFT正变换的句柄，进行图像数据的FFT变换，注意这里的句柄使用的是fftPlanFwd

	//此处为其它在频域中操作的函数
	cufftExecC2R(fftPlanInv, (cufftComplex*)d_DataSpectrum, (cufftReal*)d_Data);//将在频域中处理的结果，再反变换回空间域中，注意这里的句柄使用的是fftPlanInv
	cudaMemcpy(outdata, d_Data, sizeof(float) * dataH * dataW, cudaMemcpyDeviceToHost);

	//Release
	cufftDestroy(fftPlanInv); //销毁句柄，对应cufftPlan2d
	cufftDestroy(fftPlanFwd);//销毁句柄，对应cufftPlan2d
	cudaFree(d_DataSpectrum);//GPU中内存释放，对应cudaMalloc
	cudaFree(d_Data);//GPU中内存释放，对应cudaMalloc
	free(h_Data);//CPU中内存释放，对应malloc
}

void fft2main()
{

	double time = getTickCount();
	Mat img = imread("C:\\Users\\b515\\Downloads\\image.jpg", IMREAD_GRAYSCALE);
	//resize(img, img, Size(512, 512));
	imshow("img",img);
	img.convertTo(img, CV_32FC1,1.0/255.0);
	Mat outImg = Mat(img.rows, img.cols, CV_32FC1);
	fft1test(img.data, outImg.data, img.rows, img.cols);
	//outImg.convertTo(outImg, CV_8UC1,255.0);
	imshow("outImg", outImg);
	time = (getTickCount() - time)*1000 / getTickFrequency();
	cout << "time = " << time << endl;
	waitKey(1);
}


cv::Mat real(cv::Mat img)
{
	std::vector<cv::Mat> planes;
	cv::split(img, planes);
	return planes[0];
}

int ff1main()/*int argc, char** argv)*/
{

	Mat img = imread("01.jpg", IMREAD_GRAYSCALE);

	//int NX = 2560;
	//int NY = 2560;
	int NN = 1000;

	//if (argc == 4)
	//{
	//	NX = atoi(argv[1]);
	//	NY = atoi(argv[2]);
	//	NN = atoi(argv[3]);
	//}

	//std::cout << "NX=" << NX << " ; NY=" << NY << " ; NN=" << NN << std::endl;

	resize(img, img, Size(24, 24));
	cout << "img: " << img.channels() << " " << img.rows << "  " << img.cols << endl;
	imshow("img", img);
	waitKey(500);

	int rows = img.rows;
	int cols = img.cols;

	//normalize(img, img, 0, 1, CV_MINMAX);
	img.convertTo(img, CV_32FC1, 1.0f / 255);
	//cout << img(Rect(0, 0, 10, 96)) << endl;
	cufftHandle planFwd;
	cufftHandle planInv;
	float* data;
	float* data2;
	cufftComplex* res;
	cudaMalloc((float**)&data, sizeof(float) * rows * cols);
	cudaMalloc((float**)&data2, sizeof(float) * rows * cols);

	cudaMalloc((void**)&res, sizeof(cufftComplex) * rows * (cols / 2 + 1));

	/* Try to do the same thing than cv::randu() */
	float* host_data;
	host_data = (float*)malloc(sizeof(float) * rows * cols);

	//host_data = (float *)img.data;
	//srand(time(NULL));
	for (int i = 0; i < rows; i++)
	{
		float* ptrs = img.ptr<float>(i);
		for (size_t j = 0; j < cols; j++)
		{
			host_data[i * cols + j] = ptrs[j];
		}
		//cout << img(Rect(0, 0, 1, 2)) << endl;
		//cout << sizeof(img.data[0])<< " " << (float)img.data[1]<< endl;
		//host_data[i] = make_cuComplex(rand() % 256, rand() % 256);
		//host_data[i].x = rand() % 256;
		//host_data[i].y = rand() % 256;
	}
	//cout << host_data[0]<< endl;



	/* Warm up ? */
	/* Create a 3D FFT plan. */
	cufftPlan2d(&planFwd, rows, cols, CUFFT_R2C);
	cufftPlan2d(&planInv, rows, cols, CUFFT_C2R);
	//cufftPlan2d(&planInv, rows, cols, CUFFT_C2C);

	cufftComplex* dftRes = (cufftComplex*)malloc(sizeof(cufftComplex) * rows * (cols / 2 + 1));


	double t = cv::getTickCount();
	for (size_t i = 0; i < NN; i++)
	{
		cudaMemcpy(data, host_data, sizeof(float) * rows * cols, cudaMemcpyHostToDevice);
		/* Transform the first signal in place. */
		cufftExecR2C(planFwd, (cufftReal*)data, (cufftComplex*)res);

		cudaMemcpy(dftRes, res, sizeof(cufftComplex) * rows * (cols / 2 + 1), cudaMemcpyDeviceToHost);
		//for (size_t i = 0; i < 5; i++)
		//{
		//	cout << "gpu: " << i << "  " << dftRes[i].x << " " << dftRes[i].y << endl;

		//}
		//逆变换
		cufftExecC2R(planInv, (cufftComplex*)res, (cufftReal*)data2);
		float* new_data = (float*)malloc(sizeof(float) * rows * cols);
		cudaMemcpy(new_data, data2, sizeof(float) * rows * cols, cudaMemcpyDeviceToHost);
		//cout << new_data[0] / (rows*cols) << " " << new_data[1] / (rows*cols) << endl;
		//for (size_t i = 0; i < rows*cols; i++)
		//{
		//	new_data[i] = new_data[i] / (rows*cols);
		//}
		//Mat gpu_res = Mat(rows, cols, CV_32FC1, new_data);
		//imshow("gpu:", gpu_res);
		//waitKey(2);
	}
	t = 1000 * ((double)cv::getTickCount() - t) / cv::getTickFrequency() / NN;
	std::cout << "Cuda time=" << t << " ms" << std::endl;


	//use  CPU
	cout << "cpu img:" << img(Rect(0, 0, 1, 2)) << endl;
	Mat img2, newimg2, dst;

	double t1 = cv::getTickCount();
	bool backwards = false;
	for (size_t i = 0; i < NN; i++)
	{
		if (img.channels() == 1)
		{
			cv::Mat planes[] = { cv::Mat_<float>(img), cv::Mat_<float>::zeros(img.size()) };
			//cv::Mat planes[] = {cv::Mat_<double> (img), cv::Mat_<double>::zeros(img.size())};
			cv::merge(planes, 2, img2);
		}
		cv::dft(img2, dst, backwards ? (cv::DFT_INVERSE | cv::DFT_SCALE) : 0); // 0.01ms
		//cout << "cpu dst :" << dst(Rect(0, 0, 5, 1)) << endl;


		backwards = true;
		cv::dft(dst, newimg2, backwards ? (cv::DFT_INVERSE | cv::DFT_SCALE) : 0); // 0.01ms

		//Mat cpu_res = real(newimg2);
		//cout << "cpu newImg2:" << cpu_res(Rect(0, 0, 1, 2)) << endl;
		//imshow("cpu:", cpu_res);
		//waitKey(0);
	}
	t1 = 1000 * ((double)cv::getTickCount() - t1) / cv::getTickFrequency() / NN;
	std::cout << "cpu time=" << t1 << " ms" << std::endl;



















	//double t = cv::getTickCount();
	//for (int i = 0; i < NN; i++)
	//{
	//	/* Create a 2D FFT plan. */
	//	cufftPlan2d(&plan, cols, rows, CUFFT_C2C);

	//	/* Transform the first signal in place. */
	//	cufftExecC2C(plan, data, res, CUFFT_FORWARD);
	//}

	//t = 1000 * ((double)cv::getTickCount() - t) / cv::getTickFrequency() / NN;
	//std::cout << "Cuda time=" << t << " ms" << std::endl;


	/* Destroy the cuFFT plan. */
	cufftDestroy(planFwd);
	cufftDestroy(planInv);
	cudaFree(res);
	cudaFree(data);
	free(host_data);
	getchar();
	std::system("PAUSE");
	return 0;
}