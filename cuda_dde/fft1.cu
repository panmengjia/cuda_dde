#include "main.h"


void fft1test(uchar* indata, uchar* outdata, const int height, const int width)
{
	//����cuda����fft�任ʱ������һЩ�������õĹ���һ�¾�������˵����
	float* h_Data = (float*)indata; //"h_"�� host����ʾCPU�ڴ�
	float* d_Data; //"d_"��device����ʾGPU�ڴ�

	cufftComplex* d_DataSpectrum; //cufftComplex��Ϊfloat������ʽ��xΪʵ����yΪ����
	cufftHandle fftPlanFwd, fftPlanInv;   //Fwd��ʾ���任��Inv�任���任

	const int  dataH = height; //��άͼ��ĸ߶�
	const int  dataW = width;  //��άͼ��Ŀ��
	int fftH = dataH; //��dataH ��Ϊ2���ݴ�������Ҫ����ͼ����չ
	int fftW = dataW; //��dataW ��Ϊ2���ݴ�������Ҫ����ͼ����չ
	h_Data = (float*)malloc(dataH * dataW * sizeof(float)); //CPU���ڴ���䷽ʽ
	cudaMalloc((void**)&d_Data, dataH * dataW * sizeof(float)); //GPU���ڴ濪�̷�ʽ
	cudaMalloc((void**)&d_DataSpectrum, fftH * (fftW / 2 + 1) * sizeof(cufftComplex)); //cufft��R2C�任Ҫ������ΪH*W�����ΪH*(W/2+1)��W���뱣֤Ϊ2�ı���


	//��cpu�е��ڴ��h_Data��ֵ�������Ϊ˵���������getRand()�����������������

	cufftPlan2d(&fftPlanFwd, fftH, fftW, CUFFT_R2C);
	//����fft�任�ľ����������Ҫע�⣬����Ĳ�����ΪfftH��fftW�����ȴ��߶�ֵ���ٴ����ֵ��CUFFT_R2C��ʾ��realʵ���任��complex����
	cufftPlan2d(&fftPlanInv, fftH, fftW, CUFFT_C2R);
	//����fft���任�ľ������ͬ���任������Ĳ�����ΪfftH��fftW�����ȴ��߶�ֵ���ٴ����ֵ��CUFFT_C2R��ʾ��complex�����任��realʵ��

	cudaMemcpy(d_Data, h_Data, dataH * dataW * sizeof(float), cudaMemcpyHostToDevice);
	//��CPU�ڴ�����ݿ�copy��GPU���Դ��У�cudaMemcpyHostToDevice��ʾ��host��copy��device�ˣ�cudaMemcpyDeviceToHost��ʾ��device��copy��host��

	cufftExecR2C(fftPlanFwd, (cufftReal*)d_Data, (cufftComplex*)d_DataSpectrum);
	//����FFT���任�ľ��������ͼ�����ݵ�FFT�任��ע������ľ��ʹ�õ���fftPlanFwd

	//�˴�Ϊ������Ƶ���в����ĺ���
	cufftExecC2R(fftPlanInv, (cufftComplex*)d_DataSpectrum, (cufftReal*)d_Data);//����Ƶ���д���Ľ�����ٷ��任�ؿռ����У�ע������ľ��ʹ�õ���fftPlanInv
	cudaMemcpy(outdata, d_Data, sizeof(float) * dataH * dataW, cudaMemcpyDeviceToHost);

	//Release
	cufftDestroy(fftPlanInv); //���پ������ӦcufftPlan2d
	cufftDestroy(fftPlanFwd);//���پ������ӦcufftPlan2d
	cudaFree(d_DataSpectrum);//GPU���ڴ��ͷţ���ӦcudaMalloc
	cudaFree(d_Data);//GPU���ڴ��ͷţ���ӦcudaMalloc
	free(h_Data);//CPU���ڴ��ͷţ���Ӧmalloc
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
		//��任
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