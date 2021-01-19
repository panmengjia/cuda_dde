#include "main.h"
#include "cuda_fp16.h "

extern "C" void laplacianFilter_GPU_wrapper(const cv::Mat & input, cv::Mat & output, const cv::Mat & kernel);
vector<vector<float>> extractConvMat0();
void ycbcrUpdate(const Mat& IM_result_cbcr, const Mat& IM_bri_T, Mat& IM_result_cbcr_re);


// �鿴Mat����  https://www.cnblogs.com/nipan/p/4098373.html
//vector���ڶ��ڿ��ٿռ䣬������ջ�ڣ����Կ���������ֵ������Ҫ����


int main_(int argc,char** argv)
{
    VideoCapture* cap = new VideoCapture("C:\\Users\\b515\\Desktop\\pmj\\cuda_dde\\fenglin\\130 (3).avi");
    if (!cap->isOpened())
    {
        cout << "video is empty!!" << endl;
    }
    Mat frame; //1080 1920
    Mat meanFrame, stdDevFrame, singleChannel;
    vector<Mat> splitFrame2bgr3;

    Mat meanBri, stdDevBri;
    double thresholdRate;

    Mat IM_bri_T0;
    vector<vector<float>> temp = extractConvMat0();
    vector<Mat> hsfatMat;
    for (int i = 0; i < 15; i++)
    {  
        hsfatMat.push_back(Mat(FILTER_HEIGHT, FILTER_WIDTH, CV_32FC1, temp[i].data()));
        //printf(" ��ַǰ  %p�� %p\n", &hsfatMat[i], hsfatMat[i].data); //convertTo֮���ַ����ͷ��ַ���䣬�������ݵ�ַ�ı�Ҫ���·���ռ�
        printf("sizeof(Mat) %d\n", (int)sizeof(hsfatMat[i]));  //96
        //printf("temp[%d].data()  %p \n", i , temp[i].data());  //vector�ڴ��������ڶ��ϵģ������������
        hsfatMat[i].convertTo(hsfatMat[i], CV_16FC1);
        //printf(" ��ַ��  %p�� %p\n", &hsfatMat[i], hsfatMat[i].data);
    }
    Mat frameGray;
    while (cap->read(frame))  //type CV_8UC3
    {
        double time = (double)getTickCount();
        split(frame, splitFrame2bgr3);
        singleChannel = 0.257 * splitFrame2bgr3[2] + 0.564 * splitFrame2bgr3[1] + 0.098 * splitFrame2bgr3[0] + 0.0;
        meanStdDev(singleChannel, meanBri, stdDevBri);
        thresholdRate = meanBri.at<double>(0, 0) / stdDevBri.at<double>(0, 0) * meanBri.at<double>(0, 0) / 80;
        thresholdRate += 0.2;

        cvtColor(frame, frameGray, COLOR_BGR2GRAY);
        if (thresholdRate > 0.6)
        {
            thresholdRate = 0.6;
        }
        cout << "thresholdRate*10:   " << thresholdRate * 10 << endl;
        Mat H_S_f_A_T = hsfatMat[(unsigned __int64)(thresholdRate * 10)];

        IM_bri_T0 = singleChannel;
        /*IM_bri_T0 = frameGray;*/

       /* Mat IM_bri_T = Mat::zeros(IM_bri_T0.size(), IM_bri_T0.type());*/

        double time2 = (double)getTickCount();
        Mat IM_bri_T00;
        IM_bri_T0.convertTo(IM_bri_T00, CV_16FC1);
        Mat IM_bri_T = Mat::zeros(IM_bri_T0.size(), CV_16FC1);
        laplacianFilter_GPU_wrapper(IM_bri_T00, IM_bri_T, H_S_f_A_T);

        //filter2D(IM_bri_T0,IM_bri_T,CV_64FC1,H_S_f_A_T);
        time2 = (getTickCount() - time2)*1000 / getTickFrequency();
        cout << "time2 = " << time2 << endl;


        //Ҫ�������ȣ�������ԭ����һ��
        //Mat meanOfIM_bri_T, stdDevOfIM_bri_T;
        //mean meanStdDev��������֧��16fp
        //meanStdDev(IM_bri_T, meanOfIM_bri_T, stdDevOfIM_bri_T);
        //IM_bri_T = IM_bri_T / meanOfIM_bri_T.at<double>(0, 0) * meanBri.at<double>(0, 0) / 1;
        Mat IM_bri_T_U8;
        IM_bri_T.convertTo(IM_bri_T_U8, CV_8UC1);
        //printf("CV_8UC1 == CV_8U  %s \n", CV_8UC1 == CV_8U ? "true" : "false");
        Scalar meanOfIM_bri_T = mean(IM_bri_T_U8); //Scalar�̳�������
        IM_bri_T = IM_bri_T / meanOfIM_bri_T[0] * meanBri.at<double>(0, 0) / 1;

        Mat IM_result_cbcr;
        cvtColor(frame, IM_result_cbcr, COLOR_BGR2YCrCb);
        Mat IM_result_cbcr_re;
        ycbcrUpdate(IM_result_cbcr, IM_bri_T, IM_result_cbcr_re);

        imshow("frame", frame);
        imshow("IM_result_cbcr_re", IM_result_cbcr_re);
        waitKey(1);
        double fps = getTickFrequency() / (getTickCount() - time);
        cout << "fps" << fps << endl;
    }
    delete cap;
    return 0;
}

void ycbcrUpdate(const Mat& IM_result_cbcr, const Mat& IM_bri_T, Mat& IM_result_cbcr_re)
{
    vector<Mat> channelsOfIM;
    split(IM_result_cbcr, channelsOfIM);
    Mat IM_bri_T_8U;
    IM_bri_T.convertTo(IM_bri_T_8U, CV_8UC1);
    channelsOfIM[0] = IM_bri_T_8U;
    merge(channelsOfIM, IM_result_cbcr_re);
    cvtColor(IM_result_cbcr_re, IM_result_cbcr_re, COLOR_YCrCb2BGR);
}

vector<vector<float>> extractConvMat0()
{
    const string& str = "C:/Users/b515/Desktop/pmj/cuda_dde/fenglin/"+to_string(FILTER_WIDTH)+"/";
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
        printf(" extraMat0 HVSFT[%d].data()  %p \n", i - 1, HVSFT[i - 1].data());
    }

    return HVSFT;
}


////////////////////////////////////////////////////////////������////////////////////////////////////////////////////////////////
#define CVTCOLOR (true)
extern "C" void convfp16(const __half * indata, __half * outdata, __half * kerneldata, const int width, const int height);


void test()
{  
    /////////////////////////////////////////////��ȡtxt////////////////////////////////////////////
    const string str = "C:/Users/b515/Desktop/pmj/cuda_dde/fenglin/" + to_string(FILTER_WIDTH) + "/";  //""���治�ܶ�һ���ַ�
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
        //dataPerFile.swap(vector<float>());  //swap�����β�Ϊ���ã�ʵ�α�������ֵ void swap(vector& _Right) �ǳ������õĳ�ʼֵ����Ϊ��ֵ
        //vector<float>().swap(dataPerFile);  
        //dataPerFile�ڲ�����Ͳ���Ҫ��ɾ��vectorԪ��
        //�����涨�壬����Ҫɾ��������洢��ֻ�ǵ�һ���ļ�����
        dataPerFile.clear();
        //clear��ȻҲ���Դﵽͬ������ȷЧ����Ӧ�ø�push_back�й�

        //kernelMat.push_back(Mat(FILTER_WIDTH, FILTER_HEIGHT, CV_32FC1, dataPerFile.data()));
        //cout << "========================================================" << endl;
        //cout << "           i          " << i << endl;
        //cout << kernelMat[i] << endl;
        //vector<float>().swap(dataPerFile); //swap�ͻ�ǿ���ͷ��ڴ棬ǿ�ƽ����ڴ�
        ////dataPerFile.clear(); //clear(),����ȥ���������ڴ棬����ֻ�ǽ�vector�����ݽṹĨ��
        //cout << " ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
        //cout << "i                " << i << endl;
        //cout << kernelMat[i] << endl;
    }
    //for (int i = 0; i < 15; i++)
    //{
    //    cout <<"========================================================" << endl;
    //    cout << "           i          " << i << endl;
    //    for (int row = 0; row < FILTER_HEIGHT; ++row)
    //    {
    //        for (int col = 0; col < FILTER_WIDTH; ++col)
    //        {
    //            cout<<dataAllFile[i][row * FILTER_WIDTH + col];
    //        }
    //        cout << endl;
    //    }
    //}
    vector<Mat> kernelMat;
    for (int i = 0; i < 15; ++i) //������ֻ�ܵõ����һ�ε����ݽ����֮ǰ�Ķ���������
    {
        kernelMat.push_back(Mat(FILTER_WIDTH, FILTER_HEIGHT, CV_32FC1, dataAllFile[i].data()));
        kernelMat[i].convertTo(kernelMat[i], CV_16FC1);
        //cout << " ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
        //cout << "---------------i=====================                " << i << endl;
        //cout << kernelMat[i] << endl;

    }
    /////////////////////////////////////////////��ȡtxt//////////////////////////////////////////// 


    VideoCapture* cap = new VideoCapture("C:\\Users\\b515\\Desktop\\pmj\\cuda_dde\\fenglin\\130 (3).avi");
    if (!cap->isOpened())
    {
        cout << "video is empty!!" << endl;
    }
    Mat frame; //1080 1920
    Mat outFrame,singleFrame,meanSF,stdDevSF,selectedKernelMat,SFfp16,SFfp16out,frameYUV;
    vector<Mat> frame3channels,frame3YUV;
    double threshRate;

    while (1)
    {
        double time = (double)getTickCount();
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

        double time2 = (double)getTickCount();
        singleFrame.convertTo(SFfp16, CV_16FC1);
        SFfp16out = Mat(SFfp16.size(), CV_16FC1); //�ѿ����ڴ棬���������������-23.0
        //cout << SFfp16out(Rect(10, 10, 10, 10)) << endl;
       /* laplacianFilter_GPU_wrapper(SFfp16, SFfp16out, selectedKernelMat);*/
        convfp16((__half*)SFfp16.data, (__half*)SFfp16out.data, (__half*)selectedKernelMat.data, SFfp16.cols, SFfp16.rows); //.ptr() .data���صĶ���һ��ָ�루��ָ�룩
        //cout << SFfp16out(Rect(10, 10, 10, 10)) << endl;
        //filter2D(IM_bri_T0,IM_bri_T,CV_64FC1,H_S_f_A_T);

        time2 = (getTickCount() - time2) * 1000 / getTickFrequency();
        cout << "time2 = " << time2 << endl;

        SFfp16out.convertTo(SFfp16out, CV_8UC1);
        Scalar meanSFfp16out = mean(SFfp16out);
        SFfp16out = SFfp16out / meanSFfp16out[0] * meanSF.at<double>(0, 0) / 1;

        cvtColor(frame, frameYUV, COLOR_BGR2YCrCb);
        split(frameYUV, frame3YUV);
        frame3YUV[0] = SFfp16out;
        merge(frame3YUV, frameYUV);
        cvtColor(frameYUV, outFrame, COLOR_YCrCb2BGR);

        imshow("frame", frame);
        imshow("outFrame", outFrame);


        double fps = getTickFrequency() / (getTickCount() - time);
        cout << "fps" << fps << endl;
        waitKey(1);
    }

    delete cap;
}
///////////////////////////////////////////////////////////������////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////opencvdft////////////////////////////////////////////////////////////////

//���㺯������ʱ��
static double timeCall = 0;
static void _mul_spectrums_1_channel(const cv::Mat& m1, const cv::Mat& m2, cv::Mat& dst, bool conjB);

//http://docs.opencv.org/modules/core/doc/operations_on_arrays.html#dft[2]
void convolveDFT(Mat A, Mat B, Mat& C)
{
    double time3 = (double)getTickCount();
    // reallocate the output array if needed
    //C.create(abs(A.rows - B.rows) + 1, abs(A.cols - B.cols) + 1, A.type());
    C.create(A.rows, A.cols, CV_8UC1);
    Size dftSize;
    // calculate the size of DFT transform
    //dftSize.width = getOptimalDFTSize(A.cols + B.cols - 1);
    //dftSize.height = getOptimalDFTSize(A.rows + B.rows - 1);
    //dftSize.width = 1920;
    //dftSize.height = 1080;
    dftSize.width = getOptimalDFTSize(A.cols + B.cols - 1);
    dftSize.height = getOptimalDFTSize(A.rows + B.rows - 1);

    // allocate temporary buffers and initialize them with 0's
    Mat tempA(dftSize, A.type(), Scalar::all(0));//initial 0
    Mat tempB(dftSize, B.type(), Scalar::all(0));

    // copy A and B to the top-left corners of tempA and tempB, respectively
    Mat roiA(tempA, Rect(0, 0, A.cols, A.rows)); //����ָ����Rect�������ڴ棬���������ݿ���
    A.copyTo(roiA);

    Mat roiB(tempB, Rect(0, 0, B.cols, B.rows));
    B.copyTo(roiB);

    time3 = (double)(getTickCount() - time3) * 1000 / getTickFrequency();
    cout << "  convdft time pre =  " << time3 << "  ms" << endl;

    time3 = (double)getTickCount();

    // now transform the padded A & B in-place;
    // use "nonzeroRows" hint for faster processing
    //dft(tempA, tempA, 0, A.rows);
    //dft(tempB, tempB, 0, B.rows);
    dft(tempA, tempA, 0);  //nonzerorows������ɶ�أ��ƺ�ûɶӰ�죬�����ٶȸ�����
    dft(tempB, tempB, 0);
    
    Mat tempACopy;
    tempA.copyTo(tempACopy);
    // multiply the spectrums;
    // the function handles packed spectrum representations well
    mulSpectrums(tempA, tempB, tempA, DFT_COMPLEX_OUTPUT,true);  //�˲�Ӧ�����������
    //mulSpectrums(tempA, tempB, tempA, DFT_REAL_OUTPUT);
    Mat tempOut;
    _mul_spectrums_1_channel(tempACopy, tempB, tempOut, true);

    // transform the product back from the frequency domain.
    // Even though all the result rows will be non-zero,
    // you need only the first C.rows of them, and thus you
    // pass nonzeroRows == C.rows
    //dft(tempA, tempA, DFT_INVERSE + DFT_SCALE, C.rows);
    dft(tempA, tempA, DFT_INVERSE + DFT_SCALE);

    time3 = (double)(getTickCount() - time3)*1000 / getTickFrequency();
    cout << "  convdft timedft  =  " << time3 << "  ms" << endl;
    // now copy the result back to C.
    //tempA(Rect(0, 0, C.cols, C.rows)).copyTo(C);
    time3 = (double)getTickCount();

    tempA(Rect(0, 0, 1913 , 1073)).copyTo(C(Rect(7,7,1913,1073)));
    tempA(Rect(1937, 1118, 7, 7)).copyTo(C(Rect(0, 0, 7, 7)));
    tempA(Rect(1937, 0, 7,1073)).copyTo(C(Rect(0, 7, 7, 1073)));
    tempA(Rect(0, 1118, 1073, 7)).copyTo(C(Rect(7, 0, 1073,  7)));

    time3 = (double)(getTickCount() - time3) * 1000 / getTickFrequency();
    cout << "  convdft time copy to C  =  " << time3 << "  ms" << endl;

    //roiA(Rect(0, 0, C.cols, C.rows)).copyTo(C);  //ͼ����任��ȫ

    timeCall = (double)(getTickCount() - timeCall) * 1000 / getTickFrequency();
    cout << "  convdft timeCall  =  " << timeCall << "  ms" << endl;
    // all the temporary buffers will be deallocated automatically
}

//https://blog.csdn.net/KYJL888/article/details/79018810?utm_medium=distribute.pc_relevant.none-task-blog-title-3&spm=1001.2101.3001.4242
int mainopencvdft1(int argc, char* argv[])
{
    //const char* filename = argc >= 2 ? argv[1] : "Lenna.png";

    string filename = "C:\\Users\\b515\\Downloads\\image2.jpg";
    Mat I = imread(filename, IMREAD_GRAYSCALE);
    if (I.empty())
        return -1;

    Mat kernel = (Mat_<float>(3, 3) << -1, -1, -1, -1, 8, -1, -1, -1, -1);
    cout << kernel;

    Mat floatI = Mat_<float>(I);// change image type into float
    Mat filteredI;
    convolveDFT(floatI, kernel, filteredI);

    normalize(filteredI, filteredI, 0, 1, NORM_MINMAX); // Transform the matrix with float values into a
                                            // viewable image form (float between values 0 and 1).
    imshow("image", I);
    imshow("filtered", filteredI);
    waitKey(0);

}

#define CVTCOLOR (true)

void testopencvdft1()
{
    /////////////////////////////////////////////��ȡtxt////////////////////////////////////////////
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
    for (int i = 0; i < 15; ++i) //������ֻ�ܵõ����һ�ε����ݽ����֮ǰ�Ķ���������
    {
        kernelMat.push_back(Mat(FILTER_WIDTH, FILTER_HEIGHT, CV_32FC1, dataAllFile[i].data()));
        //kernelMat[i].convertTo(kernelMat[i], CV_16FC1);
    }
    /////////////////////////////////////////////��ȡtxt//////////////////////////////////////////// 


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
        //SFfp16out = Mat(SFfp16.size(), CV_16FC1); //�ѿ����ڴ棬���������������-23.0
        singleFrame.convertTo(SFfp32, CV_32FC1);
        SFfp32out = Mat(SFfp32.size(), CV_32FC1); //�ѿ����ڴ棬���������������-23.0

        time2 = (double)(getTickCount() - time2) * 1000 / getTickFrequency();
        cout << "   time creat SFfp32out convertTo SFfp32Frame   =  " << time2 << "  ms" << endl;
        
        time2 = (double)getTickCount();
       /* laplacianFilter_GPU_wrapper(SFfp16, SFfp16out, selectedKernelMat);*/
        //convfp16((__half*)SFfp16.data, (__half*)SFfp16out.data, (__half*)selectedKernelMat.data, SFfp16.cols, SFfp16.rows); //.ptr() .data���صĶ���һ��ָ�루��ָ�룩
        
        timeCall = (double)getTickCount();
        convolveDFT(SFfp32, selectedKernelMat, SFfp32out);

        time2 = (double)(getTickCount() - time2) * 1000 / getTickFrequency();
        cout << "   time convDFT   =  " << time2 << "  ms"<<endl;

        time2 = (double)getTickCount();
        SFfp32out.convertTo(SFfp32out, CV_8UC1);
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
        time = (double)(getTickCount() - time) *1000 / getTickFrequency();
        cout << "  time total =  " << time << "  ms" << endl;
        cout << "fps" << fps << endl;
        waitKey(1);
    }

    delete cap;
    /////////////////////////////////////////////DDEfilter////////////////////////////////////////////
}
///////////////////////////////////////////////////////////opencvdft/////////////////////////////////////////////////////////



////////////////////////////////////////////////////opencvdft cuda gpuMat////////////////////////////////////////////////////

#include <opencv2/cudaarithm.hpp>

//http://docs.opencv.org/modules/core/doc/operations_on_arrays.html#dft[2]

void convolveDFTgpu(Mat A, Mat B, Mat& C)
{
    // reallocate the output array if needed
    C.create(abs(A.rows - B.rows) + 1, abs(A.cols - B.cols) + 1, A.type());
    //C.create(A.rows, A.cols, CV_8UC1);
    Size dftSize;
    // calculate the size of DFT transform
    //dftSize.width = getOptimalDFTSize(A.cols + B.cols - 1);
    //dftSize.height = getOptimalDFTSize(A.rows + B.rows - 1);
    //dftSize.width = 1920;
    //dftSize.height = 1080;
    dftSize.width = getOptimalDFTSize(A.cols + B.cols - 1);
    dftSize.height = getOptimalDFTSize(A.rows + B.rows - 1);

    // allocate temporary buffers and initialize them with 0's
    Mat tempA(dftSize, A.type(), Scalar::all(0));//initial 0
    Mat tempB(dftSize, B.type(), Scalar::all(0));

    // copy A and B to the top-left corners of tempA and tempB, respectively
    Mat roiA(tempA, Rect(0, 0, A.cols, A.rows)); //����ָ����Rect�������ڴ棬���������ݿ���
    A.copyTo(roiA);

    Mat roiB(tempB, Rect(0, 0, B.cols, B.rows));
    B.copyTo(roiB);

    // now transform the padded A & B in-place;
    // use "nonzeroRows" hint for faster processing
    //dft(tempA, tempA, 0, A.rows);
    //dft(tempB, tempB, 0, B.rows);
    dft(tempA, tempA, 0);  //nonzerorows������ɶ�أ��ƺ�ûɶӰ�죬�����ٶȸ�����
    dft(tempB, tempB, 0);

    // multiply the spectrums;
    // the function handles packed spectrum representations well
    mulSpectrums(tempA, tempB, tempA, DFT_COMPLEX_OUTPUT, true);  //�˲�Ӧ�����������
    //mulSpectrums(tempA, tempB, tempA, DFT_REAL_OUTPUT);

    // transform the product back from the frequency domain.
    // Even though all the result rows will be non-zero,
    // you need only the first C.rows of them, and thus you
    // pass nonzeroRows == C.rows
    //dft(tempA, tempA, DFT_INVERSE + DFT_SCALE, C.rows);
    dft(tempA, tempA, DFT_INVERSE + DFT_SCALE);

    // now copy the result back to C.
    //tempA(Rect(0, 0, C.cols, C.rows)).copyTo(C);
    tempA(Rect(0, 0, 1913, 1073)).copyTo(C(Rect(7, 7, 1913, 1073)));
    tempA(Rect(1937, 1118, 7, 7)).copyTo(C(Rect(0, 0, 7, 7)));
    tempA(Rect(1937, 0, 7, 1073)).copyTo(C(Rect(0, 7, 7, 1073)));
    tempA(Rect(0, 1118, 1073, 7)).copyTo(C(Rect(7, 0, 1073, 7)));

    // all the temporary buffers will be deallocated automatically

}

//////////////////////////////////////////////////////opencvdft cuda gpuMat//////////////////////////////////////////////////

//////////////////////////////////////////////////////mulSpectrum cpu////////////////////////////////////////////////////////

static void _mul_spectrums_1_channeldouble(const cv::Mat& m1, const cv::Mat& m2, cv::Mat& dst, bool conjB)
{
    CV_Assert(m1.channels() == 1); //�ж�����ͼ���Ƿ�Ϊһͨ����
    dst.create(m1.size(), m1.type());

    dst.ptr<double>(0)[0] = m1.ptr<double>(0)[0] * m2.ptr<double>(0)[0];
    //Ȼ�������ͼ��ĵڶ��п�ʼ���Ե�һ�н��м��㣬�ѵ�һ�е���������ֵ�ֱ�
    //����һ��������ʵ�����鲿��Ȼ������ˣ������ʵ����Ϊ��һ�е�ֵ���鲿��Ϊ��һ�е�ֵ
    for (int i = 1; i <= m1.rows - 2; i += 2)
    {
        double re1 = m1.ptr<double>(i)[0];
        double im1 = m1.ptr<double>(i + 1)[0];
        double re2 = m2.ptr<double>(i)[0];
        double im2 = m2.ptr<double>(i + 1)[0];
        if (conjB)
        {
            im2 = -im2;
        }
        dst.ptr<double>(i)[0] = re1 * re2 - im1 * im2;
        dst.ptr<double>(i + 1)[0] = re1* im2 + re2 * im1;
    }
    //�������ͼ�������Ϊ˫�����������һ�е�һ�л�û�д���
    if ((m1.rows & 1) == 0)
    {
        dst.ptr<double>(m1.rows - 1)[0] = m1.ptr<double>(m1.rows - 1)[0] * m2.ptr<double>(m1.rows - 1)[0];
    }
    //��������ͼ���������˫���������һ�к͵�һ�еĴ���ʽ��һ���ģ�Ȼ��ֻʣ���м���л�û�д���
    if ((m1.cols & 1) == 0)
    {
        dst.ptr<double>(0)[m1.cols-1] = m1.ptr<double>(0)[m1.cols - 1] * m2.ptr<double>(0)[m1.cols - 1];
        //Ȼ�������ͼ��ĵڶ��п�ʼ�������һ�н��м��㣬�����һ�е���������ֵ�ֱ�
        //����һ��������ʵ�����鲿��Ȼ������ˣ������ʵ����Ϊ��һ�е�ֵ���鲿��Ϊ��һ�е�ֵ
        for (int i = 1; i <= m1.rows - 2; i += 2)
        {
            double re1 = m1.ptr<double>(i)[m1.cols - 1];
            double im1 = m1.ptr<double>(i + 1)[m1.cols - 1];
            double re2 = m2.ptr<double>(i)[m1.cols - 1];
            double im2 = m2.ptr<double>(i + 1)[m1.cols - 1];
            if (conjB)
            {
                im2 = -im2;
            }
            dst.ptr<double>(i)[m1.cols - 1] = re1 * re2 - im1 * im2;
            dst.ptr<double>(i + 1)[m1.cols - 1] = re1 * im2 + re2 * im1;
        }
        //�������ͼ�������Ϊ˫�����������һ�����һ�л�û�д���
        if ((m1.rows & 1) == 0)
        {
            dst.ptr<double>(m1.rows - 1)[m1.cols - 1] = m1.ptr<double>(m1.rows - 1)[m1.cols - 1] * m2.ptr<double>(m1.rows - 1)[m1.cols - 1];
        }
    }

    //�����м����
    int j0 = 1;
    int j1 = m1.cols - ((m1.cols & 1) == 0 ? 1 : 0);
    //����ʽ����ͷ�к�β�еķ�ʽ��һ������ȡÿ�е�ǰ��������ƴ�����������
    for (int i = 0; i < m1.rows; ++i)
    {
        for (int j = j0; j < j1; j += 2)
        {
            double re1 = m1.ptr<double>(i)[j];
            double im1 = m1.ptr<double>(i)[j+1];
            double re2 = m2.ptr<double>(i)[j];
            double im2 = m2.ptr<double>(i)[j+1];
            if (conjB)
            {
                im2 = -im2;
            }
            dst.ptr<double>(i)[j] = re1 * re2 - im1 * im2;
            dst.ptr<double>(i)[j+1] = re1 * im2 + re2 * im1;
        }
    }
}


static void _mul_spectrums_1_channel(const cv::Mat& m1, const cv::Mat& m2, cv::Mat& dst, bool conjB)
{
    CV_Assert(m1.channels() == 1); //�ж�����ͼ���Ƿ�Ϊһͨ����
    dst.create(m1.size(), m1.type());

    dst.ptr<float>(0)[0] = m1.ptr<float>(0)[0] * m2.ptr<float>(0)[0];
    //Ȼ�������ͼ��ĵڶ��п�ʼ���Ե�һ�н��м��㣬�ѵ�һ�е���������ֵ�ֱ�
    //����һ��������ʵ�����鲿��Ȼ������ˣ������ʵ����Ϊ��һ�е�ֵ���鲿��Ϊ��һ�е�ֵ
    for (int i = 1; i <= m1.rows - 2; i += 2)
    {
        float re1 = m1.ptr<float>(i)[0];
        float im1 = m1.ptr<float>(i + 1)[0];
        float re2 = m2.ptr<float>(i)[0];
        float im2 = m2.ptr<float>(i + 1)[0];
        if (conjB)
        {
            im2 = -im2;
        }
        dst.ptr<float>(i)[0] = re1 * re2 - im1 * im2;
        dst.ptr<float>(i + 1)[0] = re1 * im2 + re2 * im1;
    }
    //�������ͼ�������Ϊ˫�����������һ�е�һ�л�û�д���
    if ((m1.rows & 1) == 0)
    {
        dst.ptr<float>(m1.rows - 1)[0] = m1.ptr<float>(m1.rows - 1)[0] * m2.ptr<float>(m1.rows - 1)[0];
    }
    //��������ͼ���������˫���������һ�к͵�һ�еĴ���ʽ��һ���ģ�Ȼ��ֻʣ���м���л�û�д���
    if ((m1.cols & 1) == 0)
    {
        dst.ptr<float>(0)[m1.cols - 1] = m1.ptr<float>(0)[m1.cols - 1] * m2.ptr<float>(0)[m1.cols - 1];
        //Ȼ�������ͼ��ĵڶ��п�ʼ�������һ�н��м��㣬�����һ�е���������ֵ�ֱ�
        //����һ��������ʵ�����鲿��Ȼ������ˣ������ʵ����Ϊ��һ�е�ֵ���鲿��Ϊ��һ�е�ֵ
        for (int i = 1; i <= m1.rows - 2; i += 2)
        {
            float re1 = m1.ptr<float>(i)[m1.cols - 1];
            float im1 = m1.ptr<float>(i + 1)[m1.cols - 1];
            float re2 = m2.ptr<float>(i)[m1.cols - 1];
            float im2 = m2.ptr<float>(i + 1)[m1.cols - 1];
            if (conjB)
            {
                im2 = -im2;
            }
            dst.ptr<float>(i)[m1.cols - 1] = re1 * re2 - im1 * im2;
            dst.ptr<float>(i + 1)[m1.cols - 1] = re1 * im2 + re2 * im1;
        }
        //�������ͼ�������Ϊ˫�����������һ�����һ�л�û�д���
        if ((m1.rows & 1) == 0)
        {
            dst.ptr<float>(m1.rows - 1)[m1.cols - 1] = m1.ptr<float>(m1.rows - 1)[m1.cols - 1] * m2.ptr<float>(m1.rows - 1)[m1.cols - 1];
        }
    }

    //�����м����
    int j0 = 1;
    int j1 = m1.cols - ((m1.cols & 1) == 0 ? 1 : 0);
    //����ʽ����ͷ�к�β�еķ�ʽ��һ������ȡÿ�е�ǰ��������ƴ�����������
    for (int i = 0; i < m1.rows; ++i)
    {
        for (int j = j0; j < j1; j += 2)
        {
            float re1 = m1.ptr<float>(i)[j];
            float im1 = m1.ptr<float>(i)[j + 1];
            float re2 = m2.ptr<float>(i)[j];
            float im2 = m2.ptr<float>(i)[j + 1];
            if (conjB)
            {
                im2 = -im2;
            }
            dst.ptr<float>(i)[j] = re1 * re2 - im1 * im2;
            dst.ptr<float>(i)[j + 1] = re1 * im2 + re2 * im1;
        }
    }
}

//////////////////////////////////////////////////////mulSpectrum cpu////////////////////////////////////////////////////////

int main()
{
    //test();
    //main_(0,0);
    //mainopencvdft1(0, 0);
    //testcudamulSpectrum();
    cudaFFTmulSpectrum1119float();
    return 0;
}



