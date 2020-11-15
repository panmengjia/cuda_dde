#include "main.h"
#include "cuda_fp16.h "

extern "C" void laplacianFilter_GPU_wrapper(const cv::Mat & input, cv::Mat & output, const cv::Mat & kernel);
vector<vector<float>> extractConvMat0();
void ycbcrUpdate(const Mat& IM_result_cbcr, const Mat& IM_bri_T, Mat& IM_result_cbcr_re);

vector<Mat> extractConvMat1();


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
    const string str = "C:/Users/b515/Desktop/pmj/cuda_dde/fenglin/15/";
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

int main()
{
    test();
    //main_(0,0);
    return 0;
}