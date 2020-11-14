#include "main.h"
#include "cuda_fp16.h "

extern "C" void laplacianFilter_GPU_wrapper(const cv::Mat & input, cv::Mat & output, const cv::Mat & kernel);
vector<vector<float>> extractConvMat0();
void ycbcrUpdate(const Mat& IM_result_cbcr, const Mat& IM_bri_T, Mat& IM_result_cbcr_re);

vector<Mat> extractConvMat1();


// 查看Mat数据  https://www.cnblogs.com/nipan/p/4098373.html
//vector是在堆内开辟空间，不是在栈内，所以可以做返回值，不需要引用


int main(int argc,char** argv)
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
    //
    vector<vector<float>> temp = extractConvMat0();
    vector<Mat> hsfatMat;
    for (int i = 0; i < 15; i++)
    {  
        hsfatMat.push_back(Mat(FILTER_HEIGHT, FILTER_WIDTH, CV_32FC1, temp[i].data()));
        //printf(" 地址前  %p， %p\n", &hsfatMat[i], hsfatMat[i].data); //convertTo之后地址矩阵头地址不变，矩阵数据地址改变要重新分配空间
        printf("sizeof(Mat) %d\n", (int)sizeof(hsfatMat[i]));  //96
        //printf("temp[%d].data()  %p \n", i , temp[i].data());  //vector内存是连续在堆上的，不分作用域的
        hsfatMat[i].convertTo(hsfatMat[i], CV_16FC1);
        //printf(" 地址后  %p， %p\n", &hsfatMat[i], hsfatMat[i].data);
    }

    cout << hsfatMat[0] << endl;

    cout << "      ----------------------------------      " << endl;

    vector<Mat> ha;
    ha = extractConvMat1();  //return vector<Mat>; 只是返回了矩阵头，数据矩阵的内存，在函数结束后就释放了
    //cout << ha[0] << endl;  //已经被释放的内存，打印的都是相同的随机数
    printf("外面  ha[%d].data  %p   , &hvsft = %p\n", 0, ha[0].data, &ha[0]); 
    
    while (cap->read(frame))  //type CV_8UC3
    {
        double time = (double)getTickCount();
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
        Mat H_S_f_A_T = hsfatMat[(unsigned __int64)(thresholdRate * 10)];

        IM_bri_T0 = singleChannel;

       /* Mat IM_bri_T = Mat::zeros(IM_bri_T0.size(), IM_bri_T0.type());*/

        double time2 = (double)getTickCount();
        Mat IM_bri_T00;
        IM_bri_T0.convertTo(IM_bri_T00, CV_16FC1);
        Mat IM_bri_T = Mat::zeros(IM_bri_T0.size(), CV_16FC1);
        laplacianFilter_GPU_wrapper(IM_bri_T00, IM_bri_T, H_S_f_A_T);

        //filter2D(IM_bri_T0,IM_bri_T,CV_64FC1,H_S_f_A_T);
        time2 = (getTickCount() - time2)*1000 / getTickFrequency();
        cout << "time2 = " << time2 << endl;


        //要调节亮度，尽量与原像素一样
        //Mat meanOfIM_bri_T, stdDevOfIM_bri_T;
        //mean meanStdDev函数好像不支持16fp
        //meanStdDev(IM_bri_T, meanOfIM_bri_T, stdDevOfIM_bri_T);
        //IM_bri_T = IM_bri_T / meanOfIM_bri_T.at<double>(0, 0) * meanBri.at<double>(0, 0) / 1;
        Mat IM_bri_T_U8;
        IM_bri_T.convertTo(IM_bri_T_U8, CV_8UC1);
        //printf("CV_8UC1 == CV_8U  %s \n", CV_8UC1 == CV_8U ? "true" : "false");
        Scalar meanOfIM_bri_T = mean(IM_bri_T_U8); //Scalar继承自容器
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

vector<Mat> extractConvMat1()
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
    vector<Mat> hvsft;
    for (int i = 0; i < 15; ++i)
    {
        hvsft.push_back(Mat(FILTER_WIDTH, FILTER_HEIGHT, CV_32FC1, HVSFT[i].data()));
        printf(" extraMat1 hvsft[%d].data  %p   , &hvsft = %p\n", i, hvsft[i].data, &hvsft[i]);
        //cout << hvsft[i] << endl;
    }
    return hvsft;
}
