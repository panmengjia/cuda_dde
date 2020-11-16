#include <opencv2/opencv.hpp>
#include <iostream>
//#include <unistd.h>
#include <fstream>
#include "cuda_fp16.h"

using namespace cv;
using namespace std;

extern "C" bool laplacianFilter_GPU_wrapper(const cv::Mat& input, cv::Mat& output, const cv::Mat& kernel);

void ycbcrUpdate(const Mat& IM_result_cbcr,const Mat& IM_bri_T ,Mat& IM_result_cbcr_re);

vector<vector<float>> extractConvMat0();

int main_()
{
    VideoCapture* cap = new VideoCapture("/home/nvidia/Desktop/dde1448/130(3).avi");
    if(!cap->isOpened())
    {
        cout <<"video is empty!!"<<endl;
    }
    Mat frame; //1080 1920
    //discard the first 25 frames
    while(cap->read(frame))
    {
        static unsigned int counter=0;
        if(++counter == 25)
        {
            break;
        }
    }
    Mat meanFrame,stdDevFrame,singleChannel/*IM_bri*/;
//    double mean;
    vector<Mat> splitFrame2bgr3;

    Mat meanBri,stdDevBri;
    double thresholdRate;

    vector<vector<float>> temp = extractConvMat0();
    vector<Mat> hsfatMat;
    for(int i = 0; i < 15; i++)
    {
        hsfatMat.push_back(Mat(15,15,CV_32FC1,temp[i].data()));
    }
    Mat hsfatMat6;
    hsfatMat[6].convertTo(hsfatMat6,CV_16FC1);

    double time1 =0;

    Mat IM_bri_T016, IM_bri_T;
    Mat IM_bri_T_U8;
    Mat IM_result_cbcr;
    Mat IM_result_cbcr_re;

    while(cap->read(frame))  //type CV_8UC3
    {
        double time = getTickCount();
        split(frame,splitFrame2bgr3);
        singleChannel =  0.257*splitFrame2bgr3[2] + 0.564*splitFrame2bgr3[1] + 0.098*splitFrame2bgr3[0] +0.0;
        meanStdDev(singleChannel,meanBri,stdDevBri);
        thresholdRate = meanBri.at<double>(0,0)/stdDevBri.at<double>(0,0)*meanBri.at<double>(0,0)/80 + 0.2;

        if(thresholdRate > 0.6)
        {
            thresholdRate = 0.6;
        }
        cout<<"thresholdRate*10:   "<<thresholdRate*10<<endl;
        Mat H_S_f_A_T = hsfatMat[thresholdRate*10];

        singleChannel.convertTo(IM_bri_T016, CV_16FC1);
        IM_bri_T = Mat::zeros(IM_bri_T016.size(), CV_16FC1);

        time1 = (double)getTickCount();
        laplacianFilter_GPU_wrapper(IM_bri_T016, IM_bri_T, hsfatMat6);
        time1 = (double)(getTickCount() - time1)*1000/getTickFrequency();
        cout <<"  time1 lp gpu =  "<<time1<<"  ms"<<endl;

//        filter2D(IM_bri_T0,IM_bri_T,CV_64FC1,H_S_f_A_T);
//        Mat meanOfIM_bri_T,stdDevOfIM_bri_T;
//        meanStdDev(IM_bri_T,meanOfIM_bri_T,stdDevOfIM_bri_T);
//        IM_bri_T = IM_bri_T/meanOfIM_bri_T.at<double>(0,0)*meanBri.at<double>(0,0)/1;


        IM_bri_T.convertTo(IM_bri_T_U8,CV_8UC1);
        Scalar mean1 = mean(IM_bri_T_U8);
        IM_bri_T = IM_bri_T/mean1[0]*meanBri.at<double>(0,0)/1;


        cvtColor(frame,IM_result_cbcr,COLOR_BGR2YCrCb);

        ycbcrUpdate(IM_result_cbcr, IM_bri_T, IM_result_cbcr_re);

        time1 = (double)getTickCount();
        imshow("frame",frame);
        imshow("IM_result_cbcr_re",IM_result_cbcr_re);
        time1 = (double)(getTickCount() - time1)*1000/getTickFrequency();
        cout <<"  time1 imshow  =  "<<time1<<"  ms"<<endl;

        double fps = getTickFrequency()/(getTickCount() - time);
        cout <<"fps"<<fps<<endl;
        cout <<"  time total  =  "<<1000/fps<<"  ms"<<endl;

        waitKey(1);

    }

    return 0;
}

void ycbcrUpdate(const Mat& IM_result_cbcr,const Mat& IM_bri_T ,Mat& IM_result_cbcr_re)
{
    vector<Mat> channelsOfIM;
    split(IM_result_cbcr,channelsOfIM);
    Mat IM_bri_T_8U;
    IM_bri_T.convertTo(IM_bri_T_8U,CV_8UC1);
    channelsOfIM[0] = IM_bri_T_8U;
    merge(channelsOfIM,IM_result_cbcr_re);
    cvtColor(IM_result_cbcr_re,IM_result_cbcr_re,COLOR_YCrCb2BGR);
}

vector<vector<float>> extractConvMat0()
{
    const string& str = "/home/nvidia/Desktop/dde1448/15/";
    vector<vector<float>> HVSFT;
    HVSFT.resize(15);
    //    unsigned int counter = 0;
    for(int i = 1;i< 16;i++)
    {
        ifstream dataFile(str+to_string(i)+".txt");
        float dataElement;
        while(dataFile >> dataElement)
        {
            HVSFT[i-1].push_back(dataElement);
        }
        dataFile.close();
    }
    return HVSFT;
}


////////////////////////////////////////////////gpu conv after vs debug 副本////////////////////////////////////////////////////////////////

#define VIDEO_DIR ("/home/nvidia/Desktop/dde1448/130(3).avi")
#define TXT_DIR ("/home/nvidia/Desktop/dde1448/15/")

#define CVTCOLOR (true)
extern "C" void convfp16(const __half * indata, __half * outdata, __half * kerneldata, const int width, const int height);


#define BLOCK_SIZE      (8)
#define FILTER_WIDTH    (15)
#define FILTER_HEIGHT   (15)  //85 vs 2.2seconds  核越大黑色，边框越大

#define IMG_HEIGHT  (1080)
#define IMG_WIDTH   (1920)

void test()
{
    /////////////////////////////////////////////读取txt////////////////////////////////////////////
    const string str = TXT_DIR;  //""里面不能多一个字符
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
        //dataPerFile.swap(vector<float>());  //swap函数形参为引用，实参必须是左值 void swap(vector& _Right) 非常量引用的初始值必须为左值
        //vector<float>().swap(dataPerFile);
        //dataPerFile内部定义就不需要，删除vector元素
        //在外面定义，则需要删除，否则存储的只是第一个文件数据
        dataPerFile.clear();
        //clear居然也可以达到同样的正确效果，应该跟push_back有关

        //kernelMat.push_back(Mat(FILTER_WIDTH, FILTER_HEIGHT, CV_32FC1, dataPerFile.data()));
        //cout << "========================================================" << endl;
        //cout << "           i          " << i << endl;
        //cout << kernelMat[i] << endl;
        //vector<float>().swap(dataPerFile); //swap就会强制释放内存，强制交换内存
        ////dataPerFile.clear(); //clear(),不会去清理数据内存，可能只是将vector的数据结构抹掉
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
    for (int i = 0; i < 15; ++i) //在外面只能得到最后一次的数据结果，之前的都被覆盖了
    {
        kernelMat.push_back(Mat(FILTER_WIDTH, FILTER_HEIGHT, CV_32FC1, dataAllFile[i].data()));
        kernelMat[i].convertTo(kernelMat[i], CV_16FC1);
        //cout << " ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
        //cout << "---------------i=====================                " << i << endl;
        //cout << kernelMat[i] << endl;

    }
    /////////////////////////////////////////////读取txt////////////////////////////////////////////


    VideoCapture* cap = new VideoCapture(VIDEO_DIR);
    if (!cap->isOpened())
    {
        cout << "video is empty!!" << endl;
    }
    Mat frame; //1080 1920
    Mat outFrame,singleFrame,meanSF,stdDevSF,selectedKernelMat,SFfp16,SFfp16out,frameYUV;
    vector<Mat> frame3channels,frame3YUV;
    double threshRate;

    double time1 = 0;
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
        selectedKernelMat = kernelMat[(unsigned int)(threshRate * 10)];

        time1 = (double)getTickCount();
        singleFrame.convertTo(SFfp16, CV_16FC1);
        SFfp16out = Mat(SFfp16.size(), CV_16FC1); //已开辟内存，但是随机数，都是-23.0
        time1 = (getTickCount() - time1) * 1000 / getTickFrequency();
        cout << "time2 convertTo SFfp16 8u->fp16 SFfp16out init  =  " << time1 << "  ms" << endl;

        time1 = (double)getTickCount();
        //cout << SFfp16out(Rect(10, 10, 10, 10)) << endl;
       /* laplacianFilter_GPU_wrapper(SFfp16, SFfp16out, selectedKernelMat);*/
        convfp16((__half*)SFfp16.data, (__half*)SFfp16out.data, (__half*)selectedKernelMat.data, SFfp16.cols, SFfp16.rows); //.ptr() .data返回的都是一级指针（列指针）
        //cout << SFfp16out(Rect(10, 10, 10, 10)) << endl;
        //filter2D(IM_bri_T0,IM_bri_T,CV_64FC1,H_S_f_A_T);
        time1 = (getTickCount() - time1) * 1000 / getTickFrequency();
        cout << "time1 = " << time1 << endl;

        SFfp16out.convertTo(SFfp16out, CV_8UC1);
        Scalar meanSFfp16out = mean(SFfp16out);
        SFfp16out = SFfp16out / meanSFfp16out[0] * meanSF.at<double>(0, 0) / 1;

        cvtColor(frame, frameYUV, COLOR_BGR2YCrCb);
        split(frameYUV, frame3YUV);
        frame3YUV[0] = SFfp16out;
        merge(frame3YUV, frameYUV);
        cvtColor(frameYUV, outFrame, COLOR_YCrCb2BGR);

        time1 = (double)getTickCount();
        imshow("frame", frame);
        imshow("outFrame", outFrame);
        time1 = (double)(getTickCount() - time1)/getTickFrequency();
        cout <<"  time1 imshow"<<time1<<"  ms"<<endl;


        double fps = getTickFrequency() / (getTickCount() - time);
        cout << "fps" << fps << endl;
        cout <<"  time total  =  "<<1000/fps<<"  ms"<<endl;
        waitKey(1);
    }

    delete cap;
}

////////////////////////////////////////////////gpu conv after vs debug副本////////////////////////////////////////////////////////////////


int main()
{

    test();
    return 0;
}
