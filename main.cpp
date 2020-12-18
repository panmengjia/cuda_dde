#include <iostream>
#include <string>

#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/cudabgsegm.hpp"
#include "opencv2/video.hpp"
#include "opencv2/highgui.hpp"

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudaimgproc.hpp"
#include <opencv2/cudaarithm.hpp>

using namespace std;
using namespace cv;
using namespace cv::cuda;


#define VIDEO_DIR ("/home/nvidia/Desktop/dde1448/130(3).avi")
#define TXT_DIR ("/home/nvidia/Desktop/dde1448/15/")


#include "cuda.h"
#include "cuda_runtime.h"
#include "cufft.h"
#include<fstream>

#define BLOCK_SIZE      (32)
#define FILTER_WIDTH    (15)
#define FILTER_HEIGHT   (15)  //85 vs 2.2seconds  核越大黑色，边框越大

#define IMG_HEIGHT  (1080)
#define IMG_WIDTH   (1920)

#define ToString(x) #x

template <class _Tp>
void printf_matrix(const _Tp& d_frame,string str)
{
    cout <<str<<".size()      "<<d_frame.size()<<endl;
    cout <<str<<".depth()     "<<d_frame.depth()<<endl;
    cout <<str<<".channels()  "<<d_frame.channels()<<endl;
//        cout <<"src.total()   "<<src.total()<<endl;
    cout <<str<<".step        "<<d_frame.step<<endl;
    cout <<str<<".rows        "<<d_frame.rows<<endl;
    cout <<str<<".type        "<<d_frame.type()<<endl;
    cout <<"CV_8UC4             "<<CV_8UC4<<endl;
    cout <<str<<".iscontinue  "<<d_frame.isContinuous()<<endl;
    cout <<str<<".start-end   "<<d_frame.dataend -d_frame.datastart<<endl;
    cout<<endl;
    cout <<"------------------------------------------------------"<<endl;
    cout <<endl;
}

int main()
{



    VideoCapture cap;
//     cap.open("../../Downloads/opencv-4.5.0/samples/data/vtest.avi");

    cap.open("/home/nvidia/Desktop/dde1448/130(3).avi");
    if (!cap.isOpened())
    {
        cerr << "can not open camera or video file" << endl;
        return -1;
    }

    Mat frame;
    cap >> frame;

    GpuMat d_frame(frame);

    /////////////////////////////////////////////读取txt////////////////////////////////////////////
    string str = "/home/nvidia/Desktop/dde1448/15/";
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
//    cufftComplex* kernelComplex[15];
//    Mat kernelPadded;

    for (int i = 0; i < 15; ++i) 
    {
//        kernelComplex[i] = (cufftComplex*)malloc(sizeof(cufftComplex) * IMG_HEIGHT * (IMG_WIDTH / 2 + 1));
        kernelMat.push_back(Mat(FILTER_HEIGHT, FILTER_WIDTH, CV_32FC1, dataAllFile[i].data()));
//        copyMakeBorder(kernelMat[i], kernelPadded, 0, IMG_HEIGHT - kernelMat[i].rows, 0, IMG_WIDTH - kernelMat[i].cols, BORDER_CONSTANT, Scalar(0)); //Scalar::all(0)
//        fftKernel((cufftReal*)kernelPadded.data, kernelComplex[i], kernelPadded.rows, kernelPadded.cols);/*IMG_HEIGHT, IMG_WIDTH);*/
//        //kernelMat[i].convertTo(kernelMat[i], CV_16FC1);
    }
//    cout <<kernelMat[0]<<endl;

//        cout <<"src4ch.data     "<<src4ch.data<<endl;
    cout <<"src.size()      "<<kernelMat[0].size()<<endl;
    cout <<"src.depth()     "<<kernelMat[0].depth()<<endl;
    cout <<"src.channels()  "<<kernelMat[0].channels()<<endl;
//        cout <<"src.total()   "<<src.total()<<endl;
    cout <<"src.step        "<<kernelMat[0].step<<endl;
    cout <<"src.rows        "<<kernelMat[0].rows<<endl;
    cout <<"src.type        "<<kernelMat[0].type()<<endl;
    cout <<"CV_8UC4         "<<CV_8UC4<<endl;
    cout <<"src.iscontinue  "<<kernelMat[0].isContinuous()<<endl;
    cout <<"src.start-end   "<<kernelMat[0].dataend -kernelMat[0].datastart<<endl;

    uchar* pKernel = kernelMat[6].data;
    for(int r =0;r<FILTER_HEIGHT;++r)
    {
        for(int c =0 ; c<FILTER_WIDTH;++c)
        {
            cout <<*((float*)pKernel+c)<<" ";
        }
        pKernel+=kernelMat[6].step;
        cout <<endl;
    }

    GpuMat d_frame_gray;
    GpuMat d_frame_conv,d_frame_conv8U,d_frame_yuv,d_frame_out;
    cout <<"d_frame_conv.step        "<<d_frame_conv.step<<endl;
    Mat frame_gray,frame_out;
    Mat mulMat(1,1,CV_32FC1);

    Scalar d_mean_a,d_stdDev_a;
    Scalar d_mean,d_stdDev;

    Ptr<cuda::Convolution> convolver = cuda::createConvolution(Size(FILTER_HEIGHT, FILTER_WIDTH));
    Ptr<cuda::Convolution> mulGpuMat = cuda::createConvolution(Size(1, 1));

    static int frame_count=0;
    for(;;)
    {
        cap>>frame;
        if(frame.empty())
        {
            cout <<"-----------exit--------------"<<endl;
            break;
        }
        printf("-------------------%d--------------------------\n",++frame_count);
        d_frame.upload(frame);
//        imshow("frame",frame);
        printf_matrix<GpuMat>(d_frame,ToString(d_frame));

        //the kernelMat[6] must be Mat type,otherwise,illegal memory accs
        cuda::cvtColor(d_frame,d_frame_gray,cv::COLOR_RGB2GRAY);
        cuda::meanStdDev(d_frame_gray,d_mean_a,d_stdDev_a);
        printf_matrix<GpuMat>(d_frame_gray,ToString(d_frame_gray));

        d_frame_gray.convertTo(d_frame_gray,CV_32FC1);
        printf_matrix<GpuMat>(d_frame_gray,ToString(d_frame_grayf));

        convolver->convolve(d_frame_gray,kernelMat[6],d_frame_conv,true);
        printf_matrix<GpuMat>(d_frame_conv,ToString(d_frame_conv));

        d_frame_conv.convertTo(d_frame_conv8U,CV_8UC1);
        cuda::meanStdDev(d_frame_conv8U,d_mean,d_stdDev);
//        d_frame_conv = d_frame_conv/d_mean[0]*d_mean_a[0]/1;
        *(float*)mulMat.data = 1.0/d_mean[0]*d_mean_a[0]/1;
        mulGpuMat->convolve(d_frame_conv,mulMat,d_frame_conv);

        d_frame_conv.convertTo(d_frame_conv8U,CV_8UC1);
        printf_matrix<GpuMat>(d_frame_conv8U,ToString(d_frame_conv8U));

        cuda::cvtColor(d_frame,d_frame_yuv,cv::COLOR_BGR2YCrCb);
        printf_matrix<GpuMat>(d_frame_yuv,ToString(d_frame_yuv));

        vector<GpuMat> d_frame_yuvVec;
        cuda::split(d_frame_yuv,d_frame_yuvVec);
        d_frame_yuvVec[0].data = d_frame_conv8U.data;  //pass pointer of data
        cuda::merge(d_frame_yuvVec,d_frame_yuv);

        cuda::cvtColor(d_frame_yuv,d_frame_out,cv::COLOR_YCrCb2BGR);

        d_frame_out.download(frame_out);

//        imshow("d_frame_out",frame_out);
//        waitKey(1);

    }


    return 0;
}
