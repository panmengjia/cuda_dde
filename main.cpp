#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <iostream>
#include <opencv2/gapi.hpp>
#include <opencv2/core/opengl.hpp>

using namespace cv;
using namespace std;



int cudatest()
{
//    VideoCapture cap("/home/nvidia/Desktop/dde1448/130(3).avi");
//    if(!cap.isOpened())
//    {
//        cout <<"--------------------video file is not open-------------------------"<<endl;
//    }
//    Mat frame;
//    cuda::GpuMat dev_frame;
//    while(cap.read(frame))
//    {

//        imshow("frame",frame);
//        waitKey(30);

//    }


    if(cuda::getCudaEnabledDeviceCount()==0){
        cerr<<"此OpenCV编译的时候没有启用CUDA模块"<<endl;
        return -1;
    }

    const int rows = 16*50;
    const int cols = 16*60;
    const int type = CV_8UC3;

    // 初始化一个黑色的GpuMat
    cuda::GpuMat gpuMat(rows,cols,type,Scalar(0,0,0));
    // 定义一个空Mat
    Mat dst;
    // 把gpuMat中数据下载到dst(从显存下载到内存)
    gpuMat.download(dst);
    // 显示
    imshow("show",dst);
    waitKey(0);

    // 读取一张图片
    Mat arr = imread("/home/nvidia/Downloads/6f8473fc82bf264517fcb55e9e46ccd4.jpg");
    imshow("show",arr);
    waitKey(0);

    // 上传到gpuMat(若gpuMat不为空，会先释放原来的数据，再把新的数据上传上去)
    gpuMat.upload(arr);
    // 定义另外一个空的GpuMat
    cuda::GpuMat gray;
    // 把gpuMat转换为灰度图gray
    cuda::cvtColor(gpuMat,gray,COLOR_BGR2GRAY);
    // 下载到dst，如果dst不为空，旧数据会被覆盖
    gray.download(dst);
    // 显示
    imshow("show",dst);
    waitKey(0);
    return 0;
}


int main()
{
    Mat dst;
    cv::ogl::Texture2D tex;
    cuda::GpuMat gpuMat;
    Mat arr = imread("/home/nvidia/Downloads/6f8473fc82bf264517fcb55e9e46ccd4.jpg");
    imshow("show",arr);
    waitKey(0);


    // 上传到gpuMat(若gpuMat不为空，会先释放原来的数据，再把新的数据上传上去)
    gpuMat.upload(arr);


    cv::namedWindow("gray",WINDOW_OPENGL);
    setOpenGlContext("gray");
    tex.bind();
    tex.copyFrom(gpuMat,true);

    // 定义另外一个空的GpuMat
    cuda::GpuMat gray;
    // 把gpuMat转换为灰度图gray
    cuda::cvtColor(gpuMat,gray,COLOR_BGR2GRAY);
    // 下载到dst，如果dst不为空，旧数据会被覆盖
    gray.download(dst);
    // 显示
    imshow("gray",dst);
    waitKey(0);

    return 0;
}
