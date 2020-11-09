
#include "main.h"

extern "C" bool laplacianFilter_GPU_wrapper(const cv::Mat & input, cv::Mat & output, const cv::Mat & kernel);

void ycbcrUpdate(const Mat& IM_result_cbcr, const Mat& IM_bri_T, Mat& IM_result_cbcr_re);

vector<vector<float>> extractConvMat0();

int maindde()
{
    VideoCapture* cap = new VideoCapture("C:\\Users\\b515\\Desktop\\pmj\\cuda_dde\\fenglin\\130 (2).avi");
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
    vector<vector<float>> temp = extractConvMat0();
    vector<Mat> hsfatMat;
    for (int i = 0; i < 15; i++)
    {
        hsfatMat.push_back(Mat(15, 15, CV_32FC1, temp[i].data()));
    }

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
        Mat H_S_f_A_T = hsfatMat[(unsigned __int64)(thresholdRate * 10)];

        IM_bri_T0 = singleChannel;

        Mat IM_bri_T = Mat::zeros(IM_bri_T0.size(), IM_bri_T0.type());

        laplacianFilter_GPU_wrapper(IM_bri_T0, IM_bri_T, H_S_f_A_T);

        //                filter2D(IM_bri_T0,IM_bri_T,CV_64FC1,H_S_f_A_T);
        Mat meanOfIM_bri_T, stdDevOfIM_bri_T;
        meanStdDev(IM_bri_T, meanOfIM_bri_T, stdDevOfIM_bri_T);
        IM_bri_T = IM_bri_T / meanOfIM_bri_T.at<double>(0, 0) * meanBri.at<double>(0, 0) / 1;

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
    }
    return HVSFT;
}
