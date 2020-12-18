#include <iostream>

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudaimgproc.hpp"

using namespace std;
using namespace cv;

class App
{
public:
    App(int argc, const char* argv[]);

    int run();

private:
    void help();

    void OpenClose();
    void ErodeDilate();

    static void OpenCloseCallback(int, void*);
    static void ErodeDilateCallback(int, void*);

    cuda::GpuMat src, dst;

    int element_shape;

    int max_iters;
    int open_close_pos;
    int erode_dilate_pos;
};

App::App(int argc, const char* argv[])
{
    element_shape = MORPH_RECT;
    open_close_pos = erode_dilate_pos = max_iters = 10;

    if (argc == 2 && String(argv[1]) == "--help")
    {
        help();
        exit(0);
    }

    String filename = argc == 2 ? argv[1] : "/home/nvidia/Downloads/cv45.jpg";

    Mat img = imread(filename);
    if (img.empty())
    {
        cerr << "Can't open image " << filename.c_str() << endl;
        exit(-1);
    }
    cout <<"image.size()      "<<img.size()<<endl;
    cout <<"image.depth()     "<<img.depth()<<endl;
    cout <<"image.channels()  "<<img.channels()<<endl;
    cout <<"image.total()     "<<img.total()<<endl;
    cout <<"image.type        "<<img.type()<<endl;
    cout <<"CV_8UC3           "<<CV_8UC3<<endl;
    cout <<"img.start-end     "<<img.dataend -img.datastart<<endl;


    src.upload(img);
    if (src.channels() == 3)
    {
        // gpu support only 4th src images
        cuda::GpuMat src4ch;
        cuda::cvtColor(src, src4ch, COLOR_BGR2BGRA);
        src = src4ch;
        cout <<"src.data        "<<(src.data == src4ch.data)<<endl;  //1
//        cout <<"src4ch.data     "<<src4ch.data<<endl;
        cout <<"src.size()      "<<src.size()<<endl;
        cout <<"src.depth()     "<<src.depth()<<endl;
        cout <<"src.channels()  "<<src.channels()<<endl;
//        cout <<"src.total()   "<<src.total()<<endl;
        cout <<"src.step        "<<src.step<<endl;
        cout <<"src.rows        "<<src.rows<<endl;
        cout <<"src.type        "<<src.type()<<endl;
        cout <<"CV_8UC4         "<<CV_8UC4<<endl;
        cout <<"src.iscontinue  "<<src.isContinuous()<<endl;
        cout <<"src.start-end   "<<src.dataend -src.datastart<<endl;
    }

    help();

    cuda::printShortCudaDeviceInfo(cuda::getDevice());
}

int App::run()
{
    // create windows for output images
    namedWindow("Open/Close");
    namedWindow("Erode/Dilate");

    createTrackbar("iterations", "Open/Close", &open_close_pos, max_iters * 2 + 1, OpenCloseCallback, this);
    createTrackbar("iterations", "Erode/Dilate", &erode_dilate_pos, max_iters * 2 + 1, ErodeDilateCallback, this);

    for(;;)
    {
        OpenClose();
        ErodeDilate();

        char c = (char) waitKey();

        switch (c)
        {
        case 27:
            return 0;
            break;

        case 'e':
            element_shape = MORPH_ELLIPSE;
            break;

        case 'r':
            element_shape = MORPH_RECT;
            break;

        case 'c':
            element_shape = MORPH_CROSS;
            break;

        case ' ':
            element_shape = (element_shape + 1) % 3;
            break;
        }
    }
}

void App::help()
{
    cout << "Show off image morphology: erosion, dialation, open and close \n";
    cout << "Call: \n";
    cout << "   gpu-example-morphology [image] \n";
    cout << "This program also shows use of rect, ellipse and cross kernels \n" << endl;

    cout << "Hot keys: \n";
    cout << "\tESC - quit the program \n";
    cout << "\tr - use rectangle structuring element \n";
    cout << "\te - use elliptic structuring element \n";
    cout << "\tc - use cross-shaped structuring element \n";
    cout << "\tSPACE - loop through all the options \n" << endl;
}

void App::OpenClose()
{
    int n = open_close_pos - max_iters;
    int an = n > 0 ? n : -n;

    Mat element = getStructuringElement(element_shape, Size(an*2+1, an*2+1), Point(an, an));

    if (n < 0)
    {
        Ptr<cuda::Filter> openFilter = cuda::createMorphologyFilter(MORPH_OPEN, src.type(), element);
        openFilter->apply(src, dst);
    }
    else
    {
        Ptr<cuda::Filter> closeFilter = cuda::createMorphologyFilter(MORPH_CLOSE, src.type(), element);
        closeFilter->apply(src, dst);
    }

    Mat h_dst(dst);
    imshow("Open/Close", h_dst);
}

void App::ErodeDilate()
{
    int n = erode_dilate_pos - max_iters;
    int an = n > 0 ? n : -n;

    Mat element = getStructuringElement(element_shape, Size(an*2+1, an*2+1), Point(an, an));

    if (n < 0)
    {
        Ptr<cuda::Filter> erodeFilter = cuda::createMorphologyFilter(MORPH_ERODE, src.type(), element);
        erodeFilter->apply(src, dst);
    }
    else
    {
        Ptr<cuda::Filter> dilateFilter = cuda::createMorphologyFilter(MORPH_DILATE, src.type(), element);
        dilateFilter->apply(src, dst);
    }

    Mat h_dst(dst);
    imshow("Erode/Dilate", h_dst);
}

void App::OpenCloseCallback(int, void* data)
{
    App* thiz = (App*) data;
    thiz->OpenClose();
}

void App::ErodeDilateCallback(int, void* data)
{
    App* thiz = (App*) data;
    thiz->ErodeDilate();
}

int morphology_main(int argc, const char* argv[])
{
    App app(argc, argv);
    return app.run();
}
