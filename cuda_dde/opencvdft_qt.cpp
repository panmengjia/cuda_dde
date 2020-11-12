
#include "main.h"

static void help()
{
    printf("\nThis program demonstrated the use of the discrete Fourier transform (dft)\n"
        "The dft of an image is taken and it's power spectrum is displayed.\n"
        "Usage:\n"
        "./dft [image_name -- default lena.jpg]\n");
}

const char* keys =
{
    "{1| |lena.jpg|input image file}"
};

int opencvdft(int argc, const char** argv)
{
    help();
    CommandLineParser parser(argc, argv, keys);        // opencv���������������в�������
    string filename = parser.get<string>("1");

    //    Mat img = imread(filename.c_str(), CV_LOAD_IMAGE_GRAYSCALE);    //�ԻҶ�ͼ�����
    Mat img = imread("/home/nvidia/Downloads/8700e1c96cbbdbfc8bb32a700fd8fc85.jpg", IMREAD_GRAYSCALE);
    imshow("img", img);
    if (img.empty())
    {
        help();
        printf("Cannot read image file: %s\n", filename.c_str());
        return -1;
    }
    int M = getOptimalDFTSize(img.rows);                               // ������DFT�ߴ磬Ϊ2�Ĵη�
    int N = getOptimalDFTSize(img.cols);                                 //ͬ��
    Mat padded;
    copyMakeBorder(img, padded, 0, M - img.rows, 0, N - img.cols, BORDER_CONSTANT, Scalar::all(0));   // opencv�еı߽���չ�������ṩ���ַ�ʽ��չ

    Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };          // Mat ���飬��һ��Ϊ��չ���ͼ��һ��Ϊ��ͼ��
    imshow("planes[0]", planes[0]);
    imshow("planes[1]", planes[1]);

    Mat complexImg;
    merge(planes, 2, complexImg);                                                                              // �ϲ���һ��Mat

    dft(complexImg, complexImg);                                                                              // FFT�任�� dft��Ҫһ��2ͨ����Mat

    Mat ifftImage;
    idft(complexImg, ifftImage, DFT_REAL_OUTPUT);
    normalize(ifftImage, ifftImage, 0, 1, NORM_MINMAX);
    imshow("idft", ifftImage);


    // compute log(1 + sqrt(Re(DFT(img))**2 + Im(DFT(img))**2))
    split(complexImg, planes);                                                                                     //����ͨ���� planes[0] Ϊʵ�����֣�planes[1]Ϊ��������
    magnitude(planes[0], planes[1], planes[0]);                                                          // ��ģ
    imshow("planes[0] magnitude", planes[0]);
    Mat mag = planes[0];
    mag += Scalar::all(1);
    log(mag, mag);                                                                                                      // ģ�Ķ���

    // crop the spectrum, if it has an odd number of rows or columns
    mag = mag(Rect(0, 0, mag.cols & -2, mag.rows & -2));                                        //��֤ż���ı߳�

    int cx = mag.cols / 2;
    int cy = mag.rows / 2;

    // rearrange the quadrants of Fourier image                                                        //�Ը���Ҷ�任��ͼ��������ţ�4�����飬�����ң����ϵ��� �ֱ�Ϊq0, q1, q2, q3
    // so that the origin is at the image center                                                          //  �Ե�q0��q3, q1��q2
    Mat tmp;
    Mat q0(mag, Rect(0, 0, cx, cy));
    Mat q1(mag, Rect(cx, 0, cx, cy));
    Mat q2(mag, Rect(0, cy, cx, cy));
    Mat q3(mag, Rect(cx, cy, cx, cy));

    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);

    normalize(mag, mag, 0, 1, NORM_MINMAX);                                                           // �淶��ֵ�� 0~1 ��ʾͼƬ����Ҫ

    imshow("spectrum magnitude", mag);
    waitKey();
    return 0;
}
