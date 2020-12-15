
#include <GL/glut.h>

void myDisplay()
{
    //清除，GL_COLOR_BUFFER_BIT表示清除颜色，glClear函数还可以清除其它的东西。
    glClear(GL_COLOR_BUFFER_BIT);
    glClearColor(0,1,0,0);

    //画一个矩形，四个参数分别表示了位于对角线上的两个点的横、纵坐标。
    glRectf(-0.5f, -0.5f, 0.5f, 0.5f);

    //保证前面的OpenGL命令立即执行（而不是让它们在缓冲区中等待），其作用跟fflush(stdout)类似。
    glFlush();
}

int main1(int argc, char *argv[])
{
    //对glut初始化，这个函数必须在其它glut使用之前调用一次。
    //其格式比较死板，一般用glutInit(&argc, argv)就可以了。
    glutInit(&argc, argv);

    //设置显示方式，其中GLUT_RGB表示使用RGB颜色，与之对应的还有GLUT_INDEX（表示使用索引颜色）。
    //GLUT_SINGLE表示使用单缓冲，与之对应的还有GLUT_DOUBLE（使用双缓冲）。
    glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);

    glutInitWindowPosition(100, 100);//设置窗口在屏幕中的位置
    glutInitWindowSize(400, 400);//设置窗口的大小

    //根据前面设置的信息创建窗口，参数将被作为窗口的标题。
    //注意：窗口被创建后，并不立即显示到屏幕上。需要调用glutMainLoop才能看到窗口。
    glutCreateWindow("第一个OpenGL程序");

    //设置一个函数，当需要进行画图时，这个函数就会被调用。
    //这个说法不够准确，暂时这样说吧。
    glutDisplayFunc(myDisplay);

    //进行一个消息循环。
    //只需要知道这个函数可以显示窗口，并且等待窗口关闭后才会返回，这就足够了。
    glutMainLoop();

    return 0;
}
//----------------------------------------------------------------
#include<GL/gl.h>
#include<GL/glu.h>
#include<GL/freeglut.h>
#include<iostream>
#include<cstdio>
#include<cstring>
using namespace std;

void display(){
    glColor3f(0.25,0.25,0.0);
    glBegin(GL_POLYGON);
        glVertex3f(0.25,0.25,0.0);
        glVertex3f(0.75,0.25,0.0);
        glVertex3f(0.75,0.75,0.0);
        glVertex3f(0.25,0.75,0.0);
    glEnd();
    glFlush();
}

void init(){
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0,1.0,0.0,1.0,-1.0,1.0);
}

int main3(int argc,char** argv){
    glutInit(&argc,argv);
    glutInitDisplayMode(GLUT_SINGLE |GLUT_RGB);
    glutInitWindowSize(250,250);
    glutInitWindowPosition(100,100);
    glutCreateWindow("hello");
    init();
    glutDisplayFunc(display);
    glutMainLoop();
    return 0;
}
//----------------------------------------------------------------------------------------

#include <iostream>

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN 1
    #define NOMINMAX 1
    #include <windows.h>
#endif

#if defined(__APPLE__)
    #include <OpenGL/gl.h>
    #include <OpenGL/glu.h>
#else
    #include <GL/gl.h>
    #include <GL/glu.h>
#endif

#include "opencv2/core.hpp"
#include "opencv2/core/opengl.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;
using namespace cv::cuda;

const int win_width = 800;
const int win_height = 640;

struct DrawData
{
    ogl::Arrays arr;
    ogl::Texture2D tex;
    ogl::Buffer indices;
};

void draw(void* userdata);

void draw(void* userdata)
{
    DrawData* data = static_cast<DrawData*>(userdata);

    glRotated(0.6, 0, 1, 0);

    ogl::render(data->arr, data->indices, ogl::TRIANGLES);
}

int main_gl(int argc, char* argv[])
{
    string filename;
    if (argc < 2)
    {
//        cout << "Usage: " << argv[0] << " image" << endl;
        filename = "/home/nvidia/Desktop/lena.jpg";
    }
    else
        filename = argv[1];

    Mat img = imread(filename);
    if (img.empty())
    {
        cerr << "Can't open image " << filename << endl;
        return -1;
    }

    namedWindow("OpenGL", WINDOW_OPENGL);
    resizeWindow("OpenGL", win_width, win_height);

    Mat_<Vec2f> vertex(1, 4);
    vertex << Vec2f(-1, 1), Vec2f(-1, -1), Vec2f(1, -1), Vec2f(1, 1);

    Mat_<Vec2f> texCoords(1, 4);
    texCoords << Vec2f(0, 0), Vec2f(0, 1), Vec2f(1, 1), Vec2f(1, 0);

    Mat_<int> indices(1, 6);
    indices << 0, 1, 2, 2, 3, 0;

    DrawData data;

    data.arr.setVertexArray(vertex);
    data.arr.setTexCoordArray(texCoords);
    data.indices.copyFrom(indices);
    data.tex.copyFrom(img);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, (double)win_width / win_height, 0.1, 100.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(0, 0, 3, 0, 0, 0, 0, 1, 0);

    glEnable(GL_TEXTURE_2D);
    data.tex.bind();

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexEnvi(GL_TEXTURE_2D, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    glDisable(GL_CULL_FACE);

    setOpenGlDrawCallback("OpenGL", draw, &data);

    for (;;)
    {
        updateWindow("OpenGL");
        char key = (char)waitKey(40);
        if (key == 27)
            break;
    }

    setOpenGlDrawCallback("OpenGL", 0, 0);
    destroyAllWindows();

    return 0;
}

//----------------------------------------------------------------------------------------





#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/opengl.hpp>


using namespace cv;
using namespace std;


int main()
{
//    Mat img=imread("/home/nvidia/Downloads/cv45.jpg");
//    imshow("img",img);
//    main_gl(0,0);
    main1(0,0);

    waitKey(0);
    return 0;
}
