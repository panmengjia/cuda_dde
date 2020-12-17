/*
// Sample demonstrating interoperability of OpenCV UMat with OpenGL texture.
// At first, the data obtained from video file or camera and placed onto
// OpenGL texture, following mapping of this OpenGL texture to OpenCV UMat
// and call cv::Blur function. The result is mapped back to OpenGL texture
// and rendered through OpenGL API.
*/
#if defined(_WIN32)
# define WIN32_LEAN_AND_MEAN
# include <windows.h>
#elif defined(__linux__)
# include <X11/X.h>
# include <X11/Xlib.h>
#endif

#include <iostream>
#include <queue>
#include <string>

#include <stdio.h>

#include "opencv2/core.hpp"
#include "opencv2/core/opengl.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"

#include "winapp.hpp"

#if defined(_WIN32)
# pragma comment(lib, "opengl32.lib")
# pragma comment(lib, "glu32.lib")
#endif

#include "main.h"

class GLWinApp : public WinApp
{
public:
    enum MODE
    {
        MODE_CPU = 0,
        MODE_GPU
    };

    GLWinApp(int width, int height, std::string& window_name, cv::VideoCapture& cap) :
        WinApp(width, height, window_name)
    {
        m_shutdown        = false;
        m_use_buffer      = false;
        m_demo_processing = true;
        m_mode            = MODE_CPU;
        m_modeStr[0]      = cv::String("Processing on CPU");
        m_modeStr[1]      = cv::String("Processing on GPU");
        m_cap             = cap;
    }

    ~GLWinApp() {}

    virtual void cleanup() CV_OVERRIDE
    {
        m_shutdown = true;
#if defined(__linux__)
        glXMakeCurrent(m_display, None, NULL);
        glXDestroyContext(m_display, m_glctx);
#endif
        WinApp::cleanup();
    }

#if defined(_WIN32)
    virtual LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) CV_OVERRIDE
    {
        switch (message)
        {
        case WM_CHAR:
            if (wParam == '1')
            {
                set_mode(MODE_CPU);
                return EXIT_SUCCESS;
            }
            if (wParam == '2')
            {
                set_mode(MODE_GPU);
                return EXIT_SUCCESS;
            }
            else if (wParam == '9')
            {
                toggle_buffer();
                return EXIT_SUCCESS;
            }
            else if (wParam == VK_SPACE)
            {
                m_demo_processing = !m_demo_processing;
                return EXIT_SUCCESS;
            }
            else if (wParam == VK_ESCAPE)
            {
                cleanup();
                return EXIT_SUCCESS;
            }
            break;

        case WM_CLOSE:
            cleanup();
            return EXIT_SUCCESS;

        case WM_DESTROY:
            ::PostQuitMessage(0);
            return EXIT_SUCCESS;
        }

        return ::DefWindowProc(hWnd, message, wParam, lParam);
    }
#endif

#if defined(__linux__)
    int handle_event(XEvent& e) CV_OVERRIDE
    {
        switch(e.type)
        {
        case ClientMessage:
            if ((Atom)e.xclient.data.l[0] == m_WM_DELETE_WINDOW)
            {
                m_end_loop = true;
                cleanup();
            }
            else
            {
                return EXIT_SUCCESS;
            }
            break;
        case Expose:
            render();
            break;
        case KeyPress:
            switch(keycode_to_keysym(e.xkey.keycode))
            {
            case XK_space:
                m_demo_processing = !m_demo_processing;
                break;
            case XK_1:
                set_mode(MODE_CPU);
                break;
            case XK_2:
                set_mode(MODE_GPU);
                break;
            case XK_9:
                toggle_buffer();
                break;
            case XK_Escape:
                m_end_loop = true;
                cleanup();
                break;
            }
            break;
        default:
            return EXIT_SUCCESS;
        }
        return 1;
    }
#endif

    int init() CV_OVERRIDE
    {
#if defined(_WIN32)
        m_hDC = GetDC(m_hWnd);

        if (setup_pixel_format() != 0)
        {
            std::cerr << "Can't setup pixel format" << std::endl;
            return EXIT_FAILURE;
        }

        m_hRC = wglCreateContext(m_hDC);
        wglMakeCurrent(m_hDC, m_hRC);
#elif defined(__linux__)
        m_glctx = glXCreateContext(m_display, m_visual_info, NULL, GL_TRUE);
        glXMakeCurrent(m_display, m_window, m_glctx);
#endif

        glEnable(GL_TEXTURE_2D);
        glEnable(GL_DEPTH_TEST);

        glViewport(0, 0, m_width, m_height);

        if (cv::ocl::haveOpenCL())
        {
            (void) cv::ogl::ocl::initializeContextFromGL();
        }

        m_oclDevName = cv::ocl::useOpenCL() ?
            cv::ocl::Context::getDefault().device(0).name() :
            (char*) "No OpenCL device";

        return EXIT_SUCCESS;
    } // init()

    int get_frame(cv::ogl::Texture2D& texture, cv::ogl::Buffer& buffer, bool do_buffer)
    {
        if (!m_cap.read(m_frame_bgr))
            return EXIT_FAILURE;

        cv::cvtColor(m_frame_bgr, m_frame_rgba, cv::COLOR_RGB2RGBA);

        if (do_buffer)
            buffer.copyFrom(m_frame_rgba, cv::ogl::Buffer::PIXEL_UNPACK_BUFFER, true);
        else
            texture.copyFrom(m_frame_rgba, true);

        return EXIT_SUCCESS;
    }

    void print_info(MODE mode, double time, cv::String& oclDevName)
    {
#if defined(_WIN32)
        HDC hDC = m_hDC;

        HFONT hFont = (HFONT)::GetStockObject(SYSTEM_FONT);

        HFONT hOldFont = (HFONT)::SelectObject(hDC, hFont);

        if (hOldFont)
        {
            TEXTMETRIC tm;
            ::GetTextMetrics(hDC, &tm);

            char buf[256+1];
            int  y = 0;

            buf[0] = 0;
            sprintf_s(buf, sizeof(buf)-1, "Mode: %s OpenGL %s", m_modeStr[mode].c_str(), use_buffer() ? "buffer" : "texture");
            ::TextOut(hDC, 0, y, buf, (int)strlen(buf));

            y += tm.tmHeight;
            buf[0] = 0;
            sprintf_s(buf, sizeof(buf)-1, "Time, msec: %2.1f", time);
            ::TextOut(hDC, 0, y, buf, (int)strlen(buf));

            y += tm.tmHeight;
            buf[0] = 0;
            sprintf_s(buf, sizeof(buf)-1, "OpenCL device: %s", oclDevName.c_str());
            ::TextOut(hDC, 0, y, buf, (int)strlen(buf));

            ::SelectObject(hDC, hOldFont);
        }
#elif defined(__linux__)

        char buf[256+1];
        snprintf(buf, sizeof(buf)-1, "Time, msec: %2.1f, Mode: %s OpenGL %s, Device: %s", time, m_modeStr[mode].c_str(), use_buffer() ? "buffer" : "texture", oclDevName.c_str());
        XStoreName(m_display, m_window, buf);
#endif
    }

    void idle() CV_OVERRIDE
    {
        render();
    }

    int render() CV_OVERRIDE
    {
        try
        {
            if (m_shutdown)
                return EXIT_SUCCESS;

            int r;
            cv::ogl::Texture2D texture;
            cv::ogl::Buffer buffer;

            texture.setAutoRelease(true);
            buffer.setAutoRelease(true);

            MODE mode = get_mode();
            bool do_buffer = use_buffer();

            r = get_frame(texture, buffer, do_buffer);
            if (r != 0)
            {
                return EXIT_FAILURE;
            }

            switch (mode)
            {
                case MODE_CPU: // process frame on CPU
                    processFrameCPU(texture, buffer, do_buffer);
                    break;

                case MODE_GPU: // process frame on GPU
                    processFrameGPU(texture, buffer, do_buffer);
                    break;
            } // switch

            if (do_buffer) // buffer -> texture
            {
                cv::Mat m(m_height, m_width, CV_8UC4);
                buffer.copyTo(m);
                texture.copyFrom(m, true);
            }

#if defined(__linux__)
            XWindowAttributes window_attributes;
            XGetWindowAttributes(m_display, m_window, &window_attributes);
            glViewport(0, 0, window_attributes.width, window_attributes.height);
#endif

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glLoadIdentity();
            glEnable(GL_TEXTURE_2D);

            texture.bind();

            glBegin(GL_QUADS);
            glTexCoord2f(0.0f, 0.0f); glVertex3f(-1.0f, 1.0f, 0.1f);
            glTexCoord2f(0.0f, 1.0f); glVertex3f(-1.0f, -1.0f, 0.1f);
            glTexCoord2f(1.0f, 1.0f); glVertex3f(1.0f, -1.0f, 0.1f);
            glTexCoord2f(1.0f, 0.0f); glVertex3f(1.0f, 1.0f, 0.1f);
            glEnd();

#if defined(_WIN32)
            SwapBuffers(m_hDC);
#elif defined(__linux__)
            glXSwapBuffers(m_display, m_window);
#endif

            print_info(mode, m_timer.getTimeMilli(), m_oclDevName);
        }


        catch (const cv::Exception& e)
        {
            std::cerr << "Exception: " << e.what() << std::endl;
            return 10;
        }

        return EXIT_SUCCESS;
    }

protected:

    void processFrameCPU(cv::ogl::Texture2D& texture, cv::ogl::Buffer& buffer, bool do_buffer)
    {
        cv::Mat m(m_height, m_width, CV_8UC4);

        m_timer.reset();
        m_timer.start();

        if (do_buffer)
            buffer.copyTo(m);
        else
            texture.copyTo(m);

        if (m_demo_processing)
        {
            // blur texture image with OpenCV on CPU
            cv::blur(m, m, cv::Size(15, 15));
        }

        if (do_buffer)
            buffer.copyFrom(m, cv::ogl::Buffer::PIXEL_UNPACK_BUFFER, true);
        else
            texture.copyFrom(m, true);

        m_timer.stop();
    }

    void processFrameGPU(cv::ogl::Texture2D& texture, cv::ogl::Buffer& buffer, bool do_buffer)
    {
        cv::UMat u;

        m_timer.reset();
        m_timer.start();

        if (do_buffer)
            u = cv::ogl::mapGLBuffer(buffer);
        else
            cv::ogl::convertFromGLTexture2D(texture, u);

        if (m_demo_processing)
        {
            // blur texture image with OpenCV on GPU with OpenCL
            cv::blur(u, u, cv::Size(15, 15));
        }

        if (do_buffer)
            cv::ogl::unmapGLBuffer(u);
        else
            cv::ogl::convertToGLTexture2D(u, texture);

        m_timer.stop();
    }

#if defined(_WIN32)
    int setup_pixel_format()
    {
        PIXELFORMATDESCRIPTOR  pfd;

        pfd.nSize           = sizeof(PIXELFORMATDESCRIPTOR);
        pfd.nVersion        = 1;
        pfd.dwFlags         = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL  | PFD_DOUBLEBUFFER;
        pfd.iPixelType      = PFD_TYPE_RGBA;
        pfd.cColorBits      = 24;
        pfd.cRedBits        = 8;
        pfd.cRedShift       = 0;
        pfd.cGreenBits      = 8;
        pfd.cGreenShift     = 0;
        pfd.cBlueBits       = 8;
        pfd.cBlueShift      = 0;
        pfd.cAlphaBits      = 8;
        pfd.cAlphaShift     = 0;
        pfd.cAccumBits      = 0;
        pfd.cAccumRedBits   = 0;
        pfd.cAccumGreenBits = 0;
        pfd.cAccumBlueBits  = 0;
        pfd.cAccumAlphaBits = 0;
        pfd.cDepthBits      = 24;
        pfd.cStencilBits    = 8;
        pfd.cAuxBuffers     = 0;
        pfd.iLayerType      = PFD_MAIN_PLANE;
        pfd.bReserved       = 0;
        pfd.dwLayerMask     = 0;
        pfd.dwVisibleMask   = 0;
        pfd.dwDamageMask    = 0;

        int pfmt = ChoosePixelFormat(m_hDC, &pfd);
        if (pfmt == 0)
            return EXIT_FAILURE;

        if (SetPixelFormat(m_hDC, pfmt, &pfd) == 0)
            return -2;

        return EXIT_SUCCESS;
    }
#endif

#if defined(__linux__)
    KeySym keycode_to_keysym(unsigned keycode)
    {   // note that XKeycodeToKeysym() is considered deprecated
        int keysyms_per_keycode_return = 0;
        KeySym *keysyms = XGetKeyboardMapping(m_display, keycode, 1, &keysyms_per_keycode_return);
        KeySym keysym = keysyms[0];
        XFree(keysyms);
        return keysym;
    }
#endif

    bool use_buffer()        { return m_use_buffer; }
    void toggle_buffer()     { m_use_buffer = !m_use_buffer; }
    MODE get_mode()          { return m_mode; }
    void set_mode(MODE mode) { m_mode = mode; }

private:
    bool               m_shutdown;
    bool               m_use_buffer;
    bool               m_demo_processing;
    MODE               m_mode;
    cv::String         m_modeStr[2];
#if defined(_WIN32)
    HDC                m_hDC;
    HGLRC              m_hRC;
#elif defined(__linux__)
    GLXContext         m_glctx;
#endif
    cv::VideoCapture   m_cap;
    cv::Mat            m_frame_bgr;
    cv::Mat            m_frame_rgba;
    cv::String         m_oclDevName;
};

static const char* keys =
{
    "{c camera | 0     | camera id }"
    "{f file   |       | movie file name  }"
};

using namespace cv;
using namespace std;


//-------------------------------------------------------------------------------

# include <X11/X.h>
# include <X11/Xlib.h>
# include <X11/Xutil.h>
#include <string>
#include <GL/gl.h>
# include <GL/glx.h>


class cv_gl
{
public:
    enum MODE
    {
        MODE_CPU = 0,
        MODE_GPU
    };

    cv_gl() {}
    cv_gl(int width, int height, std::string& window_name, cv::Mat& frame)
    {
        m_width       = width;
        m_height      = height;
        m_window_name = window_name;

        m_shutdown        = false;
        m_use_buffer      = false;
        m_demo_processing = true;
        m_mode            = MODE_CPU;
        m_modeStr[0]      = cv::String("Processing on CPU");
        m_modeStr[1]      = cv::String("Processing on GPU");
        m_frame_bgr       = frame;
        m_frame_bgr.data  = frame.data;
    }

    ~cv_gl() {}

    int handle_event(XEvent& e)
    {
        switch(e.type)
        {
        case ClientMessage:
            if ((Atom)e.xclient.data.l[0] == m_WM_DELETE_WINDOW)
            {
                m_end_loop = true;
                cleanup();
            }
            else
            {
                return EXIT_SUCCESS;
            }
            break;
        case Expose:
            render();
            break;
        case KeyPress:
            switch(keycode_to_keysym(e.xkey.keycode))
            {
            case XK_space:
                m_demo_processing = !m_demo_processing;
                break;
            case XK_1:
                set_mode(MODE_CPU);
                break;
            case XK_2:
                set_mode(MODE_GPU);
                break;
            case XK_9:
                toggle_buffer();
                break;
            case XK_Escape:
                m_end_loop = true;
                cleanup();
                break;
            }
            break;
        default:
            return EXIT_SUCCESS;
        }
        return 1;
    }

    int get_frame(cv::ogl::Texture2D& texture, cv::ogl::Buffer& buffer, bool do_buffer)
    {
//        if (!m_cap.read(m_frame_bgr))
//            return EXIT_FAILURE;
        if (m_frame_bgr.empty())
        {
            printf("frame is empty!!");
            return EXIT_FAILURE;
        }
        cv::cvtColor(m_frame_bgr, m_frame_rgba, cv::COLOR_RGB2RGBA);

        if (do_buffer)
            buffer.copyFrom(m_frame_rgba, cv::ogl::Buffer::PIXEL_UNPACK_BUFFER, true);
        else
            texture.copyFrom(m_frame_rgba, true);

        return EXIT_SUCCESS;
    }

    void idle()
    {
        render();
    }

    int init()
    {
        m_glctx = glXCreateContext(m_display, m_visual_info, NULL, GL_TRUE);
        glXMakeCurrent(m_display, m_window, m_glctx);

        glEnable(GL_TEXTURE_2D);
        glEnable(GL_DEPTH_TEST);

        glViewport(0, 0, m_width, m_height);

        if (cv::ocl::haveOpenCL())
        {
            (void) cv::ogl::ocl::initializeContextFromGL();
        }

        m_oclDevName = cv::ocl::useOpenCL() ?
            cv::ocl::Context::getDefault().device(0).name() :
            (char*) "No OpenCL device";

        return EXIT_SUCCESS;
    } // init()

    int create()
    {
        m_display = XOpenDisplay(NULL);

        if (m_display == NULL)
        {
            return -1;
        }

        m_WM_DELETE_WINDOW = XInternAtom(m_display, "WM_DELETE_WINDOW", False);

        static GLint visual_attributes[] = { GLX_RGBA, GLX_DEPTH_SIZE, 24, GLX_DOUBLEBUFFER, None };
        m_visual_info = glXChooseVisual(m_display, 0, visual_attributes);

        if (m_visual_info == NULL)
        {
            XCloseDisplay(m_display);
            return -2;
        }

        Window root = DefaultRootWindow(m_display);

        m_event_mask = ExposureMask | KeyPressMask;

        XSetWindowAttributes window_attributes;
        window_attributes.colormap = XCreateColormap(m_display, root, m_visual_info->visual, AllocNone);
        window_attributes.event_mask = m_event_mask;

        m_window = XCreateWindow(
            m_display, root, 0, 0, m_width, m_height, 0, m_visual_info->depth,
            InputOutput, m_visual_info->visual, CWColormap | CWEventMask, &window_attributes);

        XMapWindow(m_display, m_window);
        XSetWMProtocols(m_display, m_window, &m_WM_DELETE_WINDOW, 1);
        XStoreName(m_display, m_window, m_window_name.c_str());

        return init();
    }

    int run()
    {
        m_end_loop = false;

        do {
            XEvent e;

            if (!XCheckWindowEvent(m_display, m_window, m_event_mask, &e) || !handle_event(e))
            {
//                cout <<"jdhfudfhjfhg------------------------"<<thread_flag<<endl;
                if(!thread_flag)
                {
                    idle();
                    cout <<"----------------running idle--------------"<<endl;
                    thread_flag = true;
                }

            }
        } while (!m_end_loop);

        return 0;
    }

    void print_info(MODE mode, double time, cv::String& oclDevName)
    {
        char buf[256+1];
        snprintf(buf, sizeof(buf)-1, "Time, msec: %2.1f, Mode: %s OpenGL %s, Device: %s", time, m_modeStr[mode].c_str(), use_buffer() ? "buffer" : "texture", oclDevName.c_str());
        XStoreName(m_display, m_window, buf);
    }

    int render()
    {
        try
        {
            if (m_shutdown)
                return EXIT_SUCCESS;

            int r;
            cv::ogl::Texture2D texture;
            cv::ogl::Buffer buffer;

            texture.setAutoRelease(true);
            buffer.setAutoRelease(true);

            MODE mode = get_mode();
            bool do_buffer = use_buffer();

            r = get_frame(texture, buffer, do_buffer);
            if (r != 0)
            {
                return EXIT_FAILURE;
            }

            switch (mode)
            {
                case MODE_CPU: // process frame on CPU
                    processFrameCPU(texture, buffer, do_buffer);
                    break;

                case MODE_GPU: // process frame on GPU
                    processFrameGPU(texture, buffer, do_buffer);
                    break;
            } // switch

            if (do_buffer) // buffer -> texture
            {
                cv::Mat m(m_height, m_width, CV_8UC4);
                buffer.copyTo(m);
                texture.copyFrom(m, true);
            }

#if defined(__linux__)
            XWindowAttributes window_attributes;
            XGetWindowAttributes(m_display, m_window, &window_attributes);
            glViewport(0, 0, window_attributes.width, window_attributes.height);
#endif

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glLoadIdentity();
            glEnable(GL_TEXTURE_2D);

            texture.bind();

            glBegin(GL_QUADS);
            glTexCoord2f(0.0f, 0.0f); glVertex3f(-1.0f, 1.0f, 0.1f);
            glTexCoord2f(0.0f, 1.0f); glVertex3f(-1.0f, -1.0f, 0.1f);
            glTexCoord2f(1.0f, 1.0f); glVertex3f(1.0f, -1.0f, 0.1f);
            glTexCoord2f(1.0f, 0.0f); glVertex3f(1.0f, 1.0f, 0.1f);
            glEnd();

#if defined(_WIN32)
            SwapBuffers(m_hDC);
#elif defined(__linux__)
            glXSwapBuffers(m_display, m_window);
#endif

            print_info(mode, m_timer.getTimeMilli(), m_oclDevName);
        }


        catch (const cv::Exception& e)
        {
            std::cerr << "Exception: " << e.what() << std::endl;
            return 10;
        }

        return EXIT_SUCCESS;
    }

    void cleanup()
    {
        m_shutdown = true;
#if defined(__linux__)
        glXMakeCurrent(m_display, None, NULL);
        glXDestroyContext(m_display, m_glctx);
#endif
        XDestroyWindow(m_display, m_window);
        XCloseDisplay(m_display);
    }

    bool thread_flag = true;

protected:

    void processFrameCPU(cv::ogl::Texture2D& texture, cv::ogl::Buffer& buffer, bool do_buffer)
    {
        cv::Mat m(m_height, m_width, CV_8UC4);

        m_timer.reset();
        m_timer.start();

        if (do_buffer)
            buffer.copyTo(m);
        else
            texture.copyTo(m);

        if (m_demo_processing)
        {
            // blur texture image with OpenCV on CPU
            cv::blur(m, m, cv::Size(15, 15));
        }

        if (do_buffer)
            buffer.copyFrom(m, cv::ogl::Buffer::PIXEL_UNPACK_BUFFER, true);
        else
            texture.copyFrom(m, true);

        m_timer.stop();
    }

    void processFrameGPU(cv::ogl::Texture2D& texture, cv::ogl::Buffer& buffer, bool do_buffer)
    {
        cv::UMat u;

        m_timer.reset();
        m_timer.start();

        if (do_buffer)
            u = cv::ogl::mapGLBuffer(buffer);
        else
            cv::ogl::convertFromGLTexture2D(texture, u);

        if (m_demo_processing)
        {
            // blur texture image with OpenCV on GPU with OpenCL
            cv::blur(u, u, cv::Size(15, 15));
        }

        if (do_buffer)
            cv::ogl::unmapGLBuffer(u);
        else
            cv::ogl::convertToGLTexture2D(u, texture);

        m_timer.stop();
    }

    KeySym keycode_to_keysym(unsigned keycode)
    {   // note that XKeycodeToKeysym() is considered deprecated
        int keysyms_per_keycode_return = 0;
        KeySym *keysyms = XGetKeyboardMapping(m_display, keycode, 1, &keysyms_per_keycode_return);
        KeySym keysym = keysyms[0];
        XFree(keysyms);
        return keysym;
    }

    bool use_buffer()        { return m_use_buffer; }
    void toggle_buffer()     { m_use_buffer = !m_use_buffer; }
    MODE get_mode()          { return m_mode; }
    void set_mode(MODE mode) { m_mode = mode; }


private:
    Display*      m_display;
    XVisualInfo*  m_visual_info;
    Window        m_window;
    long          m_event_mask;
    Atom          m_WM_DELETE_WINDOW;
    bool          m_end_loop;
    int           m_width;
    int           m_height;
    std::string   m_window_name;
    cv::TickMeter m_timer;


    bool               m_shutdown;
    bool               m_use_buffer;
    bool               m_demo_processing;
    MODE               m_mode;
    cv::String         m_modeStr[2];

    GLXContext         m_glctx;
    cv::Mat            m_frame_bgr;
    cv::Mat            m_frame_rgba;
    cv::String         m_oclDevName;
};

//-------------------------------------------------------------------------------
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
//Mat frame;
Mat* outFrame =new Mat(IMG_HEIGHT,IMG_WIDTH,CV_8UC3);
Mat frame;

cv::VideoCapture cap;

static void* cudaFFTmulSpectrum1119float(void* flag);

//void* thread_callback(void* flag)
//{
//    static int thread_count = 0;
//    while(1)
//    {
////        cout <<*(bool*)flag<<endl;
//        if(*(bool*)flag)
//        {

//           cout <<"------------"<<++thread_count<<"--------------"<<endl;
//           if(!cap.read(frame))
//           {
//               if(frame.empty())
//               {
//                  cout <<"frame is empty"<<endl;
//               }
//               cout <<"------------------exit thread of cap.read(frame)----------------"<<endl;
//               break;
//           }
//           *(bool*)flag = false;
////           cout<<"----------frame---"<<*(bool*)flag<<"------"<<endl;
////           cv::imshow("frame",frame);
////           cv::waitKey(100);
//        }
//    }
//    cout <<"---------------------hdfjhdufnkjghn-----"<<endl;
//}


std::string gstreamer_pipeline (int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method) {
    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" +
           std::to_string(capture_height) + ", format=(string)NV12, framerate=(fraction)" + std::to_string(framerate) +
           "/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" +
           std::to_string(display_height) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}


int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv, keys);
    int    camera_id = parser.get<int>("camera");
    string file      = parser.get<string>("file");

    parser.about(
        "\nA sample program demonstrating interoperability of OpenGL and OpenCL with OpenCV.\n\n"
        "Hot keys: \n"
        "  SPACE - turn processing on/off\n"
        "    1   - process GL data through OpenCV on CPU\n"
        "    2   - process GL data through OpenCV on GPU (via OpenCL)\n"
        "    9   - toggle use of GL texture/GL buffer\n"
        "   ESC  - exit\n\n");

    parser.printMessage();


    int capture_width = 1280 ;
    int capture_height = 720 ;
    int display_width = 1920 ;
    int display_height = 1080 ;
    int framerate = 24 ;
    int flip_method = 0 ;

    std::string pipeline = gstreamer_pipeline(capture_width,
    capture_height,
    display_width,
    display_height,
    framerate,
    flip_method);

//    if (file.empty())
//        cap.open(camera_id);
//    else
//        cap.open(file.c_str());

//    cap.open(0);
    cap.open(pipeline, cv::CAP_GSTREAMER);
    cap>>frame;
    imshow("frame",frame);

    waitKey(0);

    if (!cap.isOpened())
    {
        printf("can not open camera or video file\n");
        return EXIT_FAILURE;
    }

    int width  = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(CAP_PROP_FRAME_HEIGHT);

#if defined(_WIN32)
    string wndname = "WGL Window";
#elif defined(__linux__)
    string wndname = "GLX Window";
#endif

//    GLWinApp app(width, height, wndname, cap);

//    try
//    {
//        app.create();
//        return app.run();
//    }

//    cv::waitKey(10000);
//    cap>>frame;

    cv_gl app_cv_gl(width, height, wndname, *outFrame);
    cout <<"             "<<app_cv_gl.thread_flag<<endl;

    pthread_t pid;
    pthread_create(&pid,NULL,cudaFFTmulSpectrum1119float,&app_cv_gl.thread_flag);
    pthread_detach(pid);


    try
    {
        app_cv_gl.create();
        return app_cv_gl.run();

    }
    catch (const cv::Exception& e)
    {
        cerr << "Exception: " << e.what() << endl;
        return 10;
    }
    catch (...)
    {
        cerr << "FATAL ERROR: Unknown exception" << endl;
        return 11;
    }
}



/////



////////////////////////////////////////////////////////cuda dft mulSpectrum副本////////////////////////////////////////////////////////////////////


/*
* cuda dft mulSpectrum
* 2020.11.19继承前面使用cuda核函数写的频谱相乘，修改错误，规范代码
*
*
*
*/

#define CVTCOLOR (true)

static void* cudaFFTmulSpectrum1119float(void* flag)
{
    /////////////////////////////////////////////读取txt////////////////////////////////////////////
    string str = "/home/pmj-nano/Desktop/dde1448/15/";
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
    cufftComplex* kernelComplex[15];
    Mat kernelPadded;

    for (int i = 0; i < 15; ++i) //在外面只能得到最后一次的数据结果，之前的都被覆盖了
    {
        kernelComplex[i] = (cufftComplex*)malloc(sizeof(cufftComplex) * IMG_HEIGHT * (IMG_WIDTH / 2 + 1));
        kernelMat.push_back(Mat(FILTER_HEIGHT, FILTER_WIDTH, CV_32FC1, dataAllFile[i].data()));
        copyMakeBorder(kernelMat[i], kernelPadded, 0, IMG_HEIGHT - kernelMat[i].rows, 0, IMG_WIDTH - kernelMat[i].cols, BORDER_CONSTANT, Scalar(0)); //Scalar::all(0)
        fftKernel((cufftReal*)kernelPadded.data, kernelComplex[i], kernelPadded.rows, kernelPadded.cols);/*IMG_HEIGHT, IMG_WIDTH);*/
        //kernelMat[i].convertTo(kernelMat[i], CV_16FC1);
    }

    cout << kernelMat[6] << endl;
    //测试代码
    //测试fftKernel与dft结果的异同，是完全一样的
    //opencv dft的complex形式与cufft的cudaComplex的数据结构是一样的
    //行优先存储，连续
    cufftComplex* kernelComplex6 = (cufftComplex*)malloc(sizeof(cufftComplex) * IMG_HEIGHT * (IMG_WIDTH / 2 + 1));
    Mat kernelPadded6;
    copyMakeBorder(kernelMat[6], kernelPadded6, 0, IMG_HEIGHT - kernelMat[6].rows, 0, IMG_WIDTH - kernelMat[6].cols, BORDER_CONSTANT, Scalar::all(0));   // opencv中的边界扩展函数，提供多种方式扩展
    fftKernel((cufftReal*)kernelPadded6.data, kernelComplex6, kernelPadded6.rows, kernelPadded6.cols);
    //cout << kernelpadded6 << endl;
    cout << kernelMat[6] << endl;

    //通道连续存储，两者可以完美对应
    Mat kernelComplex6Mat = Mat(IMG_HEIGHT, (IMG_WIDTH / 2 + 1), CV_32FC2, kernelComplex6);
    cout << kernelComplex6Mat(Rect(0, 0, 15, 15)) << endl;

    Mat planes[] = { kernelPadded6,Mat::zeros(kernelPadded6.size(),CV_32F) };
    Mat complexImg;
    merge(planes, 2, complexImg);
    dft(complexImg, complexImg);
    /////////////////////////////////////////////读取txt////////////////////////////////////////////

    /////////////////////////////////////////////DDEfilter////////////////////////////////////////////
//    VideoCapture* cap = new VideoCapture(VIDEO_DIR);
//    if (!cap->isOpened())
//    {
//        cout << "video is empty!!" << endl;
//    }
//    Mat frame; //1080 1920
    Mat singleFrame, meanSF, stdDevSF, selectedKernelMat, SFfp32, SFfp32out, frameYUV; //outFrame,
    vector<Mat> frame3channels, frame3YUV;
    double threshRate;

    static uint thread_count=0;

    while (1)
    {
        if(*(bool*)flag)
        {
            double time = (double)getTickCount();
            double time2 = (double)getTickCount();
//            cap->read(frame);
            cout <<"------------"<<++thread_count<<"--------------"<<endl;
            if(!cap.read(frame))
            {
                if(frame.empty())
                {
                   cout <<"frame is empty"<<endl;
                }
                cout <<"------------------exit thread of dde----------------"<<endl;
                break;
            }

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
            time2 = (double)(getTickCount() - time2) * 1000 / getTickFrequency();
            cout << "   time while pre   =  " << time2 << "  ms" << endl;

            time2 = (double)getTickCount();
            //singleFrame.convertTo(SFfp16, CV_16FC1);
            //SFfp16out = Mat(SFfp16.size(), CV_16FC1); //已开辟内存，但是随机数，都是-23.0
            singleFrame.convertTo(SFfp32, CV_32FC1);
            SFfp32out = Mat(SFfp32.size(), CV_32FC1); //已开辟内存，但是随机数，都是-23.0

            time2 = (double)(getTickCount() - time2) * 1000 / getTickFrequency();
            cout << "   time creat SFfp32out convertTo SFfp32Frame   =  " << time2 << "  ms" << endl;

            time2 = (double)getTickCount();
            /* laplacianFilter_GPU_wrapper(SFfp16, SFfp16out, selectedKernelMat);*/
            //convfp16((__half*)SFfp16.data, (__half*)SFfp16out.data, (__half*)selectedKernelMat.data, SFfp16.cols, SFfp16.rows); //.ptr() .data返回的都是一级指针（列指针）

            //timeCall = (double)getTickCount();
            //convolveDFT(SFfp32, selectedKernelMat, SFfp32out);

            fftImgKernel((cufftReal*)SFfp32.data, (cufftReal*)SFfp32out.data, kernelComplex[(unsigned int)(threshRate * 10)], SFfp32.rows, SFfp32.cols);

            time2 = (double)(getTickCount() - time2) * 1000 / getTickFrequency();
            cout << "   time convDFT   =  " << time2 << "  ms" << endl;

            time2 = (double)getTickCount();
            //SFfp32out.convertTo(SFfp32out, CV_8UC1);
            SFfp32out.convertTo(SFfp32out, CV_8UC1,1/((float)IMG_HEIGHT*(float)IMG_WIDTH)); //真是不懂到底要不要乘以255
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
            cvtColor(frameYUV, *outFrame, COLOR_YCrCb2BGR);
            time2 = (double)(getTickCount() - time2) * 1000 / getTickFrequency();
            cout << "   time cvtColor yuv  =  " << time2 << "  ms" << endl;

            time2 = (double)getTickCount();
            //        imshow("frame", frame);
            //        imshow("outFrame", outFrame);
            time2 = (double)(getTickCount() - time2) * 1000 / getTickFrequency();
            cout << "   time imshow  =  " << time2 << "  ms" << endl;


            double fps = getTickFrequency() / (getTickCount() - time);
            time = (double)(getTickCount() - time) * 1000 / getTickFrequency();
            cout << "  time total =  " << time << "  ms" << endl;
            cout << "fps" << fps << endl;
            //        waitKey(1);

            ///
            *(bool*)flag = false;

        }
    }

//    delete cap;
    /////////////////////////////////////////////DDEfilter////////////////////////////////////////////
}

////////////////////////////////////////////////////////cuda dft mulSpectrum副本////////////////////////////////////////////////////////////////////

