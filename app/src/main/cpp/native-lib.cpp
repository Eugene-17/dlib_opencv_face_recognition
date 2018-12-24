#include <jni.h>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include <dlib/opencv.h>
#include <dlib/image_io.h>
#include <dlib/image_processing.h>
#include <dlib/dnn.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_processing/frontal_face_detector.h>

#include "capture_v4l.h"

#define DLIB_JPEG_SUPPORT

using namespace dlib;
using namespace std;
using namespace cv;
using namespace cv::ml;

template<template<int, template<typename> class, int, typename> class block, int N,
        template<typename> class BN, typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

template<template<int, template<typename> class, int, typename> class block, int N,
        template<typename> class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

template<int N, template<typename> class BN, int stride, typename SUBNET>
using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

template<int N, typename SUBNET> using ares = relu<residual<block, N, affine, SUBNET>>;
template<int N, typename SUBNET> using ares_down = relu<residual_down<block, N, affine, SUBNET>>;

template<typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
template<typename SUBNET> using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template<typename SUBNET> using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template<typename SUBNET> using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template<typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128, avg_pool_everything<
        alevel0<
                alevel1<
                        alevel2<
                                alevel3<
                                        alevel4<
                                                max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2,
                                                        input_rgb_image_sized<150>
                                                >>>>>>>>>>>>;


#define JNI_METHOD(NAME) Java_com_example_hungpn_facerecognition_MainActivity_##NAME

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_hungpn_facerecognition_MainActivity_stringFromJNI(
        JNIEnv *env,
        jobject /* this */) {
    std::string hello = "Hello from Binh";
    return env->NewStringUTF(hello.c_str());
}

extern "C" JNIEXPORT jint JNICALL
JNI_METHOD(captureCamera)(JNIEnv *env, jobject) {
    Mat imgbuf, img;
    int fd;
    fd = open("/dev/video0", O_RDWR);
    if (fd == -1) {
        perror("Opening video device");
        return -1;
    }
    print_caps(fd);
    init_mmap(fd);
    start_capture(fd);

    capture_image(fd);
    imgbuf = Mat(FRAME_HEIGHT, FRAME_WIDTH, CV_8U, (void *) buffer);
    img = imdecode(imgbuf, IMREAD_COLOR);
    //imwrite("/sdcard/captured_img.jpg", img);
    return 100;
}

extern "C" JNIEXPORT jint JNICALL
JNI_METHOD(faceRecognition)(JNIEnv *env, jobject) {
    Mat img_inp;
    //img_inp = imread("lena.jpg");
    //dlib::cv_image<dlib::rgb_pixel> cimg(img_inp);

    //const long w = cimg.nc();
    //const long h = cimg.nr();

    // face_detector objectstringFromJNI
    //dlib::frontal_face_detector face_detector = dlib::get_frontal_face_detector();
    //dlib::shape_predictor sp;   //shape predictor 5 landmarks
    //dlib::deserialize("shape_predictor_5_face_landmarks.data") >> sp;
    //dlib::deserialize("shape_predictor_68_face_landmarks.data") >> sp;
    return 100;
}


