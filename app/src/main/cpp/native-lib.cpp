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
JNI_METHOD(faceRecognition)(JNIEnv *env, jobject) {
    // khai bao bo detect face.
    frontal_face_detector detector = get_frontal_face_detector();
    // khai bao bo tim ra 5 diem tren mat nguoi ( mat mui mieng) de can chinh mat nguoi
    shape_predictor sp;
    // load model can chinh khuon mat
    deserialize("/sdcard/GR2/shape_predictor_5_face_landmarks.dat") >> sp;
    anet_type net;
    // load model trich chon dac trung khuon mat
    deserialize("/sdcard/GR2/dlib_face_recognition_resnet_model_v1.dat") >> net;

    // Khai bao anh
    cv::Mat temp;

    temp = imread("/sdcard/GR2/lena.jpg", CV_LOAD_IMAGE_COLOR);

    // chuyen anh dang cv:: Mat ve dang chuan cua dlib
    cv_image<bgr_pixel> img(temp);

    // khai bao vec to chua cac khuon mat detect duoc
    std::vector<matrix<rgb_pixel>> faces;

    // Run the face detector on the image, and for each face extract a
    // copy that has been normalized to 150x150 pixels in size and appropriately rotated
    // and centered.

    for (auto face : detector(img))
    {
        auto shape = sp(img, face);
        matrix<rgb_pixel> face_chip;
        //extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
        //faces.push_back(face_chip);
    }

    // kiem tra xem co khuon mat nao duoc tim thay hay khong
    if (faces.size() == 0)
    {
    }

    // khai bao va tinh toan vec to 128 dac trung cho cac khuon mat vua tim duoc
    std::vector<matrix<float, 0, 1>> face_descriptors = net(faces);
    // dung opencv de load model SVM da duoc train
    // khai bao model
    Ptr<SVM> svmNew;
    // khai bao ten model
    string svmFile = "/sdcard/GR2/opencv_svm_model.xml";
    // load model SVM da train tren ubuntu su dung python
    svmNew = Algorithm::load<SVM>(svmFile);

    // khai bao mang vecto de chuyen doi vecto 128 chieu dlib vua trich chon duoc ve dang chuan de su dung SVM du doan ket qua
    float sampleMat[128];
    // khai bao va detect, tim ra vi tri khuon mat, chieu rong, chieu cao
    std::vector<dlib::rectangle> faces_rec = detector(img);
    for (int j = 0; j < face_descriptors.size(); j++)
    {
        int x, y, w, h;
        x = faces_rec[j].left();
        y = faces_rec[j].top();
        w = faces_rec[j].right() - x;
        h = faces_rec[j].bottom() - y;
        cv::Point pt1(x, y);
        cv::Point pt2(x + w, y + h);
        cv::Point pt_text(x, faces_rec[j].bottom());
        // chuyen sang dang du lieu chuan
        for (int i = 0; i < 128; i++)
        {
            //data[i] = face_descriptors[0](i);
            sampleMat[i] = face_descriptors[j](i);

        }
        // chuyen sang du lieu chuan la tham so dau vao cho ham predict
        Mat data(1, 128, CV_32FC1, sampleMat);
        float value_svm = svmNew->predict(data);
        // ve va show ket qua.
        if (value_svm == 1)
        {
            return 1;
        }
        else
        {
            return 0;
        }
    }
return -2;
}


