#include <iomanip>  // for controlling float print precision
#include <iostream>
#include <sstream>
#include <string>
#include <windows.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp> // OpenCV window I/O
#include <opencv2/imgproc/imgproc.hpp> // Gaussian Blur
#include "MoveHandle.h"
#include "HotKey.h"
#include<thread>
#include <chrono>

using namespace std;
std::tuple<bool, cv::Mat> getContours(const cv::Mat& src, cv::Rect& rect);//ͼ��

cv::Mat skindetect_1(cv::Mat& src);
cv::Mat skindetect_2(cv::Mat& src);
cv::Mat skindetect_3(cv::Mat& src);

bool preDeal(cv::Mat& dst, cv::Mat& frame, cv::Size& winSize, MoveHandle& moveHandle, int i);



int main(int argc, char* argv[]) {

    cv::VideoCapture captRefrnc(0);
    if (!captRefrnc.isOpened()) {
        return -1;
    }
    const char* WIN_SRC = "Source";
    const char* WIN_RESULT = "Result";
    //��ʼʱ��
    auto start_time = std::chrono::steady_clock::now();

    // Windows
    namedWindow(WIN_SRC, cv::WINDOW_AUTOSIZE);
    //namedWindow(WIN_RESULT, cv::WINDOW_AUTOSIZE);

    cv::Mat frame;                                 // ������Ƶ֡����
    cv::Size winSize(600, 600);
    MoveHandle moveHandle(winSize);


    // ����һ�������Ĵ���������ʾ��ʱ��
    cv::namedWindow("Timer", cv::WINDOW_NORMAL);
    cv::resizeWindow("Timer", 300, 100);
    cv::moveWindow("Timer", 1920, 0);
    auto duration = 0;
    auto startTime_main = std::chrono::steady_clock::now();

    while (true) // ��ʾ����
    {
        captRefrnc >> frame;

        if (frame.empty()) {
            cout << " < < <  Game over!  > > > ";
            break;
        }

        // ��ֵ�˲���ȥ������
        medianBlur(frame, frame, 5);
        /////////////
        //cv::Mat dst_1 = skindetect_1(frame);
        //cv::Mat dst_2 = skindetect_2(frame);
        cv::Mat dst_1, dst_2, dst_3;
        std::thread thread1([&]() {
            dst_1 = skindetect_1(frame);
            
            });
        std::thread thread2([&]() {
            dst_2 = skindetect_2(frame);
            });
        std::thread thread3([&]() {
            dst_3 = skindetect_3(frame);
            });
        thread1.join();
        thread2.join();
        thread3.join();
        

        // �����Ѿ���ȥ��ʱ��
        
       

        bool empty1 = preDeal(dst_1, frame, winSize, moveHandle, 1);
        bool empty2 = preDeal(dst_2, frame, winSize, moveHandle, 2);
        bool empty3 = preDeal(dst_3, frame, winSize, moveHandle, 3);


        auto end_time_main = std::chrono::steady_clock::now();
        auto elapsedTime = end_time_main - startTime_main;
        auto secondsElapsed = std::chrono::duration_cast<std::chrono::seconds>(elapsedTime).count();

        std::cout <<empty1<<empty2<<empty3 << std::endl;
        
        if (secondsElapsed % 3 == 0) {
            if (empty1 || empty2 )
            {
                auto end_time_1 = std::chrono::steady_clock::now();
                duration += std::chrono::duration_cast<std::chrono::seconds>(end_time_1 - start_time).count();
                start_time = end_time_1;
            }
            else
            {
                auto end_time_1 = std::chrono::steady_clock::now();
                duration += 0;
                start_time = end_time_1;
            }
        }

      
        std::cout << duration << std::endl;
        // ��ͼ������ʾ��ʱ��Ϣ
        // �ڼ�ʱ����������ʾ��ʱ��Ϣ
        // ����Сʱ�ͷ���
        int hours = duration / 3600;
        int minutes = (duration % 3600) / 60;
        int seconds = duration % 60;

        // ��ʱ��ƴ�ӳ��ַ���
        std::string timer_msg = "You have worked for:" + std::to_string(hours) + " hours, " + std::to_string(minutes) + " minutes, " + std::to_string(seconds) + " seconds";

        cv::Mat timer_image = cv::Mat::zeros(100, 900, CV_8UC3);
        cv::putText(timer_image, timer_msg, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 255), 2);
        cv::imshow("Timer", timer_image);
      


        // ��ʾͼ��
        imshow("Frame", frame);



        int c = cv::waitKey(1);
        if (c == 27)//esc�˳�
            break;
        else if (c == 'q') {
            moveHandle.reset();
            c = -1;
        }
    }
}

cv::Mat skindetect_1(cv::Mat& src) {
    cv::Mat hsv_image;
    int h = 0;
    int s = 1;
    int v = 2;
    cvtColor(src, hsv_image, cv::COLOR_BGR2HSV); //ת����YCrCb�ռ�
    cv::Mat output_mask = cv::Mat::zeros(src.size(), CV_8UC1);
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            auto* p_mask = output_mask.ptr<uchar>(i, j);
            auto* p_src = hsv_image.ptr<uchar>(i, j);
            if (p_src[h] >= 0 && p_src[h] <= 20 && p_src[s] >= 48 && p_src[v] >= 50) {
                p_mask[0] = 255;//��ð�ɫ����ɫת��
            }
        }
    }
    cv::Mat detect;
    src.copyTo(detect, output_mask);//ֻ����ʶ���з�ɫ��Ƥ��ԭͼ������ȫ��Ϊ0
    return output_mask;
}


/*������ԲƤ��ģ�͵�Ƥ�����*/
cv::Mat skindetect_2(cv::Mat& src)
{
    cv::Mat img = src.clone();
    cv::Mat skinCrCbHist = cv::Mat::zeros(cv::Size(256, 256), CV_8UC1);
    //����opencv�Դ�����Բ���ɺ���������һ����ɫ��Բģ��
    ellipse(skinCrCbHist, cv::Point(113, 155.6), cv::Size(23.4, 15.2), 43.0, 0.0, 360.0, cv::Scalar(255, 255, 255), -1);
    cv::Mat ycrcb_image;
    cv::Mat output_mask = cv::Mat::zeros(img.size(), CV_8UC1);
    cvtColor(img, ycrcb_image, CV_BGR2YCrCb); //����ת���ɵ�YCrCb�ռ�
    for (int i = 0; i < img.cols; i++)   //������ԲƤ��ģ�ͽ���Ƥ�����
        for (int j = 0; j < img.rows; j++)
        {
            cv::Vec3b ycrcb = ycrcb_image.at<cv::Vec3b>(j, i);
            if (skinCrCbHist.at<uchar>(ycrcb[1], ycrcb[2]) > 0)   //���������Ƥ��ģ����Բ�����ڣ��õ����Ƥ�����ص�
                output_mask.at<uchar>(j, i) = 255;
        }

    cv::Mat detect;
    img.copyTo(detect, output_mask);  //���ط�ɫͼ
    return output_mask;
}



cv::Mat skindetect_3(cv::Mat& src)
{
    cv::Mat ycrcb_image, detect;
    cv::cvtColor(src, ycrcb_image, CV_BGR2YCrCb); //���Ƚ�RGBת���ɵ�YCrCb�ռ�

    std::vector<cv::Mat> channels;
    split(ycrcb_image, channels);  //ͨ������
    cv::Mat output_mask = channels[1];  //Cr����
    cv::threshold(output_mask, output_mask, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);   //OSTU��ֵ�ָ�
    src.copyTo(detect, output_mask);
    return output_mask;
}

bool preDeal(cv::Mat& dst, cv::Mat& frame, cv::Size& winSize, MoveHandle& moveHandle, int i)
{
    //cv::imwrite("Ƥ������", dst);

    cv::Mat img = frame.clone();
    cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    erode(dst, dst, element);//��ʴ��ƽ���߽�
    morphologyEx(dst, dst, cv::MORPH_OPEN, element);//�����㣬��ִ�и�ʴ�������ٽ������Ͳ�����
    dilate(dst, dst, element);//���ͣ���������
    morphologyEx(dst, dst, cv::MORPH_CLOSE, element);//�����㣬��ִ�����Ͳ������ٽ��и�ʴ������

    cv::Rect rect;
    //imshow("��������ǰ", dst);
    bool empty;
    std::tie(empty,dst) = getContours(dst, rect);//������������һ�����岢���ο�ס
    //imshow("�������κ�", dst);

    //���û�п��ѡ�в���
    cv::Point center(rect.x + rect.width / 2, rect.y + rect.height / 2);
    if (center.x != 0 && center.y != 0) {//�ж����ĵ�
        cv::circle(img, center, 4, cv::Scalar(0, 0, 255), 2);
        moveHandle.addPoint(center);
        Gesture g = moveHandle.analyseLocus();
        HotKey ht;
        ht.postKey(g);
    }//�������ο�
    line(img, cv::Point(winSize.width / 3, winSize.height / 3), cv::Point(winSize.width * 2 / 3, winSize.height / 3), cv::Scalar(255, 255, 255), 3);
    line(img, cv::Point(winSize.width / 3, winSize.height * 2 / 3), cv::Point(winSize.width * 2 / 3, winSize.height * 2 / 3), cv::Scalar(255, 255, 255),
        3);
    line(img, cv::Point(winSize.width / 3, winSize.height / 3), cv::Point(winSize.width / 3, winSize.height * 2 / 3), cv::Scalar(255, 255, 255), 3);
    line(img, cv::Point(winSize.width * 2 / 3, winSize.height / 3), cv::Point(winSize.width * 2 / 3, winSize.height * 2 / 3), cv::Scalar(255, 255, 255),
        3);
    if (i == 1)
    {
        imshow("����RGB��Ƥ�����ԭͼ", img);
        imshow("����RGB��Ƥ�����mask", dst);
    }
    else if (i==2)
    {
        imshow("������ԲƤ��ģ�͵�Ƥ�����ԭͼ", img);
        imshow("������ԲƤ��ģ�͵�Ƥ�����mask", dst);
    }
    else
    {
        imshow("YCrCb��ɫ�ռ�Cr����+Otsu����ֵ�ָ��Ƥ�����ԭͼ", img);
        imshow("YCrCb��ɫ�ռ�Cr����+Otsu����ֵ�ָ��Ƥ�����mask", dst);
    }

    
    dst.release();
    

    return empty; 


}



std::tuple<bool, cv::Mat> getContours(const cv::Mat& src, cv::Rect& rect) {
    if (src.channels() != 1)
        throw "FindTargets : ͨ��������Ϊ 1";
    cv::Mat dst = src.clone();
    vector<vector<cv::Point>> contours;       // ����
    contours.clear();

    // �õ��ֵ�����
    findContours(src, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // ȥ��α����
    double MaxArea = 0;
    vector<vector<cv::Point>> MaxContour;
    MaxContour.clear();
    for (const auto& contour : contours) {
        double area = contourArea(contour);
        //�ж��ֽ����������ֵ
        if (area > 10000 && area > MaxArea) {
            MaxArea = area;
            MaxContour.push_back(contour);
        }
    }
    // ������
    if (MaxContour.empty()) {
        return  std::make_tuple(false, dst);
    }
    else {
        drawContours(dst, MaxContour, -1, cv::Scalar(0, 0, 255), 3);
        rect = boundingRect(MaxContour.back());
        rectangle(dst, rect, cv::Scalar(255, 255, 255), 3);
    }
    return  std::make_tuple(true, dst);
}
