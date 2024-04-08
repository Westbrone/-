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
std::tuple<bool, cv::Mat> getContours(const cv::Mat& src, cv::Rect& rect);//图像集

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
    //起始时间
    auto start_time = std::chrono::steady_clock::now();

    // Windows
    namedWindow(WIN_SRC, cv::WINDOW_AUTOSIZE);
    //namedWindow(WIN_RESULT, cv::WINDOW_AUTOSIZE);

    cv::Mat frame;                                 // 输入视频帧序列
    cv::Size winSize(600, 600);
    MoveHandle moveHandle(winSize);


    // 创建一个单独的窗口用于显示计时器
    cv::namedWindow("Timer", cv::WINDOW_NORMAL);
    cv::resizeWindow("Timer", 300, 100);
    cv::moveWindow("Timer", 1920, 0);
    auto duration = 0;
    auto startTime_main = std::chrono::steady_clock::now();

    while (true) // 显示窗口
    {
        captRefrnc >> frame;

        if (frame.empty()) {
            cout << " < < <  Game over!  > > > ";
            break;
        }

        // 中值滤波，去除噪声
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
        

        // 计算已经过去的时间
        
       

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
        // 在图像上显示计时信息
        // 在计时器窗口上显示计时信息
        // 计算小时和分钟
        int hours = duration / 3600;
        int minutes = (duration % 3600) / 60;
        int seconds = duration % 60;

        // 将时间拼接成字符串
        std::string timer_msg = "You have worked for:" + std::to_string(hours) + " hours, " + std::to_string(minutes) + " minutes, " + std::to_string(seconds) + " seconds";

        cv::Mat timer_image = cv::Mat::zeros(100, 900, CV_8UC3);
        cv::putText(timer_image, timer_msg, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 255), 2);
        cv::imshow("Timer", timer_image);
      


        // 显示图像
        imshow("Frame", frame);



        int c = cv::waitKey(1);
        if (c == 27)//esc退出
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
    cvtColor(src, hsv_image, cv::COLOR_BGR2HSV); //转换成YCrCb空间
    cv::Mat output_mask = cv::Mat::zeros(src.size(), CV_8UC1);
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            auto* p_mask = output_mask.ptr<uchar>(i, j);
            auto* p_src = hsv_image.ptr<uchar>(i, j);
            if (p_src[h] >= 0 && p_src[h] <= 20 && p_src[s] >= 48 && p_src[v] >= 50) {
                p_mask[0] = 255;//获得白色的肉色转换
            }
        }
    }
    cv::Mat detect;
    src.copyTo(detect, output_mask);//只复制识别到有肤色的皮肤原图吗，其他全部为0
    return output_mask;
}


/*基于椭圆皮肤模型的皮肤检测*/
cv::Mat skindetect_2(cv::Mat& src)
{
    cv::Mat img = src.clone();
    cv::Mat skinCrCbHist = cv::Mat::zeros(cv::Size(256, 256), CV_8UC1);
    //利用opencv自带的椭圆生成函数先生成一个肤色椭圆模型
    ellipse(skinCrCbHist, cv::Point(113, 155.6), cv::Size(23.4, 15.2), 43.0, 0.0, 360.0, cv::Scalar(255, 255, 255), -1);
    cv::Mat ycrcb_image;
    cv::Mat output_mask = cv::Mat::zeros(img.size(), CV_8UC1);
    cvtColor(img, ycrcb_image, CV_BGR2YCrCb); //首先转换成到YCrCb空间
    for (int i = 0; i < img.cols; i++)   //利用椭圆皮肤模型进行皮肤检测
        for (int j = 0; j < img.rows; j++)
        {
            cv::Vec3b ycrcb = ycrcb_image.at<cv::Vec3b>(j, i);
            if (skinCrCbHist.at<uchar>(ycrcb[1], ycrcb[2]) > 0)   //如果该落在皮肤模型椭圆区域内，该点就是皮肤像素点
                output_mask.at<uchar>(j, i) = 255;
        }

    cv::Mat detect;
    img.copyTo(detect, output_mask);  //返回肤色图
    return output_mask;
}



cv::Mat skindetect_3(cv::Mat& src)
{
    cv::Mat ycrcb_image, detect;
    cv::cvtColor(src, ycrcb_image, CV_BGR2YCrCb); //首先将RGB转换成到YCrCb空间

    std::vector<cv::Mat> channels;
    split(ycrcb_image, channels);  //通道分离
    cv::Mat output_mask = channels[1];  //Cr分量
    cv::threshold(output_mask, output_mask, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);   //OSTU阈值分割
    src.copyTo(detect, output_mask);
    return output_mask;
}

bool preDeal(cv::Mat& dst, cv::Mat& frame, cv::Size& winSize, MoveHandle& moveHandle, int i)
{
    //cv::imwrite("皮肤检测后", dst);

    cv::Mat img = frame.clone();
    cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    erode(dst, dst, element);//腐蚀，平滑边界
    morphologyEx(dst, dst, cv::MORPH_OPEN, element);//开运算，先执行腐蚀操作，再进行膨胀操作。
    dilate(dst, dst, element);//膨胀，连续区域
    morphologyEx(dst, dst, cv::MORPH_CLOSE, element);//闭运算，先执行膨胀操作，再进行腐蚀操作。

    cv::Rect rect;
    //imshow("轮廓矩形前", dst);
    bool empty;
    std::tie(empty,dst) = getContours(dst, rect);//在轮廓中描述一个整体并矩形框住
    //imshow("轮廓矩形后", dst);

    //如果没有框就选中不了
    cv::Point center(rect.x + rect.width / 2, rect.y + rect.height / 2);
    if (center.x != 0 && center.y != 0) {//判断中心点
        cv::circle(img, center, 4, cv::Scalar(0, 0, 255), 2);
        moveHandle.addPoint(center);
        Gesture g = moveHandle.analyseLocus();
        HotKey ht;
        ht.postKey(g);
    }//画出矩形框
    line(img, cv::Point(winSize.width / 3, winSize.height / 3), cv::Point(winSize.width * 2 / 3, winSize.height / 3), cv::Scalar(255, 255, 255), 3);
    line(img, cv::Point(winSize.width / 3, winSize.height * 2 / 3), cv::Point(winSize.width * 2 / 3, winSize.height * 2 / 3), cv::Scalar(255, 255, 255),
        3);
    line(img, cv::Point(winSize.width / 3, winSize.height / 3), cv::Point(winSize.width / 3, winSize.height * 2 / 3), cv::Scalar(255, 255, 255), 3);
    line(img, cv::Point(winSize.width * 2 / 3, winSize.height / 3), cv::Point(winSize.width * 2 / 3, winSize.height * 2 / 3), cv::Scalar(255, 255, 255),
        3);
    if (i == 1)
    {
        imshow("基于RGB的皮肤检测原图", img);
        imshow("基于RGB的皮肤检测mask", dst);
    }
    else if (i==2)
    {
        imshow("基于椭圆皮肤模型的皮肤检测原图", img);
        imshow("基于椭圆皮肤模型的皮肤检测mask", dst);
    }
    else
    {
        imshow("YCrCb颜色空间Cr分量+Otsu法阈值分割的皮肤检测原图", img);
        imshow("YCrCb颜色空间Cr分量+Otsu法阈值分割的皮肤检测mask", dst);
    }

    
    dst.release();
    

    return empty; 


}



std::tuple<bool, cv::Mat> getContours(const cv::Mat& src, cv::Rect& rect) {
    if (src.channels() != 1)
        throw "FindTargets : 通道数必须为 1";
    cv::Mat dst = src.clone();
    vector<vector<cv::Point>> contours;       // 轮廓
    contours.clear();

    // 得到手的轮廓
    findContours(src, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 去除伪轮廓
    double MaxArea = 0;
    vector<vector<cv::Point>> MaxContour;
    MaxContour.clear();
    for (const auto& contour : contours) {
        double area = contourArea(contour);
        //判断手进入区域的阈值
        if (area > 10000 && area > MaxArea) {
            MaxArea = area;
            MaxContour.push_back(contour);
        }
    }
    // 画轮廓
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
