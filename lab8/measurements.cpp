/*
 * measurements.cpp
 *      Author: NRR
 */

#include <opencv2/core/core.hpp>
#include <opencv2/core/types.hpp>
#include "opencv2/core/cvdef.h"
#include "opencv2/core/fast_math.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;
using namespace std::chrono;

void filter(Mat &orig, Mat &res)
{
    int HEIGHT = orig.size().height;
    int WIDTH = orig.size().width;
    int channels = orig.channels();
    int thresh = 30; //* 256 - 1;  // for 16 bit image

    for (int row = 0; row < HEIGHT; row++)
    {
        for (int col = 0; col < WIDTH; col++)
        {
            // Vec3w is Vec<ushort, 3>
            Vec3b p = orig.at<Vec3b>(row, col);
            Vec3b &result = res.at<Vec3b>(row, col);
            // result[0] = p[0];
            //     result[1] = p[1];
            //     result[2] = p[2];

            if (p[0] < thresh && p[1] < thresh && p[2] < thresh)
            {
                result[0] = 0;
                result[1] = 255; //*256-1; // for 16 bit image
                result[2] = 0;
            }
        } // col loop
    }     // row loop
}

void filter2(Mat &orig, Mat &res)
{
    Mat ksobel(3, 3, CV_32F);
    ksobel.at<float>(0, 0) = -1.0;
    ksobel.at<float>(0, 1) = 0.0;
    ksobel.at<float>(0, 2) = 1.0;
    ksobel.at<float>(1, 0) = -2.0;
    ksobel.at<float>(1, 1) = 0.0;
    ksobel.at<float>(1, 2) = 2.0;
    ksobel.at<float>(2, 0) = -1.0;
    ksobel.at<float>(2, 1) = 0.0;
    ksobel.at<float>(2, 2) = 1.0;
    // cout << "kernel = " << endl
    //      << ksobel << endl;
    filter2D(orig, res, -1, ksobel);
}

void filter3(Mat &orig, Mat &res)
{
    Mat temp(orig.size(), orig.type());
    GaussianBlur(orig, temp, Size(3, 3), 1);
    filter2(temp, res);
}

int main(int argc, char const *argv[])
{
    Mat circles = imread("images/circles_320x240_888.png", IMREAD_ANYCOLOR);
    int type = circles.type();
    cout << "Image type: " << type << endl;
    cout << "Types: " << CV_8S << " " << CV_8U << " " << CV_8UC3 << endl;

    Mat result(circles.size(), type);
    cout << "Size result: " << circles.size().height << " x " << circles.size().width << endl;
    cout << "Size result: " << result.size().height << " x " << result.size().width << endl;
    cout << "type result: " << result.type() << endl;

    // first
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    filter(circles, result);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span1 = duration_cast<duration<double>>(t2 - t1);
    std::cout << "Filter 1 takes: " << time_span1.count() * 1000 << " milliseconds." << endl;
    // end first

    Mat lena = imread("images/lena_320x240_565.png", IMREAD_ANYCOLOR);
    Mat result2(lena.size(), lena.type());

    // second
    high_resolution_clock::time_point t3 = high_resolution_clock::now();
    filter2(lena, result2);
    high_resolution_clock::time_point t4 = high_resolution_clock::now();
    duration<double> time_span2 = duration_cast<duration<double>>(t4 - t3);
    std::cout << "Filter 2 takes: " << time_span2.count() * 1000 << " milliseconds." << endl;
    // end second

    Mat result3(lena.size(), lena.type());
    // third
    high_resolution_clock::time_point t5 = high_resolution_clock::now();
    filter3(lena, result3);
    high_resolution_clock::time_point t6 = high_resolution_clock::now();
    duration<double> time_span3 = duration_cast<duration<double>>(t6 - t5);
    std::cout << "Filter 3 takes: " << time_span3.count() * 1000 << " milliseconds." << endl;
    // end third

    namedWindow("Original", WINDOW_AUTOSIZE); // Create a window for display.
    imshow("Original", circles);              // Show our image inside it.

    namedWindow("thresh", WINDOW_AUTOSIZE); // Create a window for display.
    imshow("thresh", result);               // Show our image inside it.

    namedWindow("lena", WINDOW_AUTOSIZE); // Create a window for display.
    imshow("lena", lena);                 // Show our image inside it.

    namedWindow("lena sobel x", WINDOW_AUTOSIZE); // Create a window for display.
    imshow("lena sobel x", result2);              // Show our image inside it.

    namedWindow("lena filter 3", WINDOW_AUTOSIZE); // Create a window for display.
    imshow("lena filter 3", result2);              // Show our image inside it.

    waitKey(0); // Wait for a keystroke in the window
    return 0;
}
