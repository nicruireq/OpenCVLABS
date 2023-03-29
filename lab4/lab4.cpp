/*
 * lab4.cpp
 *      Author: NRR
 */

#include <opencv2/core/core.hpp>
#include <opencv2/core/types.hpp>
#include "opencv2/core/cvdef.h"
#include "opencv2/core/fast_math.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        cout << " Usage: display_image ImageToLoadAndDisplay" << endl;
        return -1;
    }

    Mat image;
    image = imread(argv[1], IMREAD_COLOR); // Read the file

    if (!image.data) // Check for invalid input
    {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    // clean lines
    Mat image2;
    dilate(image, image2, Mat(), Point(-1, -1), 1);
    Mat imgCleaned;
    erode(image2, imgCleaned, Mat(), Point(-1, -1), 1);

    // draw regions
    //Mat imgCleanedBw;
    //threshold(imgCleaned, imgCleanedBw, 120, 255, THRESH_BINARY);
    Point spain(52, 273);
    Point portugal(22, 265);
    Point germany(160, 192);
    Point sea(68, 70);
    Scalar loThreshold(20, 20, 20);
    Scalar hiThreshold(20, 20, 20);
    Scalar red(0, 0, 255), green(0,255,0), blue(255,0,0), yellow(0,255,255);

    Mat filledImg = imgCleaned.clone();
    floodFill(filledImg, spain, red, 0, loThreshold, hiThreshold); // spain
    floodFill(filledImg, portugal, green, 0, loThreshold, hiThreshold); // portugal
    floodFill(filledImg, germany, yellow, 0, loThreshold, hiThreshold); // germany
    floodFill(filledImg, sea, blue, 0, loThreshold, hiThreshold); // sea

    // display images
    namedWindow("original", WINDOW_AUTOSIZE);
    imshow("original", image);
    namedWindow("Cleaned", WINDOW_AUTOSIZE);
    imshow("Cleaned", filledImg);

    waitKey(0); // Wait for a keystroke in the window
    return 0;
}