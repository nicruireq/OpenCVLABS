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
    image = imread(argv[1], IMREAD_GRAYSCALE); // Read the file

    if (!image.data) // Check for invalid input
    {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    Mat image2;
    dilate(image, image2, Mat(), Point(-1,-1), 1);
    Mat imgCleaned;
    erode(image2, imgCleaned, Mat(), Point(-1,-1),1);

    // DRAW REGIONS

    // display images
    namedWindow("original", WINDOW_AUTOSIZE);
    imshow("original", image);
    namedWindow("Closing", WINDOW_AUTOSIZE);
    imshow("Closing", imgCleaned);

    waitKey(0); // Wait for a keystroke in the window
    return 0;
}