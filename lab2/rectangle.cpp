/*
 * basic.cpp
 *      Author: NRR
 */

#include <opencv2/core/core.hpp>
#include <opencv2/core/types.hpp>
#include "opencv2/core/cvdef.h"
#include "opencv2/core/fast_math.hpp"
#include <opencv2/highgui/highgui.hpp>
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

    // obtener rectangulo para modificar
    Range rows(100, 301);
    Range cols(100, 601);
    Mat&& rectangle = image(rows, cols);

    typedef cv::Point3_<uint8_t> Pixel;
    for (Pixel &p : Mat_<Pixel>(rectangle))
    {
        p.x = saturate_cast<uint8_t>(255-p.x);
        p.y = saturate_cast<uint8_t>(255-p.y);
        p.z = saturate_cast<uint8_t>(255-p.z);
    }

    // display images
    namedWindow("original", WINDOW_AUTOSIZE);
    imshow("original", image);
    namedWindow("rect", WINDOW_AUTOSIZE);
    imshow("rect", rectangle);

    waitKey(0); // Wait for a keystroke in the window
    return 0;
}