/*
 * lab3.cpp
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

    Mat smoothedMean;
    blur(image, smoothedMean, Size(4, 4));
    Mat smoothedMedian;
    medianBlur(image, smoothedMedian, 5);
    Mat smoothedGaussian;
    GaussianBlur(image, smoothedGaussian, Size(5, 5), 2.3);
    Mat smoothedBilateral;
    bilateralFilter(image, smoothedBilateral, 5, 180, 15);

    Mat binarized;
    threshold(image, binarized, 120, 255, THRESH_BINARY);
    Mat truncated;
    threshold(image, truncated, 120, 255, THRESH_TRUNC);

    Mat edges;
    Canny(image, edges, 180, 120, 3);

    // display images
    namedWindow("original", WINDOW_AUTOSIZE);
    imshow("original", image);
    namedWindow("Suavizado media", WINDOW_AUTOSIZE);
    imshow("Suavizado media", smoothedMean);
    namedWindow("Suavizado mediana", WINDOW_AUTOSIZE);
    imshow("Suavizado mediana", smoothedMedian);
    namedWindow("Suavizado gaussiano", WINDOW_AUTOSIZE);
    imshow("Suavizado gaussiano", smoothedGaussian);
    namedWindow("Suavizado bilateral", WINDOW_AUTOSIZE);
    imshow("Suavizado bilateral", smoothedBilateral);

    namedWindow("Binarizada", WINDOW_AUTOSIZE);
    imshow("Binarizada", binarized);
    namedWindow("Truncada", WINDOW_AUTOSIZE);
    imshow("Truncada", truncated);

    namedWindow("Deteccion de bordes con Canny", WINDOW_AUTOSIZE);
    imshow("Deteccion de bordes con Canny", edges);

    waitKey(0); // Wait for a keystroke in the window
    return 0;
}