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
    dilate(image, image2, Mat(), Point(-1, -1), 1);
    Mat imgCleaned;
    erode(image2, imgCleaned, Mat(), Point(-1, -1), 1);

    // DRAW REGIONS
    Mat imgCleanedBw;
    threshold(imgCleaned, imgCleanedBw, 120, 255, THRESH_BINARY);
    Mat labelImage(imgCleanedBw.size(), CV_32S);
    int nLabels = connectedComponents(imgCleanedBw, labelImage, 8);

    cout << "labelImage size: " << labelImage.size().height << " x " << labelImage.size().width << endl;
    cout << "Labels: " << nLabels << endl;

    // para ver componentes conexas
    std::vector<Vec3b> colors(nLabels);
    colors[0] = Vec3b(0, 0, 0); // background
    for (int label = 1; label < nLabels; ++label)
    {
        colors[label] = Vec3b((rand() & 255), (rand() & 255), (rand() & 255));
    }
    Mat dst(imgCleanedBw.size(), CV_8UC3);
    for (int r = 0; r < dst.rows; ++r)
    {
        for (int c = 0; c < dst.cols; ++c)
        {
            int label = labelImage.at<int>(r, c);
            Vec3b &pixel = dst.at<Vec3b>(r, c);
            pixel = colors[label];
        }
    }
    // fin codigo preparacion mostrar componentes conexas

    // display images
    namedWindow("original", WINDOW_AUTOSIZE);
    imshow("original", image);
    namedWindow("Cleaned binarized", WINDOW_AUTOSIZE);
    imshow("Cleaned binarized", imgCleanedBw);
    namedWindow("Closing", WINDOW_AUTOSIZE);
    imshow("Closing", dst);


    waitKey(0); // Wait for a keystroke in the window
    return 0;
}