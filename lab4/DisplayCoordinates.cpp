/*
 * DisplayImage.cpp
 *
 *      Author: NRR
 * 
 * How to get the coordinates of a point on an image
 * through a mouse click
 */

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/types.hpp>
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;

static void onMouse(int event, int x, int y, int, void *param)
{
    Mat *img = (Mat *)param;
    Mat imgToDraw = (*img).clone();
    if (event == EVENT_LBUTTONDOWN)
    {
        Point clicked(x, y);
        // show coordinates of clicked point in console
        cout << "You've clicked on point: [" << clicked.x << ", "
             << clicked.y << "]" << endl;
        // show coordinates on the image
        string coord = "x: " + to_string(x) + ", y: " + to_string(y);
        putText(imgToDraw, coord, clicked, FONT_HERSHEY_SIMPLEX, 1,
                Scalar(0, 0, 255), 2);
        imshow("Display window", imgToDraw);
    }
}

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

    namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
    imshow("Display window", image);                // Show our image inside it.

    setMouseCallback("Display window", onMouse, (void *)&image);

    waitKey(0); // Wait for a keystroke in the window
    return 0;
}