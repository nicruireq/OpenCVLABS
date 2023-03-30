/*
 * paintMap.cpp
 *      Author: NRR
 */

#include <opencv2/core/core.hpp>
#include <opencv2/core/types.hpp>
#include "opencv2/core/cvdef.h"
#include "opencv2/core/fast_math.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

// Set true to get coordinates by clicking 4 times on the map image
// Set to false to skip the selection
#define GET_COORDINATES false
#define NUM_REGIONS 4

using namespace cv;
using namespace std;

struct ClickData
{
    Mat &image;
    vector<Point> points;
    int index; // index 0: spain, 1: portugal, 2: germany, 3: sea
    ClickData(Mat &image, const vector<Point> &points)
        : image(image), points(points), index(0){};
};

// Callback function to get coordinates from mouse click
static void onMouse(int event, int x, int y, int, void *param)
{
    ClickData *data = (ClickData *)param;
    Mat imgToDraw = data->image.clone();

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
        imshow("original", imgToDraw);
        // pass coordinates to the caller
        data->points[data->index] = clicked;
        cout << "INDEX = " << data->index << endl;
        // circular index increment
        data->index = (data->index) >= NUM_REGIONS ? 0 : ++data->index;
    }
}

int main(int argc, char **argv)
{
    Mat image;
    image = imread("MapaPintado.png", IMREAD_COLOR); // Read the file

    if (!image.data)
    {
        cerr << "No se ha podido leer MapaPintado.png. "
             << "Por favor, situela en el mismo directorio desde donde llama "
             << "al ejecutable." << endl;
        return -1;
    }

    // display original map
    namedWindow("original", WINDOW_AUTOSIZE);
    imshow("original", image);

// get coordinates
#if GET_COORDINATES == true
    ClickData *data = new ClickData(image, vector<Point>(NUM_REGIONS));
    setMouseCallback("original", onMouse, (void *)data);
    waitKey(0); // Wait for a keystroke in the window
    Point spain = data->points[0];
    Point portugal = data->points[1];
    Point germany = data->points[2];
    Point sea = data->points[3];

    for (auto &&i : data->points)
    {
        cout << "point: " << i.x << ", " << i.y << endl;
    }
#endif

    // clean lines
    Mat image2;
    dilate(image, image2, Mat(), Point(-1, -1), 1);
    Mat imgCleaned;
    erode(image2, imgCleaned, Mat(), Point(-1, -1), 1);

// draw regions
#if GET_COORDINATES == false
    Point spain(52, 273);
    Point portugal(22, 265);
    Point germany(160, 192);
    Point sea(68, 70);
#endif
    Scalar loThreshold(20, 20, 20);
    Scalar hiThreshold(20, 20, 20);
    Scalar red(0, 0, 255), green(0, 255, 0), blue(255, 0, 0), yellow(0, 255, 255);

    Mat filledImg = imgCleaned.clone();
    floodFill(filledImg, spain, red, 0, loThreshold, hiThreshold);      // spain
    floodFill(filledImg, portugal, green, 0, loThreshold, hiThreshold); // portugal
    floodFill(filledImg, germany, yellow, 0, loThreshold, hiThreshold); // germany
    floodFill(filledImg, sea, blue, 0, loThreshold, hiThreshold);       // sea

    // display map painted
    namedWindow("Painted", WINDOW_AUTOSIZE);
    imshow("Painted", filledImg);

    waitKey(0); // Wait for a keystroke in the window

#if GET_COORDINATES == true
    if (data)
        delete data;
#endif
    return 0;
}
