/*
 * drawSomeShapes.cpp
 *      Author: NRR
 */

#include <opencv2/core/core.hpp>
#include <opencv2/core/types.hpp>
#include "opencv2/core/cvdef.h"
#include "opencv2/core/fast_math.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <random>
#include <cstdlib>

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
    Mat image;
    string defaultImageName("../pictures/grogu1.jpg");
    switch (argc)
    {
    case 1:
        image = imread(defaultImageName, IMREAD_COLOR);
        break;
    case 2:
        image = imread(argv[1], IMREAD_COLOR);
        break;
    default:
        cout << " Usage: display_image ImageToLoadAndDisplay" << endl;
        return -1;
    }

    if (!image.data)
    {
        cerr << "Could not open or find the image" << std::endl;
        return -2;
    }

    int h = image.size().height;
    int w = image.size().width;
    int thickness = 4;
    // paint the shapes in random points
    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    minstd_rand0 generator(seed1);
    // default_random_engine generator;
    uniform_int_distribution<int> randomRow(0, h - 1);
    uniform_int_distribution<int> randomColum(0, w - 1);
    uniform_int_distribution<int> randomColor(0, 255);

    Mat shapes = image.clone();

    // Draw line
    Point pl1, pl2;
    do
    {
        pl1 = Point(randomColum(generator) + thickness, randomRow(generator) + thickness);
        pl2 = Point(randomColum(generator) + thickness, randomRow(generator) + thickness);
    } while (pl1.dot(pl2) < 0.25 * min(h, w));

    line(shapes, pl1, pl2,
         Scalar(randomColor(generator), randomColor(generator), randomColor(generator)),
         thickness);

    // Draw rectangle
    Point topleft, bottomright;
    do
    {
        topleft = Point(randomColum(generator) + thickness, randomRow(generator) + thickness);
        bottomright = Point(randomColum(generator) + thickness, randomRow(generator) + thickness);

    } while ((abs(topleft.x - bottomright.x) < 0.25 * h) && (abs(topleft.y - bottomright.y) < 0.25 * w) && (topleft.dot(bottomright) < 0.25 * min(h, w)));

    cout << "topleft: " << topleft.x << ", " << topleft.y << "\nbottomright: " << bottomright.x << ", " << bottomright.y << endl;
    rectangle(shapes, topleft, bottomright,
              Scalar(randomColor(generator), randomColor(generator), randomColor(generator)),
              thickness);

    // Draw circle
    Point center;
    uniform_int_distribution<int> randomRadius(0.15 * min(h, w), 0.3 * min(h, w));
    int radius;
    do
    {
        radius = randomRadius(generator);
        center = Point(randomColum(generator) + thickness, randomRow(generator) + thickness);
    } while ((((center.x + radius) > (w - 0.3 * w)) || ((center.x + radius) < 0.3 * w)) && (((center.y + radius) > (h - 0.3 * h)) || ((center.y + radius) < 0.3 * h)));

    circle(shapes, center, radius, Scalar(randomColor(generator), randomColor(generator), randomColor(generator)), thickness);

    // Draw ellipse

    Size axes;
    do
    {
        axes = Size(randomRadius(generator) / 2, randomRadius(generator) / 2);
        center = Point(randomColum(generator) + thickness, randomRow(generator) + thickness);

    } while ((((center.x + axes.width) > (w - 0.3 * w)) || ((center.x + axes.width) < 0.3 * w)) && (((center.y + axes.height) > (h - 0.3 * h)) || ((center.y + axes.height) < 0.3 * h)));

    ellipse(shapes, center, axes, 0, 0, 360, Scalar(randomColor(generator), randomColor(generator), randomColor(generator)), thickness);

    // Draw text
    putText(shapes, "May the Force be with you", Point(10,40), FONT_HERSHEY_SIMPLEX, 1.1, Scalar(randomColor(generator), randomColor(generator), randomColor(generator)), thickness);

    cout << "rows: " << image.size().height << "\ncols: " << image.size().width << endl;

    // display images
    namedWindow("original", WINDOW_AUTOSIZE);
    imshow("original", image);
    namedWindow("Shapes", WINDOW_AUTOSIZE);
    imshow("Shapes", shapes);

    waitKey(0); // Wait for a keystroke in the window

    return 0;
}
