/*
 * Process frames from video
 *
 *      Author: NRR
 */

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>

using namespace cv;
using namespace std;

enum
{
    NORMAL,
    GRAY,
    BINARIZE
};

void processFrame(VideoCapture &vsrc, Mat &dst, char mode);

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        cout << " Usage: opencv_basico VideoToLoadAndDisplay" << endl;
        return -1;
    }

    VideoCapture videoSource;

    videoSource.open(argv[1]);
    if (!videoSource.isOpened())
    {
        cout << "Could not open or find the video" << std::endl;
        return -1;
    }

    namedWindow("Original", WINDOW_KEEPRATIO); // resizable window;

    Mat frame;
    char key = '3';    // to set default mode NORMAL
    int mode = NORMAL; // default mode
    for (;;)
    {
        processFrame(videoSource, frame, mode);
        if (frame.empty())
            break;
        imshow("Original", frame);
        key = (char)waitKey(5);
        switch (key)
        {
        case 'q':
        case 'Q':
        case 27: // escape key
            return 0;
        case '1':
            mode = GRAY;
            break;
        case '2':
            mode = BINARIZE;
            break;
        case '3':
            mode = NORMAL;
            break;
        default:
            break;
        }
    }

    waitKey(0); // Wait for a keystroke in the window
    return 0;
}

void processFrame(VideoCapture &videoSource, Mat &dst, char mode)
{
    Mat frame;
    videoSource >> frame;
    switch (mode)
    {
    case GRAY:
        cvtColor(frame, dst, COLOR_BGR2GRAY);
        break;
    case BINARIZE:
        cvtColor(frame, dst, COLOR_BGR2GRAY);
        threshold(dst, dst, 100, 255, THRESH_BINARY);
        break;
    case NORMAL:
    default:
        dst = frame.clone();
        break;
    }
}