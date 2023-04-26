/*
 * Process frames from video
 *
 *      Author: NRR
 */

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/videoio.hpp>
#include "opencv2/video/tracking.hpp"
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <exception>
#include <thread>
#include <condition_variable>
#include <mutex>

using namespace cv;
using namespace std;

// Synchronized buffer to pass
// processed frames between threads
class FrameBuffer
{
private:
    Mat *buffer;
    int capacity, front, rear, count;

    std::mutex lock;

    std::condition_variable not_full;
    std::condition_variable not_empty;

public:
    FrameBuffer(int capacity) : capacity(capacity), front(0), rear(0), count(0)
    {
        buffer = new Mat[capacity];
    }

    ~FrameBuffer()
    {
        delete[] buffer;
    }

    void deposit(Mat &data)
    {
        std::unique_lock<std::mutex> l(lock);

        not_full.wait(l, [this]()
                      { return count != capacity; });

        buffer[rear] = data.clone();
        rear = (rear + 1) % capacity;
        ++count;

        not_empty.notify_one();
    }

    Mat &fetch()
    {
        std::unique_lock<std::mutex> l(lock);

        not_empty.wait(l, [this]()
                       { return count != 0; });

        Mat &result = buffer[front];
        front = (front + 1) % capacity;
        --count;

        not_full.notify_one();

        return result;
    }
};

enum Action
{
    NO_ACTION,
    SHOW_MESSAGE,
    BLUR_ALL
};

enum FilterType
{
    SALT_PEPPER,
    GAUSSIAN,
    BLUR
};

// void processFrame(VideoCapture &vsrc, Mat &dst, char mode);

class ProcessVideo
{
private:
    VideoCapture &videoStream;
    FrameBuffer &buffer;
    Mat lastFrame;
    int framesForTracking;
    const string facedetectorData =
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt2.xml";
    CascadeClassifier faceDetector;
    bool showTextInfo, storeVideo, finished;
    Action noDetectAction;
    FilterType filtering;

    // to save the video
    VideoWriter videoSaver;

    std::mutex lock;
    std::condition_variable cvPauseResume;
    bool isPaused;
    bool isAborted;

    // Comparator to sort the detected faces by area
    static bool areaComparator(const Rect &l, const Rect &r)
    {
        return (l.area() < r.area());
    }

    // RotatedRect track(Mat &frame, Mat& roi, Rect& track_window)
    // {
    //     Mat roi, hsv_roi, mask;

    //     cvtColor(roi, hsv_roi, COLOR_BGR2HSV);
    //     inRange(hsv_roi, Scalar(0, 60, 32), Scalar(180, 255, 255), mask);

    //     float range_[] = {0, 180};
    //     const float *range[] = {range_};
    //     Mat roi_hist;
    //     int histSize[] = {180};
    //     int channels[] = {0};
    //     calcHist(&hsv_roi, 1, channels, mask, roi_hist, 1, histSize, range);
    //     normalize(roi_hist, roi_hist, 0, 255, NORM_MINMAX);

    //     // Setup the termination criteria
    //     TermCriteria term_crit(TermCriteria::EPS, 1, 1);


    //     while (true)
    //     {
    //         Mat hsv, dst;
    //         capture >> frame;
    //         if (frame.empty())
    //             break;
    //         cvtColor(frame, hsv, COLOR_BGR2HSV);
    //         calcBackProject(&hsv, 1, channels, roi_hist, dst, range);

    //         // apply camshift to get the new location
    //         RotatedRect rot_rect = CamShift(dst, track_window, term_crit);

    //         // Draw it on image
    //         Point2f points[4];
    //         rot_rect.points(points);
    //         for (int i = 0; i < 4; i++)
    //             line(frame, points[i], points[(i + 1) % 4], 255, 2);
    //         imshow("img2", frame);

    //         int keyboard = waitKey(30);
    //         if (keyboard == 'q' || keyboard == 27)
    //             break;
    //     }
    // }

public:
    ProcessVideo(VideoCapture &vs, FrameBuffer &buff, int n)
        : videoStream(vs), buffer(buff), framesForTracking(n), showTextInfo(false),
          storeVideo(false), noDetectAction(SHOW_MESSAGE), filtering(BLUR),
          finished(false), isPaused(false), isAborted(false)
    {
        if (!faceDetector.load(facedetectorData))
        {
            throw runtime_error("--(!)Error loading face detector");
        }

        videoSaver = VideoWriter("video_out.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'),
                                 videoStream.get(VideoCaptureProperties::CAP_PROP_FPS),
                                 Size(videoStream.get(VideoCaptureProperties::CAP_PROP_FRAME_WIDTH),
                                      videoStream.get(VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT)));

        if (!videoSaver.isOpened())
        {
            throw runtime_error("--(!)Error video_out could not be opened");
        }

        // check frames of input video
        if (videoStream.get(VideoCaptureProperties::CAP_PROP_FRAME_COUNT) < framesForTracking)
        {
            throw runtime_error("--(!)Error n must be less than number of frames in the video");
        }
    }

    void operator()()
    {
        std::unique_lock<std::mutex> lck(lock);

        videoStream >> lastFrame;
        while ((!lastFrame.empty()) && (!isAborted))
        {
            // pause frame processing
            cvPauseResume.wait(lck, [this]()
                               { return !isPaused; });
            //  Face detection
            vector<Rect> faces;
            faceDetector.detectMultiScale(lastFrame, faces, 1.1, 3, 0, Size(30, 30), Size(500, 500));
            // Process only if any face is detected
            if (!faces.empty())
            {
                if (faces.size() > 1)
                {
                    sort(faces.begin(), faces.end(), areaComparator);
                }
                Mat faceROI = lastFrame(faces.back());
                GaussianBlur(faceROI, faceROI, Size(23, 23), 30);
            }
            else
            {
                // no face detected
                cout << "FACE no detected, op = " << noDetectAction << " ADDRESS = " << &noDetectAction << endl;
                switch (noDetectAction)
                {
                case SHOW_MESSAGE:
                    putText(lastFrame, "ATENCION: Ningun rostro detectado",
                            Point(10, 40), FONT_HERSHEY_SIMPLEX, 1.1, Scalar(0, 255, 0), 4);
                    break;
                case BLUR_ALL:
                    GaussianBlur(lastFrame, lastFrame, Size(31, 31), 50, 2);
                    break;
                case NO_ACTION:
                    break;
                }
            }
            // Stop processing when main thread is aborting
            // doing it here too to avoid possible blocking on the buffer
            if (isAborted)
                return;

            // save frame on video file
            videoSaver.write(lastFrame);
            // pass new processed frame
            buffer.deposit(lastFrame);
            // get next frame to process
            videoStream >> lastFrame;
        }
        // signal finished
        finished = true;
        // Stop processing when main thread is aborting
        if (isAborted)
            return;
        // RELEASE lock at finish
        lastFrame = Mat::zeros(10, 10, CV_8U);
        buffer.deposit(lastFrame);
    }

    void pause() { isPaused = true; }

    void resume()
    {
        isPaused = false;
        cvPauseResume.notify_all();
    }

    bool isVideoPaused() { return isPaused; }

    bool isProcessingFinished() const { return finished; }

    // For abrupt termination
    void abort() { isAborted = true; }

    void setShowTextInfo(bool show) { showTextInfo = show; }

    void setStoreVideo(bool store) { storeVideo = store; }

    void changeNoDetectAction()
    {
        cout << "CHANGING ACTION METHOD, actual action: " << noDetectAction
             << " ADDRESS = " << &noDetectAction << endl;
        switch (noDetectAction)
        {
        case SHOW_MESSAGE:
            noDetectAction = BLUR_ALL;
            break;
        case NO_ACTION:
            noDetectAction = SHOW_MESSAGE;
            break;
        case BLUR_ALL:
            noDetectAction = NO_ACTION;
        default:
            break;
        }
        cout << "exiting, new action: " << noDetectAction << endl;
    }

    ~ProcessVideo()
    {
        if (videoStream.isOpened())
            videoStream.release();
        if (videoSaver.isOpened())
            videoSaver.release();
    }
};

class thread_guard
{
    thread &t;

public:
    explicit thread_guard(thread &t_) : t(t_) {}
    ~thread_guard()
    {
        if (t.joinable())
            t.join();
        // t.detach();
        // t.~thread(); // Force the thread to terminate
    }
    thread_guard(thread_guard const &) = delete;
    thread_guard &operator=(thread_guard const &) = delete;
};

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        cout << " Usage: opencv_basico N VideoToLoadAndDisplay" << endl;
        return -1;
    }

    int N = std::atoi(argv[1]);

    VideoCapture videoSource;
    FrameBuffer *buffFrames = new FrameBuffer(1000);

    videoSource.open(argv[2]);
    if (!videoSource.isOpened())
    {
        cout << "Could not open or find the video" << std::endl;
        return -1;
    }

    namedWindow("processed", WINDOW_KEEPRATIO);

    ProcessVideo processor(videoSource, *buffFrames, N);
    // important! If you don't initialize thread object
    // wrapping the Func object passed by parameter
    // with std::ref, thread object will do a copy
    // of the proporties in the passed object.
    // And it's necessary for the operation of ProcessVideo
    // object that the object operated in the thread being
    // the original object in order to do signaling over it
    thread threadProcessor(std::ref(processor));
    thread_guard guard(threadProcessor);

    char key = '3'; // to set default mode NORMAL
    bool exit = false;
    while (!exit)
    {
        // cout << "BLOCKING" << endl;

        // To avoid program blocking when processing
        // is paused, else main thread will be blocked
        // in buffFrames
        if (!processor.isVideoPaused())
        {
            imshow("processed", buffFrames->fetch());
        }
        // cout << "RELEASE" << endl;
        if (processor.isProcessingFinished())
            break;
        key = (char)waitKey(5);
        switch (key)
        {
        case 'q':
        case 'Q':
        case 27: // escape key
            processor.abort();
            exit = true;
            break;
        case 'n':
        case 'N':
            processor.changeNoDetectAction();
            break;
        case 'p':
        case 'P':
            processor.pause();
            break;
        case 'r':
        case 'R':
            processor.resume();
            break;
        default:
            break;
        }
    }

    // Mat frame;
    // char key = '3';    // to set default mode NORMAL
    // int mode = NORMAL; // default mode
    // for (;;)
    // {
    //     processFrame(videoSource, frame, mode);
    //     if (frame.empty())
    //         break;
    //     imshow("Original", frame);
    //     key = (char)waitKey(5);
    //     switch (key)
    //     {
    //     case 'q':
    //     case 'Q':
    //     case 27: // escape key
    //         return 0;
    //     case '1':
    //         mode = GRAY;
    //         break;
    //     case '2':
    //         mode = BINARIZE;
    //         break;
    //     case '3':
    //         mode = NORMAL;
    //         break;
    //     default:
    //         break;
    //     }
    // }

    // Clean
    if (buffFrames != nullptr)
    {
        delete buffFrames;
        buffFrames = nullptr;
    }

    return 0;
}

// void processFrame(VideoCapture &videoSource, Mat &dst, char mode)
// {
//     Mat frame;
//     videoSource >> frame;
//     switch (mode)
//     {
//     case GRAY:
//         cvtColor(frame, dst, COLOR_BGR2GRAY);
//         break;
//     case BINARIZE:
//         cvtColor(frame, dst, COLOR_BGR2GRAY);
//         threshold(dst, dst, 100, 255, THRESH_BINARY);
//         break;
//     case NORMAL:
//     default:
//         dst = frame.clone();
//         break;
//     }
// }