/*
 * Process frames from video
 *
 *      Author: NRR
 */

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect.hpp>
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
    const string facedetectorData =
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt2.xml";
    CascadeClassifier faceDetector;
    bool showTextInfo, storeVideo, finished;
    Action noDetectAction;
    FilterType filtering;

    // Comparator to sort the detected faces by area
    static bool areaComparator(const Rect &l, const Rect &r)
    {
        return (l.area() < r.area());
    }

public:
    ProcessVideo(VideoCapture &vs, FrameBuffer &buff)
        : videoStream(vs), buffer(buff), showTextInfo(false),
          storeVideo(false), noDetectAction(SHOW_MESSAGE), filtering(BLUR),
          finished(false)
    {
        if (!faceDetector.load(facedetectorData))
        {
            throw runtime_error("--(!)Error loading face detector");
        }
    }

    void operator()()
    {
        videoStream >> lastFrame;
        while (!lastFrame.empty())
        {
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
                cout << "FACE no detected, op = " << noDetectAction << 
                 " ADDRESS = " << &noDetectAction << endl;
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

            // pass new processed frame
            buffer.deposit(lastFrame);
            // get next frame to process
            videoStream >> lastFrame;
        }
        // singal finished
        finished = true;
        // RELEASE lock at finish
        lastFrame = Mat::zeros(10, 10, CV_8U);
        buffer.deposit(lastFrame);
    }

    void pause() {}
    void resume() {}
    bool isFinished() const { return finished; }
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

    ~ProcessVideo() {}
};

class thread_guard
{
    thread &t;

public:
    explicit thread_guard(thread &t_) : t(t_) {}
    ~thread_guard()
    {
        // if (t.joinable())
        //     t.join();
        t.~thread(); // Force the thread to terminate
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

    ProcessVideo processor(videoSource, *buffFrames);
    thread threadProcessor(std::ref(processor));
    thread_guard guard(threadProcessor);

    char key = '3'; // to set default mode NORMAL
    bool exit = false;
    while (!exit)
    {
        // cout << "BLOCKING" << endl;
        imshow("processed", buffFrames->fetch());
        // cout << "RELEASE" << endl;
        if (processor.isFinished())
            break;
        key = (char)waitKey(5);
        switch (key)
        {
        case 'q':
        case 'Q':
        case 27: // escape key
            exit = true;
            break;
        case 'n':
        case 'N':
            processor.changeNoDetectAction();
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