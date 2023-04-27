/*
 * Blur face of major area detected in frames from video
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
#include <chrono>

using namespace cv;
using namespace std;
using namespace std::chrono;

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

// To get correct font scale for a string to fit in a frame of w x h
double fitStringInFrame(const String &text, int width, int height, int fontFace,
                        int thickness, int desiredFontHeight, int minimumFontSize, int *finalFontHeight)
{
    double fontScale = getFontScaleFromHeight(fontFace, desiredFontHeight, thickness);
    Size sz;
    sz = getTextSize(text, fontFace, fontScale, thickness, nullptr);
    int newFontHeight = desiredFontHeight;
    // while result string overflow the frame in width
    // stop if the font gets too short (8pt)
    while ((sz.width > width) || (newFontHeight <= minimumFontSize))
    {
        newFontHeight -= 1;
        fontScale = getFontScaleFromHeight(fontFace, newFontHeight, thickness);
        sz = getTextSize(text, fontFace, fontScale, thickness, nullptr);
    }

    if (finalFontHeight != nullptr)
    {
        *finalFontHeight = sz.height;
    }
    return fontScale;
}

class ProcessVideo
{
private:
    VideoCapture &videoStream;
    FrameBuffer &buffer; // to pass processed frames to main thread
    Mat lastFrame;       // frame being processed

    // tracking and detection
    int framesForTracking;
    const string facedetectorData =
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt2.xml";
    CascadeClassifier faceDetector;
    // signaling
    bool showTextInfo, storeVideo, finished;
    Action noDetectAction;

    // to control level of blurring
    static constexpr float MIN_BLURR_LEVEL = 0.5;
    static constexpr float MAX_BLURR_LEVEL = 30.0;
    static constexpr float BLURR_STEP = 0.5;
    float blurrLevel;

    // to do measurements
    high_resolution_clock::time_point timePre, timePost;
    int preFrameCount, postFrameCount; // instantaneous frames number
    int framesCounter;                 // full frames counter
    struct Measure
    {
        int frameNumber, faces;
        double processingTimeSeconds;

        Measure(int fn, int fc, double t)
            : frameNumber(fn), faces(fc),
              processingTimeSeconds(t) {}

        Measure()
            : frameNumber(0), faces(0),
              processingTimeSeconds(0.0) {}
    };
    vector<Measure> measures;
    bool isShowingInfo;

    // to save the video
    VideoWriter videoSaver;

    // for concurrency management
    std::mutex lock;
    std::condition_variable cvPauseResume;
    bool isPaused;
    bool isAborted;

    // Comparator to sort the detected faces by area
    static bool areaComparator(const Rect &l, const Rect &r)
    {
        return (l.area() < r.area());
    }

    void writeFrame()
    {
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

    void paintInfoOnFrame()
    {
        if (isShowingInfo)
        {
            double totalTime = 0.0, fps;
            for (size_t i = 0; i < framesCounter; i++)
            {
                totalTime += measures[i].processingTimeSeconds;
            }
            fps = framesCounter / totalTime;

            // painting
            string info = cv::format("Frame: %d | Frame rate: %.3f fps | Faces: %d",
                                     measures[framesCounter].frameNumber,
                                     fps,
                                     measures[framesCounter].faces);
            // put origin at the bottom-left corner
            int w = videoStream.get(VideoCaptureProperties::CAP_PROP_FRAME_WIDTH);
            int h = videoStream.get(VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT);
            Point origin(w * 0.02, h - h * 0.02);
            int font = FONT_HERSHEY_SIMPLEX;
            int thickness = 2;
            // double fontScale = getFontScaleFromHeight(font, 18, thickness);
            double fontScale = fitStringInFrame(info, w, h, font, thickness, 18, 8, nullptr);
            putText(lastFrame, info, origin, font, fontScale, Scalar(0, 0, 255), thickness);
        }
    }

    void trackingAndFiltering(Mat &roi, Rect &track_window, int facesInFrame)
    {
        Mat hsv_roi, mask;

        cvtColor(roi, hsv_roi, COLOR_BGR2HSV);
        inRange(hsv_roi, Scalar(0, 60, 32), Scalar(180, 255, 255), mask);

        float range_[] = {0, 180};
        const float *range[] = {range_};
        Mat roi_hist;
        int histSize[] = {180};
        int channels[] = {0};
        calcHist(&hsv_roi, 1, channels, mask, roi_hist, 1, histSize, range);
        normalize(roi_hist, roi_hist, 0, 255, NORM_MINMAX);

        // Setup the termination criteria
        // TermCriteria term_crit(TermCriteria::EPS, 1, 1);
        TermCriteria term_crit(TermCriteria::EPS | TermCriteria::COUNT, 10, 1);

        for (size_t i = 0; i < framesForTracking; i++)
        {
            Mat hsv, dst;

            // measurements for first frame comming from operator()
            if (i == 0)
            {
                // possible refactoring of this chunk of code
                timePost = high_resolution_clock::now();
                duration<double> time_span = duration_cast<duration<double>>(timePost - timePre);
                framesCounter++;
                measures[framesCounter] = Measure(framesCounter, facesInFrame, time_span.count());
                paintInfoOnFrame(); // update info and show
            }
            else
            {
                // begin to count time for frames being processed by camshift algorithm
                timePre = high_resolution_clock::now();
            }

            // first process thr frame comming from operator()
            cvtColor(lastFrame, hsv, COLOR_BGR2HSV);
            calcBackProject(&hsv, 1, channels, roi_hist, dst, range);

            // apply camshift to get the new location
            RotatedRect rot_rect = CamShift(dst, track_window, term_crit);

            // Draw it on image
            Rect contour = rot_rect.boundingRect();
            Rect bounds(0, 0, lastFrame.cols, lastFrame.rows); // frame boundaries
            // Be sure that the tracking area is inside the boundaries of the image
            // because boundingRect() can return negative axis that are outsise the frame borders
            // Applying intersection with the full frame boundaries in a rect object
            Mat faceROIMoved = lastFrame(contour & bounds);
            // Apply filters
            // intensity of blurring is controlled by the blurrLevel property
            GaussianBlur(faceROIMoved, faceROIMoved, Size(23, 23), blurrLevel, blurrLevel);

            // Measurement
            // possible refactoring of this chunk of code
            timePost = high_resolution_clock::now();
            duration<double> time_span = duration_cast<duration<double>>(timePost - timePre);
            framesCounter++; // count processed frame
            measures[framesCounter] = Measure(framesCounter, facesInFrame, time_span.count());
            paintInfoOnFrame(); // update info and show

            // write to video and pass to main thread to show processed frame
            writeFrame();

            // grab next frame
            videoStream >> lastFrame;
            if (lastFrame.empty())
                break;
        }
    }

public:
    ProcessVideo(VideoCapture &vs, FrameBuffer &buff, int n)
        : videoStream(vs), buffer(buff), framesForTracking(n), showTextInfo(false),
          storeVideo(false), noDetectAction(SHOW_MESSAGE), isShowingInfo(false),
          finished(false), isPaused(false), isAborted(false),
          blurrLevel(ProcessVideo::MIN_BLURR_LEVEL), framesCounter(0)
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

        // initialize vector with measures (frame_number, faces_detected, processing_time)
        measures = vector<Measure>(videoStream.get(VideoCaptureProperties::CAP_PROP_FRAME_COUNT));
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

            // grab instant before processing
            timePre = high_resolution_clock::now();

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
                // camshift and blurring
                trackingAndFiltering(faceROI, faces.back(), faces.size());
            }
            else
            {
                // Configure params to fit text and show it later
                int w = videoStream.get(VideoCaptureProperties::CAP_PROP_FRAME_WIDTH);
                int h = videoStream.get(VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT);
                int font = FONT_HERSHEY_SIMPLEX;
                int thickness = 1;
                string warning = "ATENCION: Ningun rostro detectado";
                int fontHeight;
                double fontScale = fitStringInFrame(warning, w, h, font, thickness, 18, 2, &fontHeight);
                Point origin(w * 0.02, fontHeight);

                switch (noDetectAction)
                {
                case SHOW_MESSAGE:

                    putText(lastFrame, warning, origin, font, fontScale,
                            Scalar(0, 255, 0), thickness);
                    break;
                case BLUR_ALL:
                    GaussianBlur(lastFrame, lastFrame, Size(31, 31), 50, 2);
                    break;
                case NO_ACTION:
                    break;
                }
                // Measurement
                // possible refactoring of this chunk of code
                timePost = high_resolution_clock::now();
                duration<double> time_span = duration_cast<duration<double>>(timePost - timePre);
                framesCounter++; // count processed frame
                measures[framesCounter] = Measure(framesCounter, 0, time_span.count());
                paintInfoOnFrame(); // update info and show
            }

            // may be cause a problem:
            writeFrame();
        }
        // signal finished
        finished = true;
        // Stop processing when main thread is aborting
        if (isAborted)
            return;
        // RELEASE lock at finish
        lastFrame = Mat::zeros(
            videoStream.get(VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT),
            videoStream.get(VideoCaptureProperties::CAP_PROP_FRAME_WIDTH),
            CV_8U);
        buffer.deposit(lastFrame);
    }

    void showInfo()
    {
        if (isShowingInfo)
            isShowingInfo = false;
        else
            isShowingInfo = true;
    }

    void increaseBlurrLevel()
    {
        float sd = blurrLevel + ProcessVideo::BLURR_STEP;
        blurrLevel = (sd > ProcessVideo::MAX_BLURR_LEVEL)
                         ? ProcessVideo::MAX_BLURR_LEVEL
                         : sd;
        cout << "(+) BLURR : " << blurrLevel << endl;
    }

    void decreaseBlurrLevel()
    {
        float sd = blurrLevel - ProcessVideo::BLURR_STEP;
        blurrLevel = (sd < ProcessVideo::MIN_BLURR_LEVEL)
                         ? ProcessVideo::MIN_BLURR_LEVEL
                         : sd;
        cout << "(-) BLURR : " << blurrLevel << endl;
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
        case 'u':
        case 'U':
            processor.increaseBlurrLevel();
            break;
        case 'd':
        case 'D':
            processor.decreaseBlurrLevel();
            break;
        case 's':
        case 'S':
            processor.showInfo();
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