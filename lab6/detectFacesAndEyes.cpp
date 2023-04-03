/*
 * Detect faces and eyes
 *
 *      Author: NRR
 */

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    if( argc != 2)
    {
     cout <<" Usage: detectFacesAndEyes ImageToLoadAndProcess" << endl;
     return -1;
    }

    Mat image;
    image = imread(argv[1], IMREAD_COLOR);   // Read the file

    if(! image.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

	Mat image2 = image.clone();

	String facedetector_data = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt2.xml";
	CascadeClassifier face_detector;
	if( !face_detector.load(facedetector_data) )
	{ 
        cerr << "--(!)Error loading face detector" << endl;
		return -1; 
	}

	vector<Rect> faces;
	face_detector.detectMultiScale(image2, faces, 1.1, 3, 0, Size(30,30), Size(500,500));
	for (int i=0; i<faces.size(); i++)
	{
		printf("Detected face %d: %d %d %d %d\n", i, faces.at(i).x, faces.at(i).y, faces.at(i).width, faces.at(i).height);
	}

    // draw faces and get ROIs from faces
    vector<Mat> facesRois(faces.size());
    int i = 0;
    for (auto &&face : faces)
    {
        rectangle(image2, face, Scalar(0,0,255), 4);
        // extract roi using overloaded function operator:
        // Mat cv::Mat::operator()	(	const Rect & 	roi	)	const
        facesRois[i] = image2(face);
        i++;
    }

    // detect eyes in each ROI
    // configure classifier
    CascadeClassifier eyesDetector;
    String eyesDetectorData = "/usr/share/opencv4/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
    if( !eyesDetector.load(eyesDetectorData) )
	{ 
        cerr << "--(!)Error loading eyes detector" << endl;
		return -2; 
	}
    
    // vector of pairs (face - vector of eyes)
    vector<pair<Mat,vector<Rect>>> eyesForEachRoi(facesRois.size());
    i = 0;
    for (auto &&face : facesRois)
    {
        eyesForEachRoi[i].first = face;
        eyesForEachRoi[i].second = vector<Rect>();
        eyesDetector.detectMultiScale(face, eyesForEachRoi[i].second, 1.1, 3, 0, Size(10,10), Size(500,500));
        i++;
    }

    // draw eyes in the picture
    i = 0;
    for (auto &&face : eyesForEachRoi)
    {
        cout << "In face " << i << endl;
        for (auto &&eye : face.second)
        {
            cout << "Detected eye in: " << eye.x << " " << eye.y << " " << eye.width << " " << eye.height << endl;
            rectangle(face.first, eye, Scalar(0,255,0), 4);
        }
        i++;
    }

	namedWindow( "Original", WINDOW_AUTOSIZE );	
    imshow( "Original", image );                   // Original picture
	namedWindow( "Result", WINDOW_AUTOSIZE );	
    imshow( "Result", image2 );                   // Picture with faces and eyes detected and painted

    waitKey(0);                                          // Wait for a keystroke in the window
    return 0;
}
