/*
 * Detect faces
 *
 *  Created on: March 5, 2014
 *      Author: mseei
 */

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    if( argc != 2)
    {
     cout <<" Usage: ejemplo_opencv_mvc ImageToLoadAndProcess" << endl;
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

	String facedetector_data = "C:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml";
	CascadeClassifier face_detector;
	if( !face_detector.load(facedetector_data) )
	{ 
		printf("--(!)Error loading face detector\n"); 
		return -1; 
	}

	std::vector<Rect> faces;
	face_detector.detectMultiScale(image2, faces, 1.1, 3, 0, cvSize(30,30), cvSize(500,500));
	for (int i=0; i<faces.size(); i++)
	{
		printf("Detected face %d: %d %d %d %d\n", i, faces.at(i).x, faces.at(i).y, faces.at(i).width, faces.at(i).height);
	}

	namedWindow( "Original", WINDOW_AUTOSIZE );	
    imshow( "Original", image );                   // Imagen original.
	namedWindow( "Result", WINDOW_AUTOSIZE );	
    imshow( "Result", image2 );                   // Imagen que ten�a la ROI asociada. Puede verse c�mo los cambios en la subimagen 'image_roi' aparecen en la ROI de esta imagen

    waitKey(0);                                          // Wait for a keystroke in the window
    return 0;
}
