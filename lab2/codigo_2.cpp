/*
 * Invert and display image
 *
 *      Author: NRR
 */


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    if( argc != 2)
    {
     cout <<" Usage: display_image ImageToLoadAndDisplay" << endl;
     return -1;
    }

    Mat image;
    image = imread(argv[1], IMREAD_COLOR);   // Read the file

    if(! image.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

	int n_channels = image.channels();
	int width;
	Size size = image.size();
	width = size.width;

	for (int i=100; i<=300; i++)
		for (int j=100; j<=600; j++)
			for (int k=0; k<n_channels; k++) 
				image.data[i*width*n_channels + j*n_channels + k] = 255 - image.data[i*width*n_channels + j*n_channels + k];

	namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Display window", image );                   // Show our image inside it.

    waitKey(0);                                          // Wait for a keystroke in the window
    return 0;
}
