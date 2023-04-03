/*
 * Modificaciones en la ROI de una imagen
 *
 *  Created on: Feb 7, 2014
 *      Author: mseei
 */

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
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

    Size tam;
    tam = image.size();
    printf("Tama�o de la imagen: Ancho: %d y Alto %d\n",tam.width,tam.height);

    Mat image2 = image.clone();

    Mat image_roi(image2,Rect(50,100,90,50)); // se define una ROI en image2. La esquina superior izquierda est� en (50,100), la anchura es 90 y la altura 50
	
    threshold(image_roi,image_roi,100,255,THRESH_BINARY_INV); // se binariza (invertida) s�lo la ROI

    namedWindow( "Original", WINDOW_AUTOSIZE );	
    imshow( "Original", image );                   // Imagen original.
    namedWindow( "ROI", WINDOW_AUTOSIZE );	
    imshow( "ROI", image_roi );                   // ROI.
    namedWindow( "Result", WINDOW_AUTOSIZE );	
    imshow( "Result", image2 );                   // Imagen que ten�a la ROI asociada. Puede verse c�mo los cambios en la subimagen 'image_roi' aparecen en la ROI de esta imagen

    waitKey(0);                                          // Wait for a keystroke in the window
    return 0;
}