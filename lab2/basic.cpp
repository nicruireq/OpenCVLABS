/*
 * basic.cpp
 *      Author: NRR
 */

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <cstdio>

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

    // mostrar tamanio, canales y profundidad
    Size size = image.size();
    printf("Filas x columnas: %d x %d\nNumero de canales: %d\nProfundidad: %d",
        size.height, size.width, image.channels(), image.depth()
    );

    // Generar mascara clonando imagen, binarizar, y poner a cero una
    // cantidad de indices generados aleatoriamente
    Mat mask = image.clone();



    namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Display window", image );                   // Show our image inside it.

    waitKey(0);                                          // Wait for a keystroke in the window
    return 0;
}