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

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        cout << " Usage: display_image ImageToLoadAndDisplay" << endl;
        return -1;
    }

    Mat image;
    image = imread(argv[1], IMREAD_COLOR); // Read the file

    if (!image.data) // Check for invalid input
    {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    // mostrar tamanio, canales y profundidad
    Size size = image.size();
    printf("Filas x columnas: %d x %d\nNumero de canales: %d\nProfundidad: %d\n",
           size.height, size.width, image.channels(), image.depth());

    // Generar mascara clonando imagen, "binarizar"
    Mat mask = image.clone();
    typedef cv::Point3_<uint8_t> Pixel;
    int threshold = 127;
    for (Pixel &p : Mat_<Pixel>(mask))
    {
        // para blanco y negro dejar todos
        // los canales con la misma informacion
        p.x = p.y = p.z = (p.x < threshold) ? 0 : 255;
    }

    // generar imagen a partir de original y mascara
    Mat image2;
    image.copyTo(image2, mask);

    // display images
    namedWindow("original", WINDOW_AUTOSIZE); 
    imshow("original", image);     
    namedWindow("mask", WINDOW_AUTOSIZE); 
    imshow("mask", mask);     
    namedWindow("nueva", WINDOW_AUTOSIZE); 
    imshow("nueva", image2);          

    waitKey(0); // Wait for a keystroke in the window
    return 0;
}