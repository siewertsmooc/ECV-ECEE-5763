/*
 *
 *  Example by Sam Siewert for use with PPM images to compress with a 
 *  a process similar to MPEG, to provide a simple example.
 *
 *  PART 1: Simple conversion of image to DCT and back with inverse DCT
 *
 *  Based on numerous code snippets from stackoverflow.com
 *
 */
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;


int main( int argc, char** argv )
{
    IplImage *image;                  // basic image pointer
    IplImage *b,*g,*r;                // blue, green, red only images
    IplImage *bf, *gf, *rf;           // float versions
    IplImage *b_dct,*g_dct,*r_dct;    // DCT converted float versions
    IplImage *b_idct,*g_idct,*r_idct; // DCT converted float versions

    // Read in command line supplied file argument for PPM, JPG, etc.
    if( argc != 2)
    {
     cout <<" Usage: display_image ImageToLoadAndDisplay" << endl;
     return -1;
    }

    // Read in the source image file
    image = cvLoadImage(argv[1]);

    // Allocate each color band image and set pointer
    b=cvCreateImage(cvGetSize(image),IPL_DEPTH_8U,1);
    g=cvCreateImage(cvGetSize(image),IPL_DEPTH_8U,1);
    r=cvCreateImage(cvGetSize(image),IPL_DEPTH_8U,1);

    // Split read in image into the three color bands
    cvSplit(image, b, g, r, NULL);

    // Allocate each float color band image and set pointer, then convert
    bf = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, 1);
    cvConvert(b, bf);
    gf = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, 1);
    cvConvert(g, gf);
    rf = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, 1);
    cvConvert(r, rf);

    // Allocate each float DCT, iDCT color band image and set pointer
    b_dct=cvCreateImage(cvGetSize(image),IPL_DEPTH_32F,1);
    b_idct=cvCreateImage(cvGetSize(image),IPL_DEPTH_32F,1);
    g_dct=cvCreateImage(cvGetSize(image),IPL_DEPTH_32F,1);
    g_idct=cvCreateImage(cvGetSize(image),IPL_DEPTH_32F,1);
    r_dct=cvCreateImage(cvGetSize(image),IPL_DEPTH_32F,1);
    r_idct=cvCreateImage(cvGetSize(image),IPL_DEPTH_32F,1);

    cvDCT(bf, b_dct, 0);
    cvDCT(gf, g_dct, 0);
    cvDCT(rf, r_dct, 0);
    cvDCT(b_dct, b_idct, DCT_INVERSE);
    cvDCT(g_dct, g_idct, DCT_INVERSE);
    cvDCT(r_dct, r_idct, DCT_INVERSE);

    // Create a window for display.
    //namedWindow("B Display window", CV_WINDOW_AUTOSIZE );
    //namedWindow("G Display window", CV_WINDOW_AUTOSIZE );
    //namedWindow("R Display window", CV_WINDOW_AUTOSIZE );
    namedWindow("Display window", CV_WINDOW_AUTOSIZE );

    // Show original image
    cvShowImage("Display window", image);

    // Wait for a keystroke in the window
    waitKey(0);

    // convert float DCT b color band to simple image
    cvConvert(b_dct, b);
    cvConvert(g_dct, g);
    cvConvert(r_dct, r);
    cvShowImage("Display window", g);
    //cvShowImage("B Display window", b);
    //cvShowImage("G Display window", g);
    //cvShowImage("R Display window", r);

    // Wait for a keystroke in the window
    waitKey(0);

    // convert float iDCT b color band to simple image
    cvConvert(b_idct, b);
    cvConvert(g_idct, g);
    cvConvert(r_idct, r);
    cvShowImage("Display window", g);
    //cvShowImage("B Display window", b);
    //cvShowImage("G Display window", g);
    //cvShowImage("R Display window", r);

    // Wait for a keystroke in the window
    waitKey(0);

    return 0;
};
