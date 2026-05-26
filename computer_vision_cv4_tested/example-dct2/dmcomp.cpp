/*
 *
 *  Example by Sam Siewert for use with PPM images to compress with a 
 *  a process similar to MPEG, to provide a simple example.
 *
 *  PART 1: Simple conversion of image to DCT and back with inverse DCT
 *
 *
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

using namespace std;
using namespace cv;

unsigned char framebuffer[(240*320)];
unsigned char framebufferdct[(240*320)];
double framebuffer_float[(240*320)];
double framebuffer_floatdct[(240*320)];

int main( int argc, char** argv )
{
    unsigned char val=255;
    // basic image pointer
    Mat testimg(240, 320, CV_8UC1, framebuffer);
    Mat testdct(240, 320, CV_8UC1, framebufferdct);
    Mat testimgf(240, 320, CV_32FC1, framebuffer_float);
    Mat testdctf(240, 320, CV_32FC1, framebuffer_floatdct);
    Point minLoc, maxLoc;
    double minV, maxV;

    testimg=imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    
    namedWindow("Display window", CV_WINDOW_AUTOSIZE );
    imshow("Display window", testimg);
    waitKey(0);

    // convert loaded graymap into floating point array
    testimg.convertTo(testimgf, CV_32FC1);

    // Apply DCT to testimgf (floating point) to get testdctf (floating point).
    dct(testimgf, testdctf, 0);

    // Convert it back to a graymap
    testdctf.convertTo(testdct, CV_8UC1);

    // Example of finding maximum and minimum pixel in test image
    minMaxLoc(testimgf, &minV, &maxV, &minLoc, &maxLoc, noArray());

    printf("minV=%lf, maxV=%lf\n", minV, maxV);
    printf("minX=%d, minY=%d\n", minLoc.x, minLoc.y);
    printf("maxX=%d, maxY=%d\n", maxLoc.x, maxLoc.y);

    // Wait for a keystroke in the window
    imshow("Display window", testdct);
    waitKey(0);

    return 0;
};
