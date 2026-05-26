/*
 *
 *  Example by Sam Siewert 
 *
 */
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

#define HRES 640
#define VRES 480


int main( int argc, char** argv )
{
    namedWindow("Capture Example", WINDOW_AUTOSIZE);
    VideoCapture capture(0);
    Mat frame;
    int dev=0;

    capture.set(CAP_PROP_FRAME_WIDTH, HRES);
    capture.set(CAP_PROP_FRAME_HEIGHT, VRES);

    if(argc > 1)
    {
        printf("argv[1]=%s\n", argv[1]);
        sscanf(argv[1], "%d", &dev);
        printf("Will open video device %d\n", dev);
        VideoCapture capture(dev);
    }

    while(1)
    {
        capture.read(frame);
     
        //if(!frame) break;

        imshow("Capture Example", frame);

        char c = waitKey(33);
        if( c == 'q' ) break;
    }

    capture.release();
    destroyWindow("Capture Example");
    
};
