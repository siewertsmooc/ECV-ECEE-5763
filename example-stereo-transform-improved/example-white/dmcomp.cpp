/*
 *
 *  Example by Sam Siewert to show various white points and gray values
 *  for 3 channel color of different depths and for RGB and CIE XYZ
 *
 *  Based on numerous code snippets from stackoverflow.com
 *
 *  For color, refer to http://www.cs.rit.edu/~ncs/color/t_convert.html
 *
 */
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

#define SAT8 (255)
#define SAT16 (65535)
#define SATF (1.0)


int main( int argc, char** argv )
{
    IplImage *image8;
    IplImage *image16;
    IplImage *imagef;
    IplImage *imagef709; // D65 white point is default
    int idx;
    char key;
    int sat8=SAT8;
    int sat16=SAT16;
    float satf=SATF;


    image8 = cvCreateImage(Size(640,480), IPL_DEPTH_8U, 3);
    image16 = cvCreateImage(Size(640,480), IPL_DEPTH_16U, 3);
    imagef = cvCreateImage(Size(640,480), IPL_DEPTH_32F, 3);
    imagef709 = cvCreateImage(Size(640,480), IPL_DEPTH_32F, 3);

    // saturate each in all 3 channels
    cvSet(image8, CV_RGB(sat8,sat8,sat8));
    cvSet(image16, CV_RGB(sat16,sat16,sat16));
    cvSet(imagef, CV_RGB(satf,satf,satf));

    // convert RGB saturated white to XYZ Rec 709 with D65 white point
    // Y=luminance and XZ defines a chromaticity
    //
    // For CIE D65 white point, XYZ=0.95046,1.00000,1.08875
    //
    // For CEI D55 white point, XYZ=1.0985,1.0000,0.3558
    //
    cvCvtColor(imagef, imagef709, CV_RGB2XYZ);

    printf("Saturated 32F RGB=%f,%f,%f; XYZ=%f,%f,%f\n", ((float *)imagef->imageData)[0], ((float *)imagef->imageData)[1], ((float *)imagef->imageData)[2], ((float *)imagef709->imageData)[0], ((float *)imagef709->imageData)[1], ((float *)imagef709->imageData)[2]);

    // Create a window for display.
    namedWindow("Color Demo Display 8U", CV_WINDOW_AUTOSIZE );
    namedWindow("Color Demo Display 16U", CV_WINDOW_AUTOSIZE );
    namedWindow("Color Demo Display 32F", CV_WINDOW_AUTOSIZE );
    namedWindow("Color Demo Display Rec709 32F", CV_WINDOW_AUTOSIZE );

    while(1)
    {
        cvShowImage("Color Demo Display 8U", image8);
        cvShowImage("Color Demo Display 16U", image16);
        cvShowImage("Color Demo Display 32F", imagef);
        cvShowImage("Color Demo Display Rec709 32F", imagef709);

        // Wait for a keystroke in the window
        key=waitKey(0);

        printf("key code = %u, %c\n", key, key);

        if(key == 'R')
        {
            (sat8+1) > SAT8 ? sat8=SAT8 : sat8++;  printf("sat8 increase to %d\n", sat8);
            (sat16+1) > SAT16 ? sat16=SAT16 : sat16++;  printf("sat16 increase to %d\n", sat16);
            satf+=0.001;  printf("satf increase to %f\n", satf);
            printf("Saturated 32F RGB=%f,%f,%f; XYZ=%f,%f,%f\n", ((float *)imagef->imageData)[0], ((float *)imagef->imageData)[1], ((float *)imagef->imageData)[2], ((float *)imagef709->imageData)[0], ((float *)imagef709->imageData)[1], ((float *)imagef709->imageData)[2]);
        }
        else if(key == 'T')
        {
            (sat8-1) < 0 ? sat8=0 : sat8--;  printf("sat8 decrease to %d\n", sat8);
            (sat16-1) < 0 ? sat16=0 : sat16--;  printf("sat16 decrease to %d\n", sat16);
            satf-=0.001;  printf("satf decrease to %f\n", satf);
            printf("Saturated 32F RGB=%f,%f,%f; XYZ=%f,%f,%f\n", ((float *)imagef->imageData)[0], ((float *)imagef->imageData)[1], ((float *)imagef->imageData)[2], ((float *)imagef709->imageData)[0], ((float *)imagef709->imageData)[1], ((float *)imagef709->imageData)[2]);
        }
        else if(key == '-')
        {
            sat8=SAT8-sat8; printf("sat8 negative to %d\n", sat8);
            sat16=SAT16-sat16; printf("sat16 negative to %d\n", sat16);
        }

        cvSet(image8, CV_RGB(sat8,sat8,sat8));
        cvSet(image16, CV_RGB(sat16,sat16,sat16));
        cvSet(imagef, CV_RGB(satf,satf,satf));
        cvCvtColor(imagef, imagef709, CV_RGB2XYZ);
    }

    return 0;
};
