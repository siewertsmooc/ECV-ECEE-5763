/*
 *
 *  Example by Sam Siewert 
 *
 *  Updated 10/29/16 for OpenCV 3.1
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

//#define HRES 640
//#define VRES 480


int main( int argc, char** argv )
{
    //cvNamedWindow("Capture Example", CV_WINDOW_AUTOSIZE);
    //CvCapture* capture = cvCreateCameraCapture(0);
    Mat mat_frame, mat_gray, mat_component[3], mat_zero, mat_composite;
    //IplImage* frame;
    VideoCapture vcap;

    //cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, HRES);
    //cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, VRES);

    //open the video stream and make sure it's opened
    // "0" is the default video device which is normally the built-in webcam
    if(!vcap.open(0)) 
    {
        std::cout << "Error opening video stream or file" << std::endl;
        return -1;
    }
    else
    {
	   std::cout << "Opened default camera interface" << std::endl;
    }


    while(1)
    {
        //frame=cvQueryFrame(capture);
	if(!vcap.read(mat_frame)) {
		std::cout << "No frame" << std::endl;
		cv::waitKey();
	}
        //if(!frame) break;
	
	cv::cvtColor(mat_frame, mat_gray, COLOR_BGR2GRAY);

	// splits into BGR
        split(mat_frame, mat_component);

        //cvShowImage("Capture Example", frame);
	cv::imshow("Color Example", mat_frame);
	cv::imshow("Gray Example", mat_gray);

	// gray B, G, and R channels balanced
	cv::imshow("Blue Gray Example", mat_component[0]);
	cv::imshow("Green Gray Example", mat_component[1]);
	cv::imshow("Red Gray Example", mat_component[2]);

        // B, G, R channels only, others zero
	mat_zero=Mat::zeros(Size(mat_frame.cols, mat_frame.rows), CV_8UC1);
       
        {	
	    vector<Mat> channels;
            channels.push_back(mat_zero);
            channels.push_back(mat_zero);
            channels.push_back(mat_component[2]);
            merge(channels, mat_composite);
            imshow("Red Example", mat_composite);
        }

        {	
	    vector<Mat> channels;
            channels.push_back(mat_zero);
            channels.push_back(mat_component[1]);
            channels.push_back(mat_zero);
            merge(channels, mat_composite);
            imshow("Green Example", mat_composite);
        }

        {	
	    vector<Mat> channels;
            channels.push_back(mat_component[0]);
            channels.push_back(mat_zero);
            channels.push_back(mat_zero);
            merge(channels, mat_composite);
            imshow("Blue Example", mat_composite);
        }

        char c = waitKey(10);
        if( c == 'q' ) break;
    }

    //cvReleaseCapture(&capture);
    //cvDestroyWindow("Capture Example");
    
};
