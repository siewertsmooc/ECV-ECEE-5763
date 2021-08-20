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
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

//#define DHRES 160
//#define DVRES 120

//#define DHRES 320
//#define DVRES 240

#define DHRES 640
#define DVRES 480

//#define AHRES 512
//#define AVRES 384

//#define AHRES 256
//#define AVRES 192

char snapshotfile[]  = "save1/snapshot00000.jpg";
char framesavefile[] = "save1/frame00000000.jpg";

char snapshotfile2[]  = "save2/snapshot00000.jpg";
char framesavefile2[] = "save2/frame00000000.jpg";
    
IplImage* difframe2;
IplImage* difframe2_gray;
IplImage* difframe2_bw;

int main( int argc, char** argv )
{
    //cvNamedWindow("Capture Example", CV_WINDOW_AUTOSIZE);
    //CvCapture* capture = cvCreateCameraCapture(0);
    IplImage* frame;

    cvNamedWindow("Capture Example 2", CV_WINDOW_AUTOSIZE);
    CvCapture* capture2 = cvCreateCameraCapture(1);
    IplImage* frame2;
    IplImage* prevframe2;

    double diffSum=0.0;

    int cnt=0; int savingdata=0;
    int motioncnt=0;
    Mat mat_difframe2_gray;



    //cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, AHRES);
    //cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, AVRES);

    cvSetCaptureProperty(capture2, CV_CAP_PROP_FRAME_WIDTH, DHRES);
    cvSetCaptureProperty(capture2, CV_CAP_PROP_FRAME_HEIGHT, DVRES);
    
    frame2=cvQueryFrame(capture2);
    prevframe2=cvCreateImage(cvGetSize(frame2), frame2->depth, frame2->nChannels);
    difframe2=cvCreateImage(cvGetSize(frame2), frame2->depth, frame2->nChannels);
    //difframe2_gray=cvCreateImage(cvGetSize(frame2),IPL_DEPTH_8U,1);
    difframe2_bw=cvCreateImage(cvGetSize(frame2),IPL_DEPTH_8U,1);

    Mat mat_difframe2(difframe2);
    Mat mat_difframe2_bw(difframe2_bw);

    cvCopy(frame2, prevframe2);

    cvAbsDiff(frame2, prevframe2, difframe2);
    cvtColor(mat_difframe2, mat_difframe2_gray, CV_RGB2GRAY);
    threshold(mat_difframe2_gray, mat_difframe2_bw, 128, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);


    while(1)
    {
        //frame=cvQueryFrame(capture);
        frame2=cvQueryFrame(capture2);

        if(!frame || !frame2) break;

        cvAbsDiff(frame2, prevframe2, difframe2);
        cvtColor(mat_difframe2, mat_difframe2_gray, CV_RGB2GRAY);

        //threshold(mat_difframe2_gray, mat_difframe2_bw, 128, 1, CV_THRESH_BINARY | CV_THRESH_OTSU);
        threshold(mat_difframe2_gray, mat_difframe2_bw, 10, 1, CV_THRESH_BINARY);
        diffSum = (sum(mat_difframe2_bw)[0]);

        printf("diffSum=%lf, motioncnt=%d\n", diffSum, motioncnt);
       
        //cvShowImage("Capture Example", frame);
        cvShowImage("Capture Example 2", frame2);

        char c = cvWaitKey(30);

        if( c == 27 ) break;

        if( c == 'p') 
        {
            // on "p" key press
            cnt++;

            //printf("snapshot %05d taken as %s\n", cnt, snapshotfile2);
            //sprintf(&snapshotfile[8], "%05d.jpg", cnt);
            //cvSaveImage(snapshotfile, frame);

            printf("snapshot %05d taken as %s\n", cnt, snapshotfile2);
            sprintf(&snapshotfile2[8], "%05d.jpg", cnt);
            cvSaveImage(snapshotfile2, frame2);
        }

        if( c == 's' || savingdata ) 
        {
            // on "s" key press
            savingdata++;

            //printf("frame %08d saved as %s\n", savingdata, framesavefile);
            //sprintf(&framesavefile[8], "%08d.png", savingdata);
            //cvSaveImage(framesavefile, frame);
            printf("frame %08d saved as %s\n", savingdata, framesavefile2);
            sprintf(&framesavefile2[8], "%08d.png", savingdata);
            cvSaveImage(framesavefile2, frame2);
        }

        if( (c == 'm') || (motioncnt && (diffSum > 800)) ) 
        {
            // on "s" key press
            motioncnt++;


            printf("frame %08d saved as %s\n", motioncnt, framesavefile2);
            sprintf(&framesavefile2[8], "%08d.png", motioncnt);
            cvSaveImage(framesavefile2, frame2);
        }

        if( c == 'q' && (savingdata || motioncnt)) 
        {
            savingdata=0;
            motioncnt=0;
        }

        cvCopy(frame2, prevframe2);
    }

    //cvReleaseCapture(&capture);
    //cvDestroyWindow("Capture Example");
    cvReleaseCapture(&capture2);
    cvDestroyWindow("Capture Example 2");
    
};
