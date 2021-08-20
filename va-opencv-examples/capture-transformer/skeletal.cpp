#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>

using namespace cv;
using namespace std;


void help()
{
 cout << "\nThis program demonstrates a skeletal transform.\n"
         "Usage:\n"
         "./skeletal <image_name>, Default is pic1.jpg\n" << endl;
}

int main(int argc, char** argv)
{
 const char* filename = argc >= 2 ? argv[1] : "pic1.jpg";

 Mat gray, binary, mfblur;
 Mat src = imread(filename, CV_LOAD_IMAGE_COLOR);

 if(src.empty())
 {
     help();
     cout << "can not open " << filename << endl;
     return -1;
 }

 // show original source image and wait for input to next step
 imshow("source", src);
 waitKey();


 cvtColor(src, gray, CV_BGR2GRAY);

 // show graymap of source image and wait for input to next step
 imshow("graymap", gray);
 waitKey();


 // Use 70 negative for Moose, 150 positive for hand
 // 
 // To improve, compute a histogram here and set threshold to first peak
 //
 // For now, histogram analysis was done with GIMP
 //
 threshold(gray, binary, 70, 255, CV_THRESH_BINARY);
 binary = 255 - binary;

 // show bitmap of source image and wait for input to next step
 imshow("binary", binary);
 waitKey();

 // To remove median filter, just replace blurr value with 1
 medianBlur(binary, mfblur, 1);

 // show median blur filter of source image and wait for input to next step
 //imshow("mfblur", mfblur);
 //waitKey();

 // This section of code was adapted from the following post, which was
 // based in turn on the Wikipedia description of a morphological skeleton
 //
 // http://felix.abecassis.me/2011/09/opencv-morphological-skeleton/
 //
 Mat skel(mfblur.size(), CV_8UC1, Scalar(0));
 Mat temp;
 Mat eroded;
 Mat element = getStructuringElement(MORPH_CROSS, Size(3, 3));
 bool done;
 int iterations=0;

 do
 {
   erode(mfblur, eroded, element);
   dilate(eroded, temp, element);
   subtract(mfblur, temp, temp);
   bitwise_or(skel, temp, skel);
   eroded.copyTo(mfblur);

   done = (countNonZero(mfblur) == 0);
   iterations++;
 
 } while (!done && (iterations < 100));

 cout << "iterations=" << iterations << endl;
 
 imshow("skeleton", skel);
 waitKey();
 
 return 0;
}
