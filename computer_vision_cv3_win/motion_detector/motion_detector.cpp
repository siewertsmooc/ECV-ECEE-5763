#include <iostream>
#include <fstream>

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <ctime>
#include <time.h>
#include <sstream>
//#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>

using namespace std;
using namespace cv;

#define WINDOW_NAME "Motion Detector"

#define DEVICE_ID 0

//#define SHOW_DIFF

#define DELAY_IN_MSEC (15)

#define DETECT_DIRECTORY ("/mnt/data/detect/")
#define COLLECT_DIRECTORY ("/mnt/data/collect/")

#define COLLECT_EVERY_NTH_FRAME 30

#define FILE_FORMAT ("%d%h%Y_%H%M%S") // 1Jan1970/1Jan1970_12153
#define EXTENSION (".png") // extension of the images

#define MAX_THRESHOLD 255
#define MAX_DEVIATION 255 // Maximum allowable deviation of a pixel to count as "changed"

#ifdef LOW_RES_DETECTION
int currentThreshold = 5;
int currentDeviation = 3;
int currentMotionTrigger = 10;
#else
//int currentThreshold = 15;
//int currentDeviation = 9;
//int currentMotionTrigger = 30;
int currentThreshold = 10;
int currentDeviation = 5;
int currentMotionTrigger = 20;
#endif

#define CAM_WIDTH_OFFSET 0

bool running = true;

typedef struct {
  bool isMotion;
  Scalar mean;
  Scalar stddev;
  int numberOfChanges;
} MotionDetectData_t;


#if 0
// Check if the directory exists, if not create it
// This function will create a new directory if the image is the first
// image taken for a specific day
inline void directoryExistsOrCreate(const char* pzPath) 
{
  DIR *pDir;    
  if ( pzPath == NULL || (pDir = opendir (pzPath)) == NULL) { // directory doesn't exists -> create it
    mkdir(pzPath, 0777);    
  } else if (pDir != NULL) { // if directory exists we opened it and we have to close the directory again.
    (void) closedir (pDir);
  }
}

// When motion is detected we write the image to disk
//    - Check if the directory exists where the image will be stored.
//    - Build the directory and image names.
inline bool saveImg(Mat image, const string directory,
                    const string extension, const string fileFormat,
                    const int sequence, int traceOn)
{
  stringstream ss;
  time_t seconds;
  struct tm * timeinfo;
  char timeStr[80];
  time(&seconds);

  // Get the current time
  timeinfo = localtime(&seconds);

  // Create name for the date directory
  directoryExistsOrCreate(directory.c_str());

  // Create name for the image
  strftime(timeStr, 80, fileFormat.c_str(), timeinfo);

  ss.str("");
  ss << directory << timeStr << "_" << sequence << extension;

  if(traceOn)
      cout << "Saving Image: " << ss.str() << endl;

  return imwrite(ss.str().c_str(), image);
}
#endif


// Check if there is motion in the result matrix.
// Count the number of changes and return.
inline MotionDetectData_t detectMotion(const Mat &motion, 
                                       int max_deviation, 
                                       int triggerCount) 
{

  int rows = motion.rows;
  int cols = motion.cols;
  
  int min_x = motion.cols, max_x = 0;
  int min_y = motion.rows, max_y = 0;
  
  // calculate the standard deviation
  MotionDetectData_t data;
  data.mean = 0;
  data.stddev = 0;
  data.numberOfChanges = 0;
  data.isMotion = false;
  
  meanStdDev(motion, data.mean, data.stddev);

  // if not to much changes then the motion is real
  if (data.stddev[0] < max_deviation) 
  {
    // loop over image and detect changes
    for (int j = 0; j < rows; j+=1)   // height
    {
      for (int i = 0; i < cols; i+=1) // width
      {
        // check if at pixel (j,i) intensity is equal to 255 this means that the pixel
        // is different in the sequence of images (prev_frame, current_frame, next_frame)
        if ((motion.at<int>(j,i)) == 255) data.numberOfChanges++;        
      }
    }
  }
  data.isMotion = (data.numberOfChanges >= triggerCount);
  return data;
}


int main (int argc, char * const argv[]) 
{
  // Drawing variables for writing to the window
  stringstream drawnStringStream;
  Point textOrg(10, 10);
  
  // Bounding rectangle and contours to visually track target
  Rect boundingR;
  vector<vector<Point> > contours;
  
  // d1 and d2 for calculating the differences
  // result, the result of and operation, calculated on d1 and d2
  // number_of_changes, the amount of changes in the result matrix.
  // color, the color for drawing the rectangle when something has changed.
  Mat prevPrevFrame, prevFrame, currentFrame, capture_frame, resultTracked, display;  
  Mat d1, d2, motion;
  MotionDetectData_t motionDetectData;
  int numberOfSequence = 0;
  unsigned int frameCnt = 0;
  VideoCapture capture_interface;
  int no_capture_cnt=0;


  // Erode kernel
  Mat kernel_ero = getStructuringElement(MORPH_RECT, Size(2, 2));

#define CAMERA

#ifdef CAMERA  

  if(argc > 1)
  {
      // Set up camera for specified device
      capture_interface.open(argv[1]);

      if (capture_interface.isOpened()) { // check if we succeeded
        cout << "Successfully opened input stream " << argv[1] << "!" << endl;
      } else {
        cout << "Failed to open input stream " << argv[1] << "!" << endl;
        exit(EXIT_SUCCESS);
      }
  }
  else
  {
      // Set up camera for default device
      capture_interface.open(DEVICE_ID);

      if (capture_interface.isOpened()) { // check if we succeeded
        cout << "Successfully opened input stream " << DEVICE_ID << "!" << endl;
      } else {
        cout << "Failed to open input stream " << DEVICE_ID << "!" << endl;
        exit(EXIT_SUCCESS);
      }
  }
  

#else

  if(argc > 1)
  {
      capture_interface.open(argv[1]);

      if (capture_interface.isOpened()) { // check if we succeeded
        cout << "Successfully opened Device ID " << argv[1] << "!" << endl;
      } else {
        cout << "Failed to open Device ID " << argv[1] << "!" << endl;
        exit(EXIT_SUCCESS);
      }
  }
  else
  {
      printf("Usage: motion_detect <video source>\n"); exit(-1);
  }

#endif

  // Take image, initialize mats, and convert them to gray
  capture_interface >> capture_frame;

  if(capture_frame.empty()) {cout << "First frame error" << endl; exit(-1);};

  prevPrevFrame = Mat::zeros(capture_frame.size(), capture_frame.type());
  prevFrame = Mat::zeros(capture_frame.size(), capture_frame.type());
  currentFrame = Mat::zeros(capture_frame.size(), capture_frame.type());

#ifdef SHOW_DIFF
  display = Mat::zeros(Size(capture_frame.cols * 2, capture_frame.rows * 2), capture_frame.type());
#else
  display = Mat::zeros(Size(capture_frame.cols * 2, capture_frame.rows * 1), capture_frame.type());
#endif
  
  capture_frame.copyTo(prevFrame);
  capture_frame.copyTo(currentFrame);
  
  cvtColor(prevFrame, prevFrame, CV_RGB2GRAY);
  cvtColor(currentFrame, currentFrame, CV_RGB2GRAY);
  
  cout << "Image Capture Resolution: " << currentFrame.cols << "x" << currentFrame.rows << endl;

  // Setup display window	
  namedWindow(WINDOW_NAME, WINDOW_AUTOSIZE | CV_GUI_NORMAL); 
  createTrackbar("Threshold:", WINDOW_NAME, &currentThreshold, MAX_THRESHOLD, NULL);
  createTrackbar("Max Deviation:", WINDOW_NAME, &currentDeviation, MAX_DEVIATION, NULL);
  createTrackbar("Pixels Changed:", WINDOW_NAME, &currentMotionTrigger, currentFrame.cols * currentFrame.rows, NULL);	
  
  cvWaitKey (DELAY_IN_MSEC);

  // All settings have been set, now go in endless frame acquisition loop
  while (running)
  {
    // Take a new image
    capture_interface >> capture_frame;
    if(capture_frame.empty()) 
    {
      no_capture_cnt++;

      if(no_capture_cnt < 10)
        continue;
      else
        break; 
    }

    prevFrame.copyTo(prevPrevFrame);
    currentFrame.copyTo(prevFrame);
    capture_frame.copyTo(currentFrame);
    
    cvtColor(currentFrame, currentFrame, CV_RGB2GRAY);
    
    // Calc differences between the images and do AND-operation threshold image
    absdiff(prevPrevFrame, currentFrame, d1);
    absdiff(prevFrame, currentFrame, d2);
    bitwise_and(d1, d2, motion);
    threshold(motion, motion, currentThreshold, 255, CV_THRESH_BINARY);
    erode(motion, motion, kernel_ero);
    
    motionDetectData = detectMotion(motion(Rect(CAM_WIDTH_OFFSET, 0, capture_frame.cols-(CAM_WIDTH_OFFSET * 2), capture_frame.rows)), currentDeviation, currentMotionTrigger);

    /* 
    * I think it's self-descriptive. We pick 4 different ROIs and copy
    * the frames we need into them. Piece of cake!
    */
    display = Scalar(0, 0, 0);
    
    capture_frame.copyTo(resultTracked);
    if (motionDetectData.isMotion) 
    {
      findContours(motion, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));
      for( int i = 0; i < contours.size(); i++ ) 
      {
         boundingR = boundingRect(contours[i]);
         rectangle(resultTracked, boundingR.tl(), boundingR.br(), Scalar(0, 255, 0), 2, CV_AA , 0);
       }
    }
    
#ifdef SHOW_DIFF
    cvtColor(d1, d1, CV_GRAY2RGB);
    d1.copyTo(display(Rect(capture_frame.cols * 0, capture_frame.rows * 0, capture_frame.cols, capture_frame.rows)));
    cvtColor(d2, d2, CV_GRAY2RGB);
    d2.copyTo(display(Rect(capture_frame.cols * 1, capture_frame.rows * 0, capture_frame.cols, capture_frame.rows)));
    cvtColor(motion, motion, CV_GRAY2RGB);
    motion.copyTo(display(Rect(capture_frame.cols * 0, capture_frame.rows * 1, capture_frame.cols, capture_frame.rows)));
    resultTracked.copyTo(display(Rect(capture_frame.cols * 1, capture_frame.rows * 1, capture_frame.cols, capture_frame.rows)));
#else
    cvtColor(motion, motion, CV_GRAY2RGB);
    motion.copyTo(display(Rect(capture_frame.cols * 0, capture_frame.rows * 0, capture_frame.cols, capture_frame.rows)));
    resultTracked.copyTo(display(Rect(capture_frame.cols * 1, capture_frame.rows * 0, capture_frame.cols, capture_frame.rows)));
#endif

    drawnStringStream.str("");
    drawnStringStream << "Mean: " << motionDetectData.mean[0];
    textOrg.x = 10;
    textOrg.y = 20;
    putText(display, drawnStringStream.str(), textOrg, FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar::all(255), 2, 8);
    
    drawnStringStream.str("");
    drawnStringStream << "Deviation: " << motionDetectData.stddev[0];
    textOrg.x = 10;
    textOrg.y = 50;
    putText(display, drawnStringStream.str(), textOrg, FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar::all(255), 2, 8);
    
    drawnStringStream.str("");
    drawnStringStream << "Changes: " << motionDetectData.numberOfChanges;		
    textOrg.x = 10;
    textOrg.y = 80;
    putText(display, drawnStringStream.str(), textOrg, FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar::all(255), 2, 8);

    // save detected frames
    if (motionDetectData.isMotion) 
    {
      //if (numberOfSequence > 0) saveImg(capture_frame, DETECT_DIRECTORY, EXTENSION, FILE_FORMAT, numberOfSequence, 1);
      //saveImg(capture_frame, DETECT_DIRECTORY, EXTENSION, FILE_FORMAT, frameCnt, 1);
      numberOfSequence++;
    } 
    else
    {
      numberOfSequence = 0;
    }

    // save every frame to compare detected frames to
    if((frameCnt % COLLECT_EVERY_NTH_FRAME) ==  0)
    {
      //saveImg(capture_frame, COLLECT_DIRECTORY, EXTENSION, FILE_FORMAT, frameCnt, 0);
    }
    
    imshow(WINDOW_NAME, display);
    frameCnt++;
    cvWaitKey (DELAY_IN_MSEC);
  }
  
  capture_interface.release();

  destroyAllWindows();
  
  return 0;
}
