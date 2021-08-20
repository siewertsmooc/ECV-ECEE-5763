#include "opencv2/opencv.hpp"
#include <iostream>

using namespace std;
using namespace cv;

#define DEVICE_ID 0

//#define CLIP_RIGHT 0
#define CLIP_RIGHT 384
#define CLIP_LEFT 0
//#define CLIP_LEFT 384
#define CLIP_TOP 0
#define CLIP_BOTTOM 0

int main(int argc, char * const argv[]) {
  VideoCapture cap;
  cv::Rect roi;
  Mat frame;

  // Create VideoCapture
  if(argc > 1)
  {
    cap.open(argv[1]);
  }
  else
  {
    cap.open(DEVICE_ID);
  }

  if(!cap.isOpened()) {
    cout << "Error opening stream" << endl;
    exit(-1);
  }

  int frame_width = int(cap.get(CV_CAP_PROP_FRAME_WIDTH));
  int frame_height = int(cap.get(CV_CAP_PROP_FRAME_HEIGHT));

  frame_width = frame_width - CLIP_LEFT - CLIP_RIGHT;
  frame_height = frame_height - CLIP_TOP - CLIP_BOTTOM;

  VideoWriter video("newvideo.avi", CV_FOURCC('M','J','P','G'),30, Size(frame_width, frame_height));


  while(1) {

    cap >> frame;

    if(frame.empty()) break;

    if( (CLIP_LEFT+CLIP_RIGHT > 0) || (CLIP_TOP+CLIP_BOTTOM > 0) )
    {
      roi.x = CLIP_LEFT;
      roi.y = CLIP_TOP;
      roi.width = frame.size().width - CLIP_LEFT - CLIP_RIGHT; 
      roi.height = frame.size().height - CLIP_TOP - CLIP_BOTTOM; 

      frame = frame(roi);
    }

    imshow("Frame", frame);

    video.write(frame);

    char c=(char)waitKey(25);
    if(c==27) break;
  }

  video.release();

  cap.release();

  destroyAllWindows();
 
  return 0;
}
