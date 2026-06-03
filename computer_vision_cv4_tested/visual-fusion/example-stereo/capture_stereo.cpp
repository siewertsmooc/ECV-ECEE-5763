/*
 * capture_stereo.cpp
 * OpenCV 4.11+ compatible stereo capture sample
 *
 * Typical updates from older OpenCV 2/3 code:
 * - CV_CAP_PROP_*      -> cv::CAP_PROP_*
 * - CV_FOURCC(...)     -> cv::VideoWriter::fourcc(...)
 * - CV_WINDOW_AUTOSIZE -> cv::WINDOW_AUTOSIZE
 * - old C API constants removed
 *
 * Keys:
 *   q / ESC : quit
 *   s       : save left/right pair
 */

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>

using namespace cv;
using namespace std;

static string makeName(const string& prefix, int index, const string& ext = ".png")
{
    ostringstream oss;
    oss << prefix << "_" << setfill('0') << setw(4) << index << ext;
    return oss.str();
}

int main(int argc, char** argv)
{
    int leftCam  = 0;
    int rightCam = 2;
    int width    = 640;
    int height   = 480;
    double fps   = 30.0;

    if (argc >= 3)
    {
        leftCam  = atoi(argv[1]);
        rightCam = atoi(argv[2]);
    }

    VideoCapture capL(leftCam, CAP_ANY);
    VideoCapture capR(rightCam, CAP_ANY);

    if (!capL.isOpened())
    {
        cerr << "ERROR: Could not open left camera index " << leftCam << endl;
        return -1;
    }

    if (!capR.isOpened())
    {
        cerr << "ERROR: Could not open right camera index " << rightCam << endl;
        return -1;
    }

    capL.set(CAP_PROP_FRAME_WIDTH,  width);
    capL.set(CAP_PROP_FRAME_HEIGHT, height);
    capL.set(CAP_PROP_FPS,          fps);

    capR.set(CAP_PROP_FRAME_WIDTH,  width);
    capR.set(CAP_PROP_FRAME_HEIGHT, height);
    capR.set(CAP_PROP_FPS,          fps);

    width  = static_cast<int>(capL.get(CAP_PROP_FRAME_WIDTH));
    height = static_cast<int>(capL.get(CAP_PROP_FRAME_HEIGHT));
    fps    = capL.get(CAP_PROP_FPS);

    cout << "Left camera  : " << leftCam  << '\n'
         << "Right camera : " << rightCam << '\n'
         << "Resolution   : " << width << " x " << height << '\n'
         << "FPS          : " << fps << '\n'
         << "Press 's' to save stereo pair, 'q' or ESC to quit.\n";

    namedWindow("left", WINDOW_AUTOSIZE);
    namedWindow("right", WINDOW_AUTOSIZE);
    namedWindow("stereo", WINDOW_NORMAL);

    int savedCount = 0;

    for (;;)
    {
        Mat frameL, frameR;
        if (!capL.read(frameL))
        {
            cerr << "ERROR: Failed to read frame from left camera.\n";
            break;
        }

        if (!capR.read(frameR))
        {
            cerr << "ERROR: Failed to read frame from right camera.\n";
            break;
        }

        if (frameL.empty() || frameR.empty())
        {
            cerr << "ERROR: Empty frame received.\n";
            break;
        }

        Mat stereoView;
        hconcat(frameL, frameR, stereoView);

        imshow("left", frameL);
        imshow("right", frameR);
        imshow("stereo", stereoView);

        const int key = waitKey(1) & 0xFF;

        if (key == 27 || key == 'q')
        {
            break;
        }
        else if (key == 's')
        {
            const string leftName  = makeName("left", savedCount);
            const string rightName = makeName("right", savedCount);

            if (!imwrite(leftName, frameL))
            {
                cerr << "ERROR: Failed to save " << leftName << '\n';
            }
            else if (!imwrite(rightName, frameR))
            {
                cerr << "ERROR: Failed to save " << rightName << '\n';
            }
            else
            {
                cout << "Saved pair " << savedCount
                     << ": " << leftName << ", " << rightName << '\n';
                savedCount++;
            }
        }
    }

    capL.release();
    capR.release();
    destroyAllWindows();
    return 0;
}
