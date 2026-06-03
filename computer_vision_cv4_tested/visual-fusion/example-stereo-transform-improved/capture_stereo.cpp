/*
 *
 *  Example by Sam Siewert  - modified for dual USB Camera capture to use to
 *                            experiment and learn Stereo vision concepts pairing with
 *                            "Learning OpenCV" by Gary Bradski and Adrian Kaehler
 *
 *  Updated for OpenCV 4.11+:
 *  - Replaced removed C API (CvCapture/IplImage/cvQueryFrame/cvShowImage/cvSaveImage)
 *    with VideoCapture/Mat/imshow/imwrite
 *  - Replaced removed opencv2/contrib/contrib.hpp and StereoVar usage
 *    with StereoSGBM
 *  - Replaced CV_WINDOW_AUTOSIZE with WINDOW_AUTOSIZE
 *  - Replaced CV_AA with LINE_AA
 *
 */

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;
using namespace std;

// If you have enough USB 2.0 bandwidth, then run at higher resolution
//#define HRES_COLS (1280)
//#define VRES_ROWS (720)
#define HRES_COLS (640)
#define VRES_ROWS (480)

// Should always work for uncompressed USB 2.0 dual cameras
//#define HRES_COLS (320)
//#define VRES_ROWS (240)

#define ESC_KEY (27)

static string makeSnapshotName(const string& prefix, double timestamp_ms)
{
    char buf[128];
    snprintf(buf, sizeof(buf), "snapshot_%s_%8.4lf.jpg", prefix.c_str(), timestamp_ms);
    return string(buf);
}

static double now_ms(timespec& ts)
{
    clock_gettime(CLOCK_REALTIME, &ts);
    return ((double)ts.tv_sec * 1000.0) + ((double)ts.tv_nsec / 1000000.0);
}

int main(int argc, char** argv)
{
    double prev_frame_time = 0.0, prev_frame_time_l = 0.0, prev_frame_time_r = 0.0;
    double curr_frame_time = 0.0, curr_frame_time_l = 0.0, curr_frame_time_r = 0.0;
    struct timespec frame_time{}, frame_time_l{}, frame_time_r{};

    VideoCapture capture;
    VideoCapture capture_l;
    VideoCapture capture_r;

    Mat frame, frame_l, frame_r;
    int dev = 0, devl = 0, devr = 1;

    int computeDisparity = 0;
    int applyCannyTransform = 0;
    int applyHoughTransform = 0;

    Mat gray_l, canny_frame_l;
    vector<Vec4i> lines_l;
    Mat gray_r, canny_frame_r;
    vector<Vec4i> lines_r;

    Mat disp, disp8;

    // Modern replacement for StereoVar
    int minDisparity = 0;
    int numDisparities = 64;   // must be divisible by 16
    int blockSize = 5;         // must be odd
    Ptr<StereoSGBM> stereo = StereoSGBM::create(
        minDisparity,
        numDisparities,
        blockSize,
        8 * 3 * blockSize * blockSize,
        32 * 3 * blockSize * blockSize,
        1,   // disp12MaxDiff
        63,  // preFilterCap
        10,  // uniquenessRatio
        100, // speckleWindowSize
        32,  // speckleRange
        StereoSGBM::MODE_SGBM
    );

    if (argc == 1)
    {
	printf("Usage: capture_stereo <cam dev 0> <cam dev 2> [d=disparity, h=hough, c=canny]\n");
        //printf("Will open DEFAULT video device video0\n");
        //capture.open(0, CAP_ANY);
        //capture.set(CAP_PROP_FRAME_WIDTH, HRES_COLS);
        //capture.set(CAP_PROP_FRAME_HEIGHT, VRES_ROWS);
    }

    if (argc == 2)
    {
        printf("argv[1]=%s\n", argv[1]);
        sscanf(argv[1], "%d", &dev);
        printf("Will open SINGLE video device %d\n", dev);
        capture.open(dev, CAP_ANY);
        capture.set(CAP_PROP_FRAME_WIDTH, HRES_COLS);
        capture.set(CAP_PROP_FRAME_HEIGHT, VRES_ROWS);
    }

    if (argc >= 3)
    {
        printf("argv[1]=%s, argv[2]=%s\n", argv[1], argv[2]);
        sscanf(argv[1], "%d", &devl);
        sscanf(argv[2], "%d", &devr);
        printf("Will open DUAL video devices %d and %d\n", devl, devr);

        capture_l.open(devl, CAP_ANY);
        capture_r.open(devr, CAP_ANY);

        capture_l.set(CAP_PROP_FRAME_WIDTH, HRES_COLS);
        capture_l.set(CAP_PROP_FRAME_HEIGHT, VRES_ROWS);
        capture_r.set(CAP_PROP_FRAME_WIDTH, HRES_COLS);
        capture_r.set(CAP_PROP_FRAME_HEIGHT, VRES_ROWS);

        if (!capture_l.isOpened() || !capture_r.isOpened())
        {
            cerr << "ERROR: Could not open stereo camera devices " << devl << " and " << devr << endl;
            return -1;
        }

        if (argc == 4)
        {
            if (strncmp(argv[3], "d", 1) == 0)
            {
                computeDisparity = 1;
                namedWindow("Capture LEFT", WINDOW_AUTOSIZE);
                namedWindow("Capture RIGHT", WINDOW_AUTOSIZE);
                namedWindow("Capture DISPARITY", WINDOW_AUTOSIZE);
            }
            else if (strncmp(argv[3], "h", 1) == 0)
            {
                applyHoughTransform = 1;
                namedWindow("Capture LEFT", WINDOW_AUTOSIZE);
                namedWindow("Capture RIGHT", WINDOW_AUTOSIZE);
            }
            else
            {
                applyCannyTransform = 1;
                namedWindow("Capture LEFT", WINDOW_AUTOSIZE);
                namedWindow("Capture RIGHT", WINDOW_AUTOSIZE);
            }
        }
        else
        {
            namedWindow("Capture LEFT", WINDOW_AUTOSIZE);
            namedWindow("Capture RIGHT", WINDOW_AUTOSIZE);
        }

        while (1)
        {
            // Better for stereo timing than read(): grab both first, then retrieve both.
            if (!capture_l.grab() || !capture_r.grab())
                break;
            if (!capture_l.retrieve(frame_l) || !capture_r.retrieve(frame_r))
                break;
            if (frame_l.empty() || frame_r.empty())
                break;

            Mat display_l = frame_l.clone();
            Mat display_r = frame_r.clone();

            if (applyCannyTransform || applyHoughTransform)
            {
                cvtColor(frame_l, gray_l, COLOR_BGR2GRAY);
                cvtColor(frame_r, gray_r, COLOR_BGR2GRAY);

                Canny(gray_l, canny_frame_l, 50, 200, 3);
                Canny(gray_r, canny_frame_r, 50, 200, 3);

                if (applyHoughTransform)
                {
                    lines_l.clear();
                    lines_r.clear();

                    HoughLinesP(canny_frame_l, lines_l, 1, CV_PI / 180, 50, 50, 10);
                    HoughLinesP(canny_frame_r, lines_r, 1, CV_PI / 180, 50, 50, 10);

                    for (size_t i = 0; i < lines_l.size(); i++)
                    {
                        Vec4i l = lines_l[i];
                        line(display_l, Point(l[0], l[1]), Point(l[2], l[3]),
                             Scalar(0, 0, 255), 3, LINE_AA);
                    }

                    for (size_t i = 0; i < lines_r.size(); i++)
                    {
                        Vec4i l = lines_r[i];
                        line(display_r, Point(l[0], l[1]), Point(l[2], l[3]),
                             Scalar(0, 0, 255), 3, LINE_AA);
                    }
                }
            }
            else if (computeDisparity)
            {
                Mat left_gray, right_gray;
                cvtColor(frame_l, left_gray, COLOR_BGR2GRAY);
                cvtColor(frame_r, right_gray, COLOR_BGR2GRAY);

                stereo->compute(left_gray, right_gray, disp);
                disp.convertTo(disp8, CV_8U, 255.0 / (numDisparities * 16.0));
            }

            curr_frame_time_l = now_ms(frame_time_l);
            curr_frame_time_r = now_ms(frame_time_r);

            if (applyCannyTransform)
            {
                imshow("Capture LEFT", canny_frame_l);
                imshow("Capture RIGHT", canny_frame_r);
            }
            else
            {
                imshow("Capture LEFT", display_l);
                imshow("Capture RIGHT", display_r);
            }

            if (computeDisparity)
                imshow("Capture DISPARITY", disp8);

            printf("LEFT dt=%5.2lf msec, RIGHT dt=%5.2lf msec\n",
                   (curr_frame_time_l - prev_frame_time_l),
                   (curr_frame_time_r - prev_frame_time_r));

            char c = (char)(waitKey(10) & 0xFF);
            if (c == ESC_KEY)
            {
                string left_name = makeSnapshotName("left", curr_frame_time_l);
                string right_name = makeSnapshotName("right", curr_frame_time_r);
                imwrite(left_name, frame_l);
                imwrite(right_name, frame_r);
                printf("Saved %s and %s\n", left_name.c_str(), right_name.c_str());
            }
            else if ((c == 'q') || (c == 'Q'))
            {
                printf("Exiting ...\n");
                capture_l.release();
                capture_r.release();
                destroyWindow("Capture LEFT");
                destroyWindow("Capture RIGHT");
                if (computeDisparity) destroyWindow("Capture DISPARITY");
                return 0;
            }

            prev_frame_time_l = curr_frame_time_l;
            prev_frame_time_r = curr_frame_time_r;
        }

        capture_l.release();
        capture_r.release();
        destroyWindow("Capture LEFT");
        destroyWindow("Capture RIGHT");
        if (computeDisparity) destroyWindow("Capture DISPARITY");
    }
    else
    {
        // used to compute running averages for single camera frame rates
        double ave_framedt = 0.0, ave_frame_rate = 0.0, fc = 0.0, framedt = 0.0;
        unsigned int frame_count = 0;

        if (!capture.isOpened())
        {
            cerr << "ERROR: Could not open single camera device " << dev << endl;
            return -1;
        }

        namedWindow("Capture Example", WINDOW_AUTOSIZE);

        while (1)
        {
            if (!capture.read(frame))
                break;

            if (frame.empty())
                break;
            else
            {
                curr_frame_time = now_ms(frame_time);
                frame_count++;

                if (frame_count > 2)
                {
                    fc = (double)frame_count;
                    ave_framedt = ((fc - 1.0) * ave_framedt + framedt) / fc;
                    if (ave_framedt > 0.0)
                        ave_frame_rate = 1.0 / (ave_framedt / 1000.0);
                }
            }

            imshow("Capture Example", frame);
            printf("Frame @ %u sec, %lu nsec, dt=%5.2lf msec, avedt=%5.2lf msec, rate=%5.2lf fps\n",
                   (unsigned)frame_time.tv_sec,
                   (unsigned long)frame_time.tv_nsec,
                   framedt, ave_framedt, ave_frame_rate);

            char c = (char)(waitKey(10) & 0xFF);
            if (c == ESC_KEY)
            {
                string name = makeSnapshotName("single", curr_frame_time);
                imwrite(name, frame);
                printf("Saved %s\n", name.c_str());
            }
            else if ((c == 'q') || (c == 'Q'))
            {
                printf("Exiting ...\n");
                capture.release();
                destroyWindow("Capture Example");
                return 0;
            }

            framedt = curr_frame_time - prev_frame_time;
            prev_frame_time = curr_frame_time;
        }

        capture.release();
        destroyWindow("Capture Example");
    }

    return 0;
}
