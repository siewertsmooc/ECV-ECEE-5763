// rgb2cam.cpp
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

static bool stitchSideBySide(const Mat& leftImg, const Mat& rightImg, Mat& stitchedOut)
{
    if (leftImg.empty() || rightImg.empty())
        return false;

    Mat left = leftImg;
    Mat right = rightImg;

    // Make heights match before concatenation
    if (left.rows != right.rows)
    {
        double scale = static_cast<double>(left.rows) / static_cast<double>(right.rows);
        int newWidth = static_cast<int>(right.cols * scale);
        resize(rightImg, right, Size(newWidth, left.rows), 0, 0, INTER_LINEAR);
    }

    hconcat(left, right, stitchedOut);
    return !stitchedOut.empty();
}




int main(int argc, char** argv)
{
    int camLeftIndex = 0;
    int camRightIndex = 1;

    if (argc >= 3)
    {
        camLeftIndex = atoi(argv[1]);
        camRightIndex = atoi(argv[2]);
    }

    VideoCapture capLeft(camLeftIndex, CAP_ANY);
    VideoCapture capRight(camRightIndex, CAP_ANY);

    if (!capLeft.isOpened() || !capRight.isOpened())
    {
        cerr << "ERROR: Could not open cameras "
             << camLeftIndex << " and/or " << camRightIndex << "\n";
        return 1;
    }

    // Try to keep both cameras at the same resolution
    capLeft.set(CAP_PROP_FRAME_WIDTH, 640);
    capLeft.set(CAP_PROP_FRAME_HEIGHT, 480);
    capRight.set(CAP_PROP_FRAME_WIDTH, 640);
    capRight.set(CAP_PROP_FRAME_HEIGHT, 480);

    cout << "Press 'q' or ESC to quit.\n";
    cout << "Press 's' to save the stitched image.\n";

    namedWindow("Left", WINDOW_AUTOSIZE);
    namedWindow("Right", WINDOW_AUTOSIZE);
    namedWindow("Stitched", WINDOW_AUTOSIZE);

    Mat frameLeft, frameRight, stitched;
    int saveCount = 0;

    while (true)
    {
        // Better timing for two cameras than separate >>
        if (!capLeft.grab() || !capRight.grab())
            break;
        if (!capLeft.retrieve(frameLeft) || !capRight.retrieve(frameRight))
            break;

        if (frameLeft.empty() || frameRight.empty())
            break;

        bool ok = stitchSideBySide(frameLeft, frameRight, stitched);

        imshow("Left", frameLeft);
        imshow("Right", frameRight);

        if (ok)
        {
            putText(stitched, "Left | Right",
                    Point(20, 30), FONT_HERSHEY_SIMPLEX, 0.9,
                    Scalar(0, 255, 0), 2, LINE_AA);
            imshow("Stitched", stitched);
        }
        else
        {
            Mat errorFrame(200, 600, CV_8UC3, Scalar(0, 0, 0));
            putText(errorFrame, "Failed to stitch frames",
                    Point(30, 100), FONT_HERSHEY_SIMPLEX, 0.9,
                    Scalar(0, 0, 255), 2, LINE_AA);
            imshow("Stitched", errorFrame);
        }

        int key = waitKey(1) & 0xFF;
        if (key == 27 || key == 'q')
        {
            break;
        }
        else if (key == 's' && ok)
        {
            string filename = "stitched_" + to_string(saveCount++) + ".png";
            if (imwrite(filename, stitched))
                cout << "Saved " << filename << "\n";
            else
                cerr << "ERROR: Failed to save stitched image\n";
        }
    }

    capLeft.release();
    capRight.release();
    destroyAllWindows();
    return 0;
}
