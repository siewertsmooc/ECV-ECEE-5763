#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <cfloat>
#include <algorithm>

using namespace cv;
using namespace std;

static int estimateOverlapPixels(const Mat& leftImg,
                                 const Mat& rightImg,
                                 int minOverlap = 20,
                                 int maxOverlap = -1,
                                 int sampleRows = 120)
{
    if (leftImg.empty() || rightImg.empty())
        return 0;

    Mat leftGray, rightGray;

    if (leftImg.channels() == 3)
        cvtColor(leftImg, leftGray, COLOR_BGR2GRAY);
    else
        leftGray = leftImg.clone();

    if (rightImg.channels() == 3)
        cvtColor(rightImg, rightGray, COLOR_BGR2GRAY);
    else
        rightGray = rightImg.clone();

    if (leftGray.rows != rightGray.rows)
    {
        double scale = static_cast<double>(leftGray.rows) / static_cast<double>(rightGray.rows);
        int newWidth = static_cast<int>(rightGray.cols * scale);
        resize(rightGray, rightGray, Size(newWidth, leftGray.rows), 0, 0, INTER_LINEAR);
    }

    GaussianBlur(leftGray, leftGray, Size(5, 5), 0);
    GaussianBlur(rightGray, rightGray, Size(5, 5), 0);

    const int rows = leftGray.rows;
    const int colsL = leftGray.cols;
    const int colsR = rightGray.cols;

    if (colsL < 2 || colsR < 2)
        return 0;

    if (maxOverlap < 0)
        maxOverlap = std::min(colsL, colsR) / 2;

    maxOverlap = std::min(maxOverlap, std::min(colsL - 1, colsR - 1));
    minOverlap = std::max(1, minOverlap);

    if (minOverlap >= maxOverlap)
        return minOverlap;

    const int bandHeight = std::min(sampleRows, rows);
    const int y0 = std::max(0, (rows - bandHeight) / 2);

    Mat leftBand = leftGray(Rect(0, y0, colsL, bandHeight));
    Mat rightBand = rightGray(Rect(0, y0, colsR, bandHeight));

    double bestScore = DBL_MAX;
    int bestOverlap = minOverlap;

    for (int overlap = minOverlap; overlap <= maxOverlap; ++overlap)
    {
        Mat leftStrip  = leftBand(Rect(colsL - overlap, 0, overlap, bandHeight));
        Mat rightStrip = rightBand(Rect(0, 0, overlap, bandHeight));

        Mat diff;
        absdiff(leftStrip, rightStrip, diff);

        Scalar s = sum(diff);
        double score = s[0] / static_cast<double>(overlap * bandHeight);

        if (score < bestScore)
        {
            bestScore = score;
            bestOverlap = overlap;
        }
    }

    return bestOverlap;
}

static bool stitchSideBySideAutoOverlap(const Mat& leftImg,
                                        const Mat& rightImg,
                                        Mat& stitchedOut,
                                        int& overlapPixels)
{
    if (leftImg.empty() || rightImg.empty())
        return false;

    Mat left = leftImg;
    Mat right = rightImg;

    if (left.rows != right.rows)
    {
        double scale = static_cast<double>(left.rows) / static_cast<double>(right.rows);
        int newWidth = static_cast<int>(right.cols * scale);
        resize(right, right, Size(newWidth, left.rows), 0, 0, INTER_LINEAR);
    }

    overlapPixels = estimateOverlapPixels(left, right, 20, std::min(left.cols, right.cols) / 2);
    overlapPixels = std::max(0, std::min(overlapPixels, right.cols - 1));

    Mat rightCropped = right(Rect(overlapPixels, 0,
                                  right.cols - overlapPixels,
                                  right.rows));

    hconcat(left, rightCropped, stitchedOut);

    if (stitchedOut.empty())
        return false;

    putText(stitchedOut,
            "Auto overlap: " + std::to_string(overlapPixels) + " px",
            Point(20, 30),
            FONT_HERSHEY_SIMPLEX,
            0.8,
            Scalar(0, 255, 0),
            2,
            LINE_AA);

    return true;
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

    if (!capLeft.isOpened())
    {
        cerr << "ERROR: Could not open left camera " << camLeftIndex << "\n";
        return 1;
    }

    if (!capRight.isOpened())
    {
        cerr << "ERROR: Could not open right camera " << camRightIndex << "\n";
        return 1;
    }

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
        if (!capLeft.grab() || !capRight.grab())
        {
            cerr << "ERROR: Failed to grab frames.\n";
            break;
        }

        if (!capLeft.retrieve(frameLeft) || !capRight.retrieve(frameRight))
        {
            cerr << "ERROR: Failed to retrieve frames.\n";
            break;
        }

        if (frameLeft.empty() || frameRight.empty())
        {
            cerr << "ERROR: Empty frame received.\n";
            break;
        }

        int overlapPixels = 0;
        bool ok = stitchSideBySideAutoOverlap(frameLeft, frameRight, stitched, overlapPixels);

        imshow("Left", frameLeft);
        imshow("Right", frameRight);

        if (ok)
        {
            imshow("Stitched", stitched);
            cout << "\rEstimated overlap: " << overlapPixels << " px     " << flush;
        }
        else
        {
            Mat errorFrame(200, 600, CV_8UC3, Scalar(0, 0, 0));
            putText(errorFrame,
                    "Failed to stitch frames",
                    Point(30, 100),
                    FONT_HERSHEY_SIMPLEX,
                    0.9,
                    Scalar(0, 0, 255),
                    2,
                    LINE_AA);
            imshow("Stitched", errorFrame);
        }

        int key = waitKey(1) & 0xFF;

        if (key == 27 || key == 'q' || key == 'Q')
        {
            cout << "\nExiting.\n";
            break;
        }
        else if (key == 's' || key == 'S')
        {
            if (ok)
            {
                string filename = "stitched_" + to_string(saveCount++) + ".png";
                if (imwrite(filename, stitched))
                    cout << "\nSaved " << filename << "\n";
                else
                    cerr << "\nERROR: Failed to save stitched image\n";
            }
        }
    }

    capLeft.release();
    capRight.release();
    destroyAllWindows();
    return 0;
}
