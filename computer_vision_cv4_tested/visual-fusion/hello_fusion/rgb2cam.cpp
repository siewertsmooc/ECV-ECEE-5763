#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

using namespace cv;
using namespace std;

static bool fuseTwoFramesORB(const Mat& imgLeft, const Mat& imgRight, Mat& fusedOut, Mat& debugOut)
{
    fusedOut.release();
    debugOut.release();

    if (imgLeft.empty() || imgRight.empty())
        return false;

    Mat left = imgLeft;
    Mat right = imgRight;

    if (left.rows != right.rows)
    {
        double scale = static_cast<double>(left.rows) / static_cast<double>(right.rows);
        int newWidth = static_cast<int>(right.cols * scale);
        resize(right, right, Size(newWidth, left.rows), 0, 0, INTER_LINEAR);
    }

    Mat grayLeft, grayRight;
    if (left.channels() == 3) cvtColor(left, grayLeft, COLOR_BGR2GRAY);
    else grayLeft = left.clone();

    if (right.channels() == 3) cvtColor(right, grayRight, COLOR_BGR2GRAY);
    else grayRight = right.clone();

    GaussianBlur(grayLeft, grayLeft, Size(3, 3), 0);
    GaussianBlur(grayRight, grayRight, Size(3, 3), 0);

    Ptr<ORB> orb = ORB::create(
        4000,
        1.2f,
        8,
        31,
        0,
        2,
        ORB::HARRIS_SCORE,
        31,
        20
    );

    vector<KeyPoint> kpsLeft, kpsRight;
    Mat desLeft, desRight;
    orb->detectAndCompute(grayLeft, noArray(), kpsLeft, desLeft);
    orb->detectAndCompute(grayRight, noArray(), kpsRight, desRight);

    if (desLeft.empty() || desRight.empty() || kpsLeft.size() < 20 || kpsRight.size() < 20)
        return false;

    BFMatcher matcher(NORM_HAMMING, false);
    vector<vector<DMatch>> knnMatches;
    matcher.knnMatch(desRight, desLeft, knnMatches, 2);

    vector<DMatch> goodMatches;
    goodMatches.reserve(knnMatches.size());

    const float ratioThresh = 0.75f;
    for (const auto& m : knnMatches)
    {
        if (m.size() < 2) continue;
        if (m[0].distance < ratioThresh * m[1].distance)
            goodMatches.push_back(m[0]);
    }

    if (goodMatches.size() < 20)
        return false;

    vector<Point2f> ptsRight, ptsLeft;
    ptsRight.reserve(goodMatches.size());
    ptsLeft.reserve(goodMatches.size());

    for (const auto& m : goodMatches)
    {
        ptsRight.push_back(kpsRight[m.queryIdx].pt);
        ptsLeft.push_back(kpsLeft[m.trainIdx].pt);
    }

    Mat inlierMask;
    Mat H = findHomography(ptsRight, ptsLeft, RANSAC, 4.0, inlierMask);
    if (H.empty())
        return false;

    int inlierCount = 0;
    for (int i = 0; i < inlierMask.rows; ++i)
        if (inlierMask.at<uchar>(i, 0)) ++inlierCount;

    if (inlierCount < 15)
        return false;

    drawMatches(
        right, kpsRight,
        left, kpsLeft,
        goodMatches,
        debugOut,
        Scalar::all(-1),
        Scalar::all(-1),
        inlierMask,
        DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
    );

    const int wL = left.cols,  hL = left.rows;
    const int wR = right.cols, hR = right.rows;

    vector<Point2f> cornersRight = {
        Point2f(0.f, 0.f),
        Point2f(static_cast<float>(wR), 0.f),
        Point2f(static_cast<float>(wR), static_cast<float>(hR)),
        Point2f(0.f, static_cast<float>(hR))
    };

    vector<Point2f> warpedCornersRight;
    perspectiveTransform(cornersRight, warpedCornersRight, H);

    float xmin = 0.f, ymin = 0.f, xmax = static_cast<float>(wL), ymax = static_cast<float>(hL);

    for (const auto& p : warpedCornersRight)
    {
        xmin = std::min(xmin, p.x);
        ymin = std::min(ymin, p.y);
        xmax = std::max(xmax, p.x);
        ymax = std::max(ymax, p.y);
    }

    int tx = static_cast<int>(-std::floor(xmin));
    int ty = static_cast<int>(-std::floor(ymin));
    int outW = static_cast<int>(std::ceil(xmax - xmin));
    int outH = static_cast<int>(std::ceil(ymax - ymin));

    if (outW <= 0 || outH <= 0 || outW > 8000 || outH > 8000)
        return false;

    Mat T = (Mat_<double>(3, 3) <<
             1, 0, tx,
             0, 1, ty,
             0, 0, 1);

    Mat Ht = T * H;

    Mat warpedRight;
    warpPerspective(right, warpedRight, Ht, Size(outW, outH), INTER_LINEAR, BORDER_CONSTANT);

    Mat canvasLeft(outH, outW, left.type(), Scalar::all(0));
    left.copyTo(canvasLeft(Rect(tx, ty, wL, hL)));

    Mat maskLeft(outH, outW, CV_8U, Scalar(0));
    rectangle(maskLeft, Rect(tx, ty, wL, hL), Scalar(255), FILLED);

    Mat maskRight;
    cvtColor(warpedRight, maskRight, COLOR_BGR2GRAY);
    threshold(maskRight, maskRight, 0, 255, THRESH_BINARY);

    Mat invLeft, invRight;
    bitwise_not(maskLeft, invLeft);
    bitwise_not(maskRight, invRight);

    Mat distLeft, distRight;
    distanceTransform(invLeft, distLeft, DIST_L2, 3);
    distanceTransform(invRight, distRight, DIST_L2, 3);

    Mat validLeft, validRight;
    compare(maskLeft, 0, validLeft, CMP_GT);
    compare(maskRight, 0, validRight, CMP_GT);

    distLeft += 1.0f;
    distRight += 1.0f;

    distLeft.setTo(0.0f, ~validLeft);
    distRight.setTo(0.0f, ~validRight);

    Mat denom = distLeft + distRight + 1e-6f;
    Mat wLeft, wRight;
    divide(distLeft, denom, wLeft);
    divide(distRight, denom, wRight);

    Mat onlyLeft, onlyRight;
    Mat notValidRight, notValidLeft;
    bitwise_not(validRight, notValidRight);
    bitwise_not(validLeft, notValidLeft);
    bitwise_and(validLeft, notValidRight, onlyLeft);
    bitwise_and(validRight, notValidLeft, onlyRight);

    wLeft.setTo(1.0f, onlyLeft);
    wRight.setTo(1.0f, onlyRight);

    vector<Mat> wlch(3, wLeft), wrch(3, wRight);
    Mat wLeft3, wRight3;
    merge(wlch, wLeft3);
    merge(wrch, wRight3);

    Mat left32f, right32f, fused32f;
    canvasLeft.convertTo(left32f, CV_32FC3);
    warpedRight.convertTo(right32f, CV_32FC3);

    fused32f = left32f.mul(wLeft3) + right32f.mul(wRight3);
    fused32f.convertTo(fusedOut, CV_8UC3);

    Mat fusedGray, fusedMask;
    cvtColor(fusedOut, fusedGray, COLOR_BGR2GRAY);
    threshold(fusedGray, fusedMask, 0, 255, THRESH_BINARY);

    vector<Point> nonZeroPts;
    findNonZero(fusedMask, nonZeroPts);
    if (!nonZeroPts.empty())
    {
        Rect roi = boundingRect(nonZeroPts);
        fusedOut = fusedOut(roi).clone();
    }

    putText(fusedOut,
            "ORB panorama mosaic",
            Point(20, 30),
            FONT_HERSHEY_SIMPLEX,
            0.8,
            Scalar(0, 255, 0),
            2,
            LINE_AA);

    return true;
}

static Mat makeSideBySideFallback(const Mat& a, const Mat& b, const string& msg)
{
    int outH = max(a.rows, b.rows);
    int outW = a.cols + b.cols;

    Mat side(outH, outW, a.type(), Scalar::all(0));
    a.copyTo(side(Rect(0, 0, a.cols, a.rows)));
    b.copyTo(side(Rect(a.cols, 0, b.cols, b.rows)));

    putText(side, msg,
            Point(10, 30),
            FONT_HERSHEY_SIMPLEX,
            0.8,
            Scalar(0, 0, 255),
            2,
            LINE_AA);

    return side;
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
        cerr << "ERROR: Could not open cameras " << camLeftIndex
             << " and/or " << camRightIndex << "\n";
        return 1;
    }

    capLeft.set(CAP_PROP_FRAME_WIDTH, 640);
    capLeft.set(CAP_PROP_FRAME_HEIGHT, 480);
    capRight.set(CAP_PROP_FRAME_WIDTH, 640);
    capRight.set(CAP_PROP_FRAME_HEIGHT, 480);

    cout << "Press 'q' or ESC to quit.\n";
    cout << "Press 's' to save the fused panorama.\n";
    cout << "Press 'd' to toggle match debug view.\n";

    namedWindow("Left", WINDOW_AUTOSIZE);
    namedWindow("Right", WINDOW_AUTOSIZE);
    namedWindow("Fused mosaic", WINDOW_AUTOSIZE);

    bool showDebug = false;
    Mat frameLeft, frameRight, fused, debugMatches;
    int saveCount = 0;

    while (true)
    {
        if (!capLeft.grab() || !capRight.grab())
            break;
        if (!capLeft.retrieve(frameLeft) || !capRight.retrieve(frameRight))
            break;

        if (frameLeft.empty() || frameRight.empty())
            break;

        bool ok = fuseTwoFramesORB(frameLeft, frameRight, fused, debugMatches);

        imshow("Left", frameLeft);
        imshow("Right", frameRight);

        if (ok)
        {
            imshow("Fused mosaic", fused);
            if (showDebug && !debugMatches.empty())
                imshow("ORB matches", debugMatches);
        }
        else
        {
            Mat fallback = makeSideBySideFallback(
                frameLeft, frameRight,
                "Fusion failed: need more overlap / texture / stable frames"
            );
            imshow("Fused mosaic", fallback);

            if (showDebug)
            {
                Mat msg(240, 640, CV_8UC3, Scalar::all(0));
                putText(msg, "No valid ORB homography this frame",
                        Point(20, 120), FONT_HERSHEY_SIMPLEX, 0.8,
                        Scalar(0, 0, 255), 2, LINE_AA);
                imshow("ORB matches", msg);
            }
        }

        int key = waitKey(1) & 0xFF;

        if (key == 27 || key == 'q' || key == 'Q')
        {
            break;
        }
        else if ((key == 's' || key == 'S') && ok)
        {
            string filename = "fused_panorama_" + to_string(saveCount++) + ".png";
            if (imwrite(filename, fused))
                cout << "Saved " << filename << "\n";
            else
                cerr << "ERROR: Failed to save fused panorama\n";
        }
        else if (key == 'd' || key == 'D')
        {
            showDebug = !showDebug;
            if (!showDebug)
                destroyWindow("ORB matches");
        }
    }

    capLeft.release();
    capRight.release();
    destroyAllWindows();
    return 0;
}
