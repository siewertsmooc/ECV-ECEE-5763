#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>

using namespace cv;
using namespace std;

static float medianOfVector(vector<float>& v)
{
    if (v.empty()) return 0.0f;
    size_t n = v.size() / 2;
    nth_element(v.begin(), v.begin() + n, v.end());
    float med = v[n];
    if (v.size() % 2 == 0)
    {
        nth_element(v.begin(), v.begin() + n - 1, v.end());
        med = 0.5f * (med + v[n - 1]);
    }
    return med;
}

static bool estimateRobustShiftSIFT(const Mat& imgLeft,
                                    const Mat& imgRight,
                                    int& dx,
                                    int& dy,
                                    Mat& debugOut)
{
    dx = 0;
    dy = 0;
    debugOut.release();

    if (imgLeft.empty() || imgRight.empty())
        return false;

    Mat left = imgLeft.clone();
    Mat right = imgRight.clone();

    if (left.rows != right.rows)
    {
        double scale = static_cast<double>(left.rows) / right.rows;
        int newWidth = static_cast<int>(right.cols * scale);
        resize(right, right, Size(newWidth, left.rows), 0, 0, INTER_LINEAR);
    }

    Mat grayLeft, grayRight;
    if (left.channels() == 3) cvtColor(left, grayLeft, COLOR_BGR2GRAY);
    else grayLeft = left.clone();

    if (right.channels() == 3) cvtColor(right, grayRight, COLOR_BGR2GRAY);
    else grayRight = right.clone();

    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8, 8));
    clahe->apply(grayLeft, grayLeft);
    clahe->apply(grayRight, grayRight);

    GaussianBlur(grayLeft, grayLeft, Size(3, 3), 0);
    GaussianBlur(grayRight, grayRight, Size(3, 3), 0);

    Ptr<SIFT> sift = SIFT::create(
        2500,   // nfeatures
        3,      // nOctaveLayers
        0.02,   // contrastThreshold
        12.0,   // edgeThreshold
        1.6     // sigma
    );

    vector<KeyPoint> kpsLeft, kpsRight;
    Mat desLeft, desRight;
    sift->detectAndCompute(grayLeft, noArray(), kpsLeft, desLeft);
    sift->detectAndCompute(grayRight, noArray(), kpsRight, desRight);

    if (desLeft.empty() || desRight.empty() || kpsLeft.size() < 20 || kpsRight.size() < 20)
        return false;

    if (desLeft.type() != CV_32F) desLeft.convertTo(desLeft, CV_32F);
    if (desRight.type() != CV_32F) desRight.convertTo(desRight, CV_32F);

    FlannBasedMatcher matcher(
        makePtr<flann::KDTreeIndexParams>(5),
        makePtr<flann::SearchParams>(64)
    );

    vector<vector<DMatch>> knnRL, knnLR;
    matcher.knnMatch(desRight, desLeft, knnRL, 2); // right -> left
    matcher.knnMatch(desLeft, desRight, knnLR, 2); // left -> right

    const float ratioThresh = 0.80f;

    vector<DMatch> goodRL, goodLR;
    for (const auto& m : knnRL)
    {
        if (m.size() >= 2 && m[0].distance < ratioThresh * m[1].distance)
            goodRL.push_back(m[0]);
    }

    for (const auto& m : knnLR)
    {
        if (m.size() >= 2 && m[0].distance < ratioThresh * m[1].distance)
            goodLR.push_back(m[0]);
    }

    vector<DMatch> goodMatches;
    for (const auto& a : goodRL)
    {
        for (const auto& b : goodLR)
        {
            if (a.queryIdx == b.trainIdx && a.trainIdx == b.queryIdx)
            {
                goodMatches.push_back(a);
                break;
            }
        }
    }

    if (goodMatches.size() < 12)
        return false;

    vector<float> shiftsX, shiftsY;
    vector<DMatch> filteredMatches;

    for (const auto& m : goodMatches)
    {
        Point2f pr = kpsRight[m.queryIdx].pt;
        Point2f pl = kpsLeft[m.trainIdx].pt;

        float sx = pl.x - pr.x;
        float sy = pl.y - pr.y;

        // Stereo rig prior: mostly horizontal displacement, small vertical drift
        if (std::abs(sy) < 80.0f)
        {
            shiftsX.push_back(sx);
            shiftsY.push_back(sy);
            filteredMatches.push_back(m);
        }
    }

    if (filteredMatches.size() < 10)
        return false;

    float medX = medianOfVector(shiftsX);
    float medY = medianOfVector(shiftsY);

    // Second-stage inlier selection around median shift
    vector<DMatch> inlierMatches;
    vector<float> inlierX, inlierY;

    for (const auto& m : filteredMatches)
    {
        Point2f pr = kpsRight[m.queryIdx].pt;
        Point2f pl = kpsLeft[m.trainIdx].pt;

        float sx = pl.x - pr.x;
        float sy = pl.y - pr.y;

        if (std::abs(sx - medX) < 30.0f && std::abs(sy - medY) < 20.0f)
        {
            inlierMatches.push_back(m);
            inlierX.push_back(sx);
            inlierY.push_back(sy);
        }
    }

    if (inlierMatches.size() < 8)
        return false;

    float finalDx = medianOfVector(inlierX);
    float finalDy = medianOfVector(inlierY);

    dx = cvRound(finalDx);
    dy = cvRound(finalDy);

    vector<char> mask(inlierMatches.size(), 1);
    drawMatches(
        right, kpsRight,
        left, kpsLeft,
        inlierMatches,
        debugOut,
        Scalar::all(-1),
        Scalar::all(-1),
        mask,
        DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
    );

    putText(debugOut,
            "Shift dx=" + to_string(dx) + " dy=" + to_string(dy),
            Point(20, 30),
            FONT_HERSHEY_SIMPLEX, 0.8,
            Scalar(0, 255, 0), 2, LINE_AA);

    return true;
}

static bool makeShiftMosaic(const Mat& imgLeft,
                            const Mat& imgRight,
                            int dx,
                            int dy,
                            Mat& mosaicOut)
{
    mosaicOut.release();

    if (imgLeft.empty() || imgRight.empty())
        return false;

    Mat left = imgLeft.clone();
    Mat right = imgRight.clone();

    if (left.rows != right.rows)
    {
        double scale = static_cast<double>(left.rows) / right.rows;
        int newWidth = static_cast<int>(right.cols * scale);
        resize(right, right, Size(newWidth, left.rows), 0, 0, INTER_LINEAR);
    }

    int wL = left.cols, hL = left.rows;
    int wR = right.cols, hR = right.rows;

    // right image placed relative to left image
    int xR = dx;
    int yR = dy;

    int xmin = min(0, xR);
    int ymin = min(0, yR);
    int xmax = max(wL, xR + wR);
    int ymax = max(hL, yR + hR);

    int outW = xmax - xmin;
    int outH = ymax - ymin;

    if (outW <= 0 || outH <= 0)
        return false;

    int tx = -xmin;
    int ty = -ymin;

    Rect roiLeft(tx, ty, wL, hL);
    Rect roiRight(xR + tx, yR + ty, wR, hR);

    Mat canvasLeft(outH, outW, CV_8UC3, Scalar::all(0));
    Mat canvasRight(outH, outW, CV_8UC3, Scalar::all(0));
    Mat maskLeft(outH, outW, CV_8U, Scalar(0));
    Mat maskRight(outH, outW, CV_8U, Scalar(0));

    left.copyTo(canvasLeft(roiLeft));
    right.copyTo(canvasRight(roiRight));

    rectangle(maskLeft, roiLeft, Scalar(255), FILLED);
    rectangle(maskRight, roiRight, Scalar(255), FILLED);

    Mat overlapMask;
    bitwise_and(maskLeft, maskRight, overlapMask);

    // Blend weights based on horizontal seam in overlap
    Mat fused = canvasLeft.clone();

    // Non-overlap: direct copy
    Mat rightOnly;
    Mat notLeft;
    bitwise_not(maskLeft, notLeft);
    bitwise_and(maskRight, notLeft, rightOnly);
    canvasRight.copyTo(fused, rightOnly);

    vector<Point> overlapPts;
    findNonZero(overlapMask, overlapPts);

    if (!overlapPts.empty())
    {
        Rect overlapROI = boundingRect(overlapPts);

        for (int y = overlapROI.y; y < overlapROI.y + overlapROI.height; ++y)
        {
            for (int x = overlapROI.x; x < overlapROI.x + overlapROI.width; ++x)
            {
                if (overlapMask.at<uchar>(y, x) == 0)
                    continue;

                float alpha;
                if (overlapROI.width <= 1)
                    alpha = 0.5f;
                else
                    alpha = float(x - overlapROI.x) / float(overlapROI.width - 1);

                Vec3b cl = canvasLeft.at<Vec3b>(y, x);
                Vec3b cr = canvasRight.at<Vec3b>(y, x);
                Vec3b out;

                for (int c = 0; c < 3; ++c)
                    out[c] = saturate_cast<uchar>((1.0f - alpha) * cl[c] + alpha * cr[c]);

                fused.at<Vec3b>(y, x) = out;
            }
        }
    }

    // Crop to valid content
    Mat validMask;
    bitwise_or(maskLeft, maskRight, validMask);

    vector<Point> nonZeroPts;
    findNonZero(validMask, nonZeroPts);
    if (!nonZeroPts.empty())
    {
        Rect roi = boundingRect(nonZeroPts);
        mosaicOut = fused(roi).clone();
    }
    else
    {
        mosaicOut = fused;
    }

    putText(mosaicOut,
            "Shift mosaic",
            Point(20, 30),
            FONT_HERSHEY_SIMPLEX,
            0.8,
            Scalar(0, 255, 0),
            2,
            LINE_AA);

    return !mosaicOut.empty();
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
    capLeft.set(CAP_PROP_FRAME_HEIGHT, 360);
    capRight.set(CAP_PROP_FRAME_WIDTH, 640);
    capRight.set(CAP_PROP_FRAME_HEIGHT, 360);

    cout << "Press 'q' or ESC to quit.\n";
    cout << "Press 's' to save the fused mosaic.\n";
    cout << "Press 'd' to toggle match debug view.\n";

    namedWindow("Left", WINDOW_AUTOSIZE);
    namedWindow("Right", WINDOW_AUTOSIZE);
    namedWindow("Fused mosaic", WINDOW_AUTOSIZE);

    bool showDebug = false;
    Mat frameLeft, frameRight, fused, debugMatches;
    int saveCount = 0;

    // temporal smoothing for less jitter
    bool haveShift = false;
    float smoothDx = 0.0f, smoothDy = 0.0f;

    while (true)
    {
        if (!capLeft.grab() || !capRight.grab())
            break;
        if (!capLeft.retrieve(frameLeft) || !capRight.retrieve(frameRight))
            break;
        if (frameLeft.empty() || frameRight.empty())
            break;

        int dx = 0, dy = 0;
        bool gotShift = estimateRobustShiftSIFT(frameLeft, frameRight, dx, dy, debugMatches);

        if (gotShift)
        {
            if (!haveShift)
            {
                smoothDx = (float)dx;
                smoothDy = (float)dy;
                haveShift = true;
            }
            else
            {
                smoothDx = 0.85f * smoothDx + 0.15f * dx;
                smoothDy = 0.85f * smoothDy + 0.15f * dy;
            }
        }

        bool ok = false;
        if (haveShift)
            ok = makeShiftMosaic(frameLeft, frameRight, cvRound(smoothDx), cvRound(smoothDy), fused);

        imshow("Left", frameLeft);
        imshow("Right", frameRight);

        if (ok)
        {
            putText(fused,
                    "dx=" + to_string(cvRound(smoothDx)) + " dy=" + to_string(cvRound(smoothDy)),
                    Point(20, 60),
                    FONT_HERSHEY_SIMPLEX, 0.8,
                    Scalar(255, 255, 0), 2, LINE_AA);

            imshow("Fused mosaic", fused);

            if (showDebug && !debugMatches.empty())
                imshow("SIFT matches", debugMatches);
        }
        else
        {
            Mat fallback = makeSideBySideFallback(
                frameLeft, frameRight,
                "Fusion not stable yet - show textured overlap"
            );
            imshow("Fused mosaic", fallback);

            if (showDebug)
            {
                Mat msg(240, 700, CV_8UC3, Scalar::all(0));
                putText(msg, "No stable shift estimate this frame",
                        Point(20, 120), FONT_HERSHEY_SIMPLEX, 0.8,
                        Scalar(0, 0, 255), 2, LINE_AA);
                imshow("SIFT matches", msg);
            }
        }

        int key = waitKey(1) & 0xFF;

        if (key == 27 || key == 'q' || key == 'Q')
        {
            break;
        }
        else if ((key == 's' || key == 'S') && ok)
        {
            string filename = "shift_mosaic_" + to_string(saveCount++) + ".png";
            if (imwrite(filename, fused))
                cout << "Saved " << filename << "\n";
            else
                cerr << "ERROR: Failed to save fused mosaic\n";
        }
        else if (key == 'd' || key == 'D')
        {
            showDebug = !showDebug;
            if (!showDebug)
                destroyWindow("SIFT matches");
        }
    }

    capLeft.release();
    capRight.release();
    destroyAllWindows();
    return 0;
}
