// stereocam.cpp
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

using namespace cv;
using namespace std;

static Point g_lastClick(-1, -1);

static void onMouseDisparity(int event, int x, int y, int, void*)
{
    if (event == EVENT_LBUTTONDOWN)
        g_lastClick = Point(x, y);
}

static bool loadStereoCalibration(
    const string& path,
    Mat& K1, Mat& D1, Mat& K2, Mat& D2, Mat& R, Mat& T, Size& imageSize)
{
    FileStorage fs(path, FileStorage::READ);
    if (!fs.isOpened())
    {
        cerr << "Failed to open calibration file: " << path << "\n";
        return false;
    }

    fs["K1"] >> K1; fs["D1"] >> D1;
    fs["K2"] >> K2; fs["D2"] >> D2;
    fs["R"]  >> R;  fs["T"]  >> T;

    int w = 0, h = 0;
    fs["image_width"]  >> w;
    fs["image_height"] >> h;

    if (w <= 0 || h <= 0)
    {
        cerr << "Missing/invalid image_width or image_height in calibration file.\n";
        return false;
    }

    imageSize = Size(w, h);

    if (K1.empty() || D1.empty() || K2.empty() || D2.empty() || R.empty() || T.empty())
    {
        cerr << "Calibration file missing required matrices (K1,D1,K2,D2,R,T).\n";
        return false;
    }

    return true;
}

static Mat colorizeDisparity(const Mat& disp16S, int minDisparity, int numDisparities)
{
    Mat disp8U;
    double scale = 255.0 / (numDisparities * 16.0);
    double shift = -minDisparity * 16.0 * scale;
    disp16S.convertTo(disp8U, CV_8U, scale, shift);

    Mat colored;
    applyColorMap(disp8U, colored, COLORMAP_TURBO);
    return colored;
}

static void drawRectificationOverlay(Mat& visRect, int leftWidth, const Rect& roi1, const Rect& roi2)
{
    for (int y = 0; y < visRect.rows; y += 40)
        line(visRect, Point(0, y), Point(visRect.cols, y), Scalar(0, 255, 0), 1);

    rectangle(visRect, roi1, Scalar(0, 255, 255), 2);

    Rect shiftedROI2 = roi2;
    shiftedROI2.x += leftWidth;
    rectangle(visRect, shiftedROI2, Scalar(255, 255, 0), 2);

    putText(visRect, "ROI1", Point(roi1.x + 5, roi1.y + 20),
            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 2, LINE_AA);
    putText(visRect, "ROI2", Point(shiftedROI2.x + 5, shiftedROI2.y + 20),
            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 0), 2, LINE_AA);
}

static bool makeDebugKeypointMatches(const Mat& rectLGray,
                                     const Mat& rectRGray,
                                     Mat& debugMatchesOut)
{
    debugMatchesOut.release();

    Ptr<ORB> orb = ORB::create(1200);
    vector<KeyPoint> kpsL, kpsR;
    Mat desL, desR;

    orb->detectAndCompute(rectLGray, noArray(), kpsL, desL);
    orb->detectAndCompute(rectRGray, noArray(), kpsR, desR);

    if (desL.empty() || desR.empty() || kpsL.size() < 10 || kpsR.size() < 10)
        return false;

    BFMatcher matcher(NORM_HAMMING, false);
    vector<vector<DMatch>> knnMatches;
    matcher.knnMatch(desL, desR, knnMatches, 2);

    vector<DMatch> goodMatches;
    goodMatches.reserve(knnMatches.size());

    for (const auto& m : knnMatches)
    {
        if (m.size() < 2) continue;
        if (m[0].distance < 0.75f * m[1].distance)
        {
            const Point2f& pl = kpsL[m[0].queryIdx].pt;
            const Point2f& pr = kpsR[m[0].trainIdx].pt;

            // Rectified stereo prior: y should be similar
            if (std::abs(pl.y - pr.y) < 4.0f)
                goodMatches.push_back(m[0]);
        }
    }

    if (goodMatches.size() < 8)
        return false;

    if (goodMatches.size() > 80)
        goodMatches.resize(80);

    drawMatches(rectLGray, kpsL,
                rectRGray, kpsR,
                goodMatches,
                debugMatchesOut,
                Scalar::all(-1),
                Scalar::all(-1),
                vector<char>(),
                DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    putText(debugMatchesOut,
            "ORB rectified matches: " + to_string(goodMatches.size()),
            Point(20, 30),
            FONT_HERSHEY_SIMPLEX, 0.8,
            Scalar(0, 255, 0), 2, LINE_AA);

    return true;
}

int main(int argc, char** argv)
{
    // Usage:
    //   ./stereocam calib.yml [leftIndex] [rightIndex]
    if (argc < 2)
    {
        cerr << "Usage: " << argv[0] << " calib.yml [camLeftIndex] [camRightIndex]\n";
        return 1;
    }

    const string calibPath = argv[1];
    const int camL = (argc >= 3) ? atoi(argv[2]) : 0;
    const int camR = (argc >= 4) ? atoi(argv[3]) : 1;

    Mat K1, D1, K2, D2, R, T;
    Size imageSize;
    if (!loadStereoCalibration(calibPath, K1, D1, K2, D2, R, T, imageSize))
        return 1;

    VideoCapture capL(camL, CAP_ANY);
    VideoCapture capR(camR, CAP_ANY);
    if (!capL.isOpened() || !capR.isOpened())
    {
        cerr << "Could not open cameras " << camL << " and/or " << camR << "\n";
        return 1;
    }

    capL.set(CAP_PROP_FRAME_WIDTH,  imageSize.width);
    capL.set(CAP_PROP_FRAME_HEIGHT, imageSize.height);
    capR.set(CAP_PROP_FRAME_WIDTH,  imageSize.width);
    capR.set(CAP_PROP_FRAME_HEIGHT, imageSize.height);

    Mat R1, R2, P1mat, P2mat, Q;
    Rect roi1, roi2;

    double alpha = 0.0;
    stereoRectify(
        K1, D1, K2, D2, imageSize,
        R, T,
        R1, R2, P1mat, P2mat, Q,
        CALIB_ZERO_DISPARITY,
        alpha,
        imageSize,
        &roi1, &roi2
    );

    Mat map1x, map1y, map2x, map2y;
    initUndistortRectifyMap(K1, D1, R1, P1mat, imageSize, CV_32FC1, map1x, map1y);
    initUndistortRectifyMap(K2, D2, R2, P2mat, imageSize, CV_32FC1, map2x, map2y);

    int minDisparity = 0;
    int numDisparities = 16 * 8;
    int blockSize = 5;

    int cn = 1;
    int sgbmP1 = 8 * cn * blockSize * blockSize;
    int sgbmP2 = 32 * cn * blockSize * blockSize;

    Ptr<StereoSGBM> sgbm = StereoSGBM::create(
        minDisparity,
        numDisparities,
        blockSize,
        sgbmP1,
        sgbmP2,
        1,
        31,
        10,
        100,
        2,
        StereoSGBM::MODE_SGBM_3WAY
    );

    namedWindow("Rectified (L|R)", WINDOW_AUTOSIZE);
    namedWindow("Disparity", WINDOW_AUTOSIZE);
    //namedWindow("Rectified (L|R)", WINDOW_NORMAL);
    //namedWindow("Disparity", WINDOW_NORMAL);

    //resizeWindow("Rectified (L|R)", 640, 480);
    //resizeWindow("Disparity", 640, 480);

    setMouseCallback("Disparity", onMouseDisparity);

    bool showDebugMatches = false;

    cout << "Press q/ESC to quit.\n";
    cout << "Press d to toggle debug keypoint matches.\n";
    cout << "Click on Disparity window to print 3D point and disparity.\n";

    Mat frameL, frameR;
    Mat rectL, rectR, grayL, grayR;
    Mat disp16S, disp16SFiltered, disp32F, dispColor;
    Mat points3D;
    Mat visRect, debugMatches;

    while (true)
    {
        if (!capL.grab() || !capR.grab())
            break;
        if (!capL.retrieve(frameL) || !capR.retrieve(frameR))
            break;
        if (frameL.empty() || frameR.empty())
            break;

        remap(frameL, rectL, map1x, map1y, INTER_LINEAR);
        remap(frameR, rectR, map2x, map2y, INTER_LINEAR);

        cvtColor(rectL, grayL, COLOR_BGR2GRAY);
        cvtColor(rectR, grayR, COLOR_BGR2GRAY);

        sgbm->compute(grayL, grayR, disp16S);

        // Mild post-filter for display stability
        medianBlur(disp16S, disp16SFiltered, 3);

        disp16SFiltered.convertTo(disp32F, CV_32F, 1.0 / 16.0);

        // Mask invalid disparities before reprojection
        Mat invalidMask = (disp32F <= minDisparity);
        reprojectImageTo3D(disp32F, points3D, Q, true);
        points3D.setTo(Scalar(0, 0, 0), invalidMask);

        dispColor = colorizeDisparity(disp16SFiltered, minDisparity, numDisparities);

        hconcat(rectL, rectR, visRect);
        drawRectificationOverlay(visRect, rectL.cols, roi1, roi2);

        if (showDebugMatches)
        {
            namedWindow("Debug Keypoint Matches", WINDOW_AUTOSIZE);
            //namedWindow("Debug Keypoint Matches", WINDOW_NORMAL);
            //resizeWindow("Debug Keypoint Matches", 640, 480);

            if (makeDebugKeypointMatches(grayL, grayR, debugMatches))
                imshow("Debug Keypoint Matches", debugMatches);
            else
            {
                Mat msg(480, 640, CV_8UC3, Scalar::all(0));
                putText(msg, "Not enough valid ORB stereo matches",
                        Point(20, 240), FONT_HERSHEY_SIMPLEX, 0.8,
                        Scalar(0, 0, 255), 2, LINE_AA);
                imshow("Debug Keypoint Matches", msg);
            }
        }

        // Overlay clicked depth text
        if (g_lastClick.x >= 0 && g_lastClick.y >= 0 &&
            g_lastClick.x < points3D.cols && g_lastClick.y < points3D.rows)
        {
            Vec3f p = points3D.at<Vec3f>(g_lastClick.y, g_lastClick.x);
            float disparity = disp32F.at<float>(g_lastClick.y, g_lastClick.x);

            if (std::isfinite(p[2]) && disparity > minDisparity && std::abs(p[2]) < 1e4f)
            {
                cout << "Pixel (" << g_lastClick.x << ", " << g_lastClick.y << ") -> "
                     << "disp=" << disparity
                     << "  X=" << p[0]
                     << "  Y=" << p[1]
                     << "  Z=" << p[2] << "\n";

                circle(dispColor, g_lastClick, 5, Scalar(255, 255, 255), 2, LINE_AA);
                string txt = "Z=" + to_string(p[2]) + "  d=" + to_string(disparity);
                putText(dispColor, txt,
                        Point(max(10, g_lastClick.x + 10), max(20, g_lastClick.y - 10)),
                        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2, LINE_AA);
            }
            else
            {
                cout << "Pixel (" << g_lastClick.x << ", " << g_lastClick.y
                     << ") has invalid disparity/depth\n";
            }

            g_lastClick = Point(-1, -1);
        }

        putText(dispColor,
                "SGBM disparity | d=debug matches | click for depth",
                Point(20, 30),
                FONT_HERSHEY_SIMPLEX, 0.7,
                Scalar(255, 255, 255), 2, LINE_AA);

        imshow("Rectified (L|R)", visRect);
        imshow("Disparity", dispColor);

        int key = waitKey(1) & 0xFF;
        if (key == 27 || key == 'q')
            break;
        else if (key == 'd' || key == 'D')
        {
            showDebugMatches = !showDebugMatches;
            if (!showDebugMatches)
                destroyWindow("Debug Keypoint Matches");
        }
    }

    capL.release();
    capR.release();
    destroyAllWindows();
    return 0;
}
