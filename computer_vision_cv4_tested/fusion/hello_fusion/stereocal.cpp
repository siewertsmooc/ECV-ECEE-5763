// ./stereo_calibrate calib.yml 0 1 9 6 0.020
//
// That matches the target in the canvas:
//
// 9 × 6 internal corners
// 20 mm squares
// A4 page

// stereo_calibrate.cpp
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>

using namespace cv;
using namespace std;

struct StereoCalibrationData
{
    vector<vector<Point2f>> imagePoints1;
    vector<vector<Point2f>> imagePoints2;
    vector<vector<Point3f>> objectPoints;
    Size imageSize;
};

static vector<Point3f> createChessboardCorners(Size boardSize, float squareSize)
{
    vector<Point3f> corners;
    corners.reserve(boardSize.width * boardSize.height);

    for (int y = 0; y < boardSize.height; ++y)
        for (int x = 0; x < boardSize.width; ++x)
            corners.emplace_back(x * squareSize, y * squareSize, 0.0f);

    return corners;
}

static bool detectChessboard(const Mat& frame,
                             Size boardSize,
                             vector<Point2f>& corners,
                             Mat& visOut)
{
    visOut = frame.clone();

    Mat gray;
    if (frame.channels() == 3)
        cvtColor(frame, gray, COLOR_BGR2GRAY);
    else
        gray = frame.clone();

    bool found = findChessboardCorners(
        gray,
        boardSize,
        corners,
        CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK
    );

    if (!found)
        return false;

    cornerSubPix(
        gray,
        corners,
        Size(11, 11),
        Size(-1, -1),
        TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.01)
    );

    drawChessboardCorners(visOut, boardSize, corners, found);
    return true;
}

static double computeAverageReprojectionError(
    const vector<vector<Point3f>>& objectPoints,
    const vector<vector<Point2f>>& imagePoints,
    const vector<Mat>& rvecs,
    const vector<Mat>& tvecs,
    const Mat& K,
    const Mat& D)
{
    double totalErr = 0.0;
    size_t totalPoints = 0;

    vector<Point2f> projected;
    for (size_t i = 0; i < objectPoints.size(); ++i)
    {
        projectPoints(objectPoints[i], rvecs[i], tvecs[i], K, D, projected);
        double err = norm(imagePoints[i], projected, NORM_L2);

        totalErr += err * err;
        totalPoints += objectPoints[i].size();
    }

    return totalPoints ? std::sqrt(totalErr / totalPoints) : -1.0;
}

static bool runStereoCalibration(const StereoCalibrationData& data,
                                 Mat& K1, Mat& D1,
                                 Mat& K2, Mat& D2,
                                 Mat& R, Mat& T,
                                 Mat& E, Mat& F,
                                 double& monoErr1, double& monoErr2, double& stereoErr)
{
    if (data.imagePoints1.size() < 8 || data.imagePoints2.size() < 8)
    {
        cerr << "Need at least 8 valid stereo pairs for calibration.\n";
        return false;
    }

    vector<Mat> rvecs1, tvecs1, rvecs2, tvecs2;

    K1 = Mat::eye(3, 3, CV_64F);
    D1 = Mat::zeros(1, 5, CV_64F);
    K2 = Mat::eye(3, 3, CV_64F);
    D2 = Mat::zeros(1, 5, CV_64F);

    monoErr1 = calibrateCamera(
        data.objectPoints, data.imagePoints1, data.imageSize,
        K1, D1, rvecs1, tvecs1
    );

    monoErr2 = calibrateCamera(
        data.objectPoints, data.imagePoints2, data.imageSize,
        K2, D2, rvecs2, tvecs2
    );

    // Optional reprojection RMS using projected points
    double reproj1 = computeAverageReprojectionError(
        data.objectPoints, data.imagePoints1, rvecs1, tvecs1, K1, D1
    );
    double reproj2 = computeAverageReprojectionError(
        data.objectPoints, data.imagePoints2, rvecs2, tvecs2, K2, D2
    );

    cout << "Mono camera 1 RMS from calibrateCamera: " << monoErr1 << "\n";
    cout << "Mono camera 2 RMS from calibrateCamera: " << monoErr2 << "\n";
    cout << "Mono camera 1 reprojection RMSE: " << reproj1 << " px\n";
    cout << "Mono camera 2 reprojection RMSE: " << reproj2 << " px\n";

    int flags = CALIB_FIX_INTRINSIC;
    TermCriteria criteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-5);

    stereoErr = stereoCalibrate(
        data.objectPoints,
        data.imagePoints1,
        data.imagePoints2,
        K1, D1,
        K2, D2,
        data.imageSize,
        R, T, E, F,
        flags,
        criteria
    );

    return true;
}

static bool saveCalibration(const string& filename,
                            const Mat& K1, const Mat& D1,
                            const Mat& K2, const Mat& D2,
                            const Mat& R,  const Mat& T,
                            Size imageSize)
{
    FileStorage fs(filename, FileStorage::WRITE);
    if (!fs.isOpened())
    {
        cerr << "Failed to open output file for writing: " << filename << "\n";
        return false;
    }

    fs << "image_width" << imageSize.width;
    fs << "image_height" << imageSize.height;
    fs << "K1" << K1;
    fs << "D1" << D1;
    fs << "K2" << K2;
    fs << "D2" << D2;
    fs << "R"  << R;
    fs << "T"  << T;

    fs.release();
    return true;
}

int main(int argc, char** argv)
{
    // Usage:
    // ./stereo_calibrate output.yml [leftCam=0] [rightCam=1] [cols=9] [rows=6] [squareSize=0.025]
    if (argc < 2)
    {
        cerr << "Usage: " << argv[0]
             << " output.yml [leftCam=0] [rightCam=1] [cols=9] [rows=6] [squareSize=0.025]\n";
        cerr << "Example: " << argv[0] << " calib.yml 0 1 9 6 0.0245\n";
        return 1;
    }

    string outputFile = argv[1];
    int camLeft = (argc >= 3) ? atoi(argv[2]) : 0;
    int camRight = (argc >= 4) ? atoi(argv[3]) : 1;
    int cols = (argc >= 5) ? atoi(argv[4]) : 9;
    int rows = (argc >= 6) ? atoi(argv[5]) : 6;
    float squareSize = (argc >= 7) ? static_cast<float>(atof(argv[6])) : 0.025f;

    Size boardSize(cols, rows);

    VideoCapture capL(camLeft, CAP_ANY);
    VideoCapture capR(camRight, CAP_ANY);

    if (!capL.isOpened() || !capR.isOpened())
    {
        cerr << "Could not open cameras " << camLeft << " and/or " << camRight << "\n";
        return 1;
    }

    capL.set(CAP_PROP_FRAME_WIDTH, 640);
    capL.set(CAP_PROP_FRAME_HEIGHT, 480);
    capR.set(CAP_PROP_FRAME_WIDTH, 640);
    capR.set(CAP_PROP_FRAME_HEIGHT, 480);

    cout << "Stereo calibration capture\n";
    cout << "Board size (internal corners): " << cols << " x " << rows << "\n";
    cout << "Square size: " << squareSize << " (meters or your chosen unit)\n";
    cout << "Keys:\n";
    cout << "  c = capture pair when both boards are found\n";
    cout << "  k = calibrate and save\n";
    cout << "  r = reset samples\n";
    cout << "  q / ESC = quit\n";

    StereoCalibrationData data;
    vector<Point3f> objectTemplate = createChessboardCorners(boardSize, squareSize);

    namedWindow("Left", WINDOW_AUTOSIZE);
    namedWindow("Right", WINDOW_AUTOSIZE);
    namedWindow("Stereo Preview", WINDOW_NORMAL);

    Mat frameL, frameR;
    Mat visL, visR, stereoVis;
    vector<Point2f> cornersL, cornersR;

    while (true)
    {
        if (!capL.grab() || !capR.grab())
            break;
        if (!capL.retrieve(frameL) || !capR.retrieve(frameR))
            break;
        if (frameL.empty() || frameR.empty())
            break;

        bool foundL = detectChessboard(frameL, boardSize, cornersL, visL);
        bool foundR = detectChessboard(frameR, boardSize, cornersR, visR);

        string status = "Pairs: " + to_string(data.imagePoints1.size());
        putText(visL, status, Point(20, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2, LINE_AA);
        putText(visR, status, Point(20, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2, LINE_AA);

        putText(visL, foundL ? "LEFT found" : "LEFT not found",
                Point(20, 65), FONT_HERSHEY_SIMPLEX, 0.7,
                foundL ? Scalar(0,255,0) : Scalar(0,0,255), 2, LINE_AA);

        putText(visR, foundR ? "RIGHT found" : "RIGHT not found",
                Point(20, 65), FONT_HERSHEY_SIMPLEX, 0.7,
                foundR ? Scalar(0,255,0) : Scalar(0,0,255), 2, LINE_AA);

        hconcat(visL, visR, stereoVis);
        putText(stereoVis,
                "c=capture  k=calibrate+save  r=reset  q=quit",
                Point(20, stereoVis.rows - 20),
                FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255,255,0), 2, LINE_AA);

        imshow("Left", visL);
        imshow("Right", visR);
        imshow("Stereo Preview", stereoVis);

        int key = waitKey(1) & 0xFF;

        if (key == 27 || key == 'q' || key == 'Q')
        {
            break;
        }
        else if (key == 'r' || key == 'R')
        {
            data.imagePoints1.clear();
            data.imagePoints2.clear();
            data.objectPoints.clear();
            cout << "Cleared captured samples.\n";
        }
        else if (key == 'c' || key == 'C')
        {
            if (foundL && foundR)
            {
                data.imagePoints1.push_back(cornersL);
                data.imagePoints2.push_back(cornersR);
                data.objectPoints.push_back(objectTemplate);
                data.imageSize = frameL.size();

                cout << "Captured pair #" << data.imagePoints1.size() << "\n";
            }
            else
            {
                cout << "Capture skipped: chessboard not found in both views.\n";
            }
        }
        else if (key == 'k' || key == 'K')
        {
            if (data.imagePoints1.size() < 8)
            {
                cout << "Need at least 8 valid pairs before calibration.\n";
                continue;
            }

            Mat K1, D1, K2, D2, R, T, E, F;
            double monoErr1 = 0.0, monoErr2 = 0.0, stereoErr = 0.0;

            bool ok = runStereoCalibration(
                data, K1, D1, K2, D2, R, T, E, F,
                monoErr1, monoErr2, stereoErr
            );

            if (!ok)
            {
                cerr << "Stereo calibration failed.\n";
                continue;
            }

            cout << fixed << setprecision(6);
            cout << "Stereo RMS from stereoCalibrate: " << stereoErr << "\n";
            cout << "K1:\n" << K1 << "\n";
            cout << "D1:\n" << D1 << "\n";
            cout << "K2:\n" << K2 << "\n";
            cout << "D2:\n" << D2 << "\n";
            cout << "R:\n"  << R  << "\n";
            cout << "T:\n"  << T  << "\n";

            if (saveCalibration(outputFile, K1, D1, K2, D2, R, T, data.imageSize))
                cout << "Saved calibration to " << outputFile << "\n";
            else
                cerr << "Failed to save calibration file.\n";
        }
    }

    capL.release();
    capR.release();
    destroyAllWindows();
    return 0;
}
