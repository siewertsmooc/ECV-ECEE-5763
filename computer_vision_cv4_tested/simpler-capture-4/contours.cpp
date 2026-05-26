#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <set>

using namespace cv;
using namespace std;

static double pointDist(const Point& a, const Point& b) {
    double dx = double(a.x - b.x);
    double dy = double(a.y - b.y);
    return std::sqrt(dx * dx + dy * dy);
}

static double angleDeg(const Point& s, const Point& f, const Point& e) {
    // angle at point f formed by s-f-e
    double a = pointDist(f, e);
    double b = pointDist(f, s);
    double c = pointDist(s, e);

    if (a <= 1e-6 || b <= 1e-6) return 180.0;

    double cosVal = (a * a + b * b - c * c) / (2.0 * a * b);
    cosVal = std::max(-1.0, std::min(1.0, cosVal));
    return acos(cosVal) * 180.0 / CV_PI;
}

static Point contourCentroid(const vector<Point>& contour) {
    Moments m = moments(contour);
    if (fabs(m.m00) < 1e-6) return Point(0, 0);
    return Point(int(m.m10 / m.m00), int(m.m01 / m.m00));
}

static int clampFingerCount(int n) {
    if (n < 0) return 0;
    if (n > 5) return 5;
    return n;
}

int main() {
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Could not open camera.\n";
        return 1;
    }

    // Try to keep camera size reasonable
    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);

    const Rect roiRect(320, 80, 260, 300); // right side ROI

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        flip(frame, frame, 1); // mirror view

        // Make sure ROI is valid for current frame size
        Rect safeROI = roiRect & Rect(0, 0, frame.cols, frame.rows);
        if (safeROI.width <= 0 || safeROI.height <= 0) {
            cerr << "ROI is out of frame bounds.\n";
            break;
        }

        Mat roi = frame(safeROI).clone();

        // --- Skin segmentation in YCrCb ---
        Mat ycrcb;
        cvtColor(roi, ycrcb, COLOR_BGR2YCrCb);

        // Common rough skin range in YCrCb; tune for your lighting/skin tone
        Scalar lower(0, 133, 77);
        Scalar upper(255, 173, 127);

        Mat mask;
        inRange(ycrcb, lower, upper, mask);

        // Clean up mask
        GaussianBlur(mask, mask, Size(7, 7), 0);
        morphologyEx(mask, mask, MORPH_OPEN,
                     getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
        morphologyEx(mask, mask, MORPH_CLOSE,
                     getStructuringElement(MORPH_ELLIPSE, Size(7, 7)));
        medianBlur(mask, mask, 5);

        // Optional: make the hand blob larger and smoother
        dilate(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

        // --- Find contours ---
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        int fingerCount = 0;

        if (!contours.empty()) {
            // Largest contour assumed to be the hand
            int bestIdx = -1;
            double bestArea = 0.0;
            for (int i = 0; i < (int)contours.size(); ++i) {
                double a = contourArea(contours[i]);
                if (a > bestArea) {
                    bestArea = a;
                    bestIdx = i;
                }
            }

            if (bestIdx >= 0 && bestArea > 4000) {
                vector<Point> handContour = contours[bestIdx];

                // Smooth contour a bit
                double eps = 0.003 * arcLength(handContour, true);
                vector<Point> approxContour;
                approxPolyDP(handContour, approxContour, eps, true);

                // Draw contour
                drawContours(roi, vector<vector<Point>>{approxContour}, -1, Scalar(255, 0, 0), 2);

                Point center = contourCentroid(approxContour);
                circle(roi, center, 5, Scalar(0, 255, 255), FILLED);

                // Convex hull indices required for convexityDefects
                vector<int> hullIndices;
                convexHull(approxContour, hullIndices, false, false);

                vector<Point> hullPoints;
                convexHull(approxContour, hullPoints, false, true);
                polylines(roi, hullPoints, true, Scalar(0, 255, 0), 2);

                vector<Vec4i> defects;
                if (hullIndices.size() >= 3) {
                    convexityDefects(approxContour, hullIndices, defects);
                }

                set<int> fingertipIndices;

                // Bounding box for scale-aware thresholds
                Rect handBox = boundingRect(approxContour);
                rectangle(roi, handBox, Scalar(255, 255, 0), 2);

                // Palm-ish reference: lower half of contour bbox often contains palm
                int palmYThreshold = handBox.y + int(handBox.height * 0.75);

                for (const auto& d : defects) {
                    int startIdx = d[0];
                    int endIdx   = d[1];
                    int farIdx   = d[2];
                    float depth  = d[3] / 256.0f; // OpenCV stores with fixed-point scale

                    Point start = approxContour[startIdx];
                    Point end   = approxContour[endIdx];
                    Point far   = approxContour[farIdx];

                    double ang = angleDeg(start, far, end);
                    double startDist = pointDist(start, center);
                    double endDist   = pointDist(end, center);
                    double farDist   = pointDist(far, center);

                    // Heuristics:
                    // - valley angle should be fairly sharp
                    // - defect depth should be significant
                    // - fingertips usually above the palm area
                    // - fingertip points should be far from contour centroid
                    bool validDepth = depth > handBox.height * 0.06;
                    bool validAngle = ang < 90.0;
                    bool startHigh  = start.y < palmYThreshold;
                    bool endHigh    = end.y < palmYThreshold;
                    bool farBelowTips = (far.y > start.y || far.y > end.y);
                    bool startFarEnough = startDist > handBox.height * 0.35;
                    bool endFarEnough   = endDist > handBox.height * 0.35;
                    bool valleyReasonable = farDist > handBox.height * 0.15;

                    if (validDepth && validAngle && farBelowTips && valleyReasonable) {
                        circle(roi, far, 6, Scalar(0, 0, 255), FILLED);

                        if (startHigh && startFarEnough) {
                            fingertipIndices.insert(startIdx);
                        }
                        if (endHigh && endFarEnough) {
                            fingertipIndices.insert(endIdx);
                        }
                    }
                }

                // Merge fingertip candidates that are too close together
                vector<Point> fingertips;
                for (int idx : fingertipIndices) {
                    Point p = approxContour[idx];
                    bool merged = false;
                    for (auto& q : fingertips) {
                        if (pointDist(p, q) < 25) {
                            merged = true;
                            break;
                        }
                    }
                    if (!merged) fingertips.push_back(p);
                }

                // Draw fingertip markers
                for (const auto& p : fingertips) {
                    circle(roi, p, 8, Scalar(0, 255, 255), FILLED);
                }

                fingerCount = clampFingerCount((int)fingertips.size());

                // Fist / one-finger fallback:
                // If no good defects were found, use contour extent + topmost point heuristic
                if (fingerCount == 0) {
                    double area = contourArea(approxContour);
                    double rectArea = handBox.area();
                    double extent = (rectArea > 0) ? area / rectArea : 0.0;

                    Point topmost = approxContour[0];
                    for (const auto& p : approxContour) {
                        if (p.y < topmost.y) topmost = p;
                    }

                    bool maybeOneFinger =
                        (topmost.y < handBox.y + int(handBox.height * 0.25)) &&
                        (extent < 0.72);

                    if (maybeOneFinger) {
                        fingerCount = 1;
                        circle(roi, topmost, 10, Scalar(255, 0, 255), FILLED);
                    }
                }

                putText(roi, "Fingers: " + to_string(fingerCount),
                        Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0,
                        Scalar(0, 255, 0), 2);
            } else {
                putText(roi, "Show hand in box", Point(10, 30),
                        FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
            }
        } else {
            putText(roi, "No hand found", Point(10, 30),
                    FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
        }

        // Draw ROI on main frame
        rectangle(frame, safeROI, Scalar(0, 255, 0), 2);
        roi.copyTo(frame(safeROI));

        imshow("Finger Counter", frame);
        imshow("Mask", mask);

        char key = (char)waitKey(10);
        if (key == 27 || key == 'q') break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
