#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace cv;
using namespace std;

struct Detection {
    Rect box;
    Point2f center;
};

struct Track {
    int id;
    KalmanFilter kf;
    Rect box;
    Point2f predicted;
    int age = 0;
    int missed = 0;
    int hits = 1;
};

static KalmanFilter makeKalman(float x, float y) {
    KalmanFilter kf(4, 2, 0, CV_32F);
    kf.transitionMatrix = (Mat_<float>(4, 4) <<
        1, 0, 1, 0,
        0, 1, 0, 1,
        0, 0, 1, 0,
        0, 0, 0, 1);

    kf.measurementMatrix = (Mat_<float>(2, 4) <<
        1, 0, 0, 0,
        0, 1, 0, 0);

    setIdentity(kf.processNoiseCov, Scalar::all(1e-2));
    setIdentity(kf.measurementNoiseCov, Scalar::all(2e-1));
    setIdentity(kf.errorCovPost, Scalar::all(1));

    kf.statePost = (Mat_<float>(4, 1) << x, y, 0, 0);
    return kf;
}

static float distancePts(const Point2f& a, const Point2f& b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <input_video> [output_video]" << endl;
        return 0;
    }

    string inputPath = argv[1];
    string outputPath = (argc >= 3) ? argv[2] : "tracked_output.mp4";

    VideoCapture cap(inputPath);
    if (!cap.isOpened()) {
        cerr << "Error: could not open input video: " << inputPath << endl;
        return -1;
    }

    int width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(CAP_PROP_FPS);
    if (fps <= 1.0) fps = 30.0;

    int fourcc = VideoWriter::fourcc('m', 'p', '4', 'v');
    VideoWriter writer(outputPath, fourcc, fps, Size(width, height));
    if (!writer.isOpened()) {
        cerr << "Error: could not create output video: " << outputPath << endl;
        return -1;
    }

    Ptr<BackgroundSubtractor> bg = createBackgroundSubtractorMOG2(120, 20, false);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));

    vector<Track> tracks;
    int nextId = 1;
    const float maxMatchDistance = 120.0f;
    const int maxMissed = 18;
    const double minArea = 1800.0;

    namedWindow("Vehicle Tracking", WINDOW_NORMAL);

    Mat frame;
    while (true) {
        if (!cap.read(frame) || frame.empty()) break;

        Mat fgMask;
        bg->apply(frame, fgMask);
        threshold(fgMask, fgMask, 200, 255, THRESH_BINARY);
        morphologyEx(fgMask, fgMask, MORPH_OPEN, kernel, Point(-1, -1), 1);
        morphologyEx(fgMask, fgMask, MORPH_CLOSE, kernel, Point(-1, -1), 2);
        dilate(fgMask, fgMask, kernel, Point(-1, -1), 1);

        vector<vector<Point>> contours;
        findContours(fgMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        vector<Detection> detections;
        for (const auto& c : contours) {
            double area = contourArea(c);
            if (area < minArea) continue;

            Rect box = boundingRect(c);
            if (box.width < 35 || box.height < 20) continue;

            Detection d;
            d.box = box;
            d.center = Point2f(box.x + box.width * 0.5f, box.y + box.height * 0.5f);
            detections.push_back(d);
        }

        for (auto& tr : tracks) {
            Mat pred = tr.kf.predict();
            tr.predicted = Point2f(pred.at<float>(0), pred.at<float>(1));
            tr.age += 1;
            tr.missed += 1;
        }

        struct Candidate {
            float dist;
            int trackIdx;
            int detIdx;
        };

        vector<Candidate> candidates;
        for (int ti = 0; ti < static_cast<int>(tracks.size()); ++ti) {
            for (int di = 0; di < static_cast<int>(detections.size()); ++di) {
                float dist = distancePts(tracks[ti].predicted, detections[di].center);
                if (dist <= maxMatchDistance) {
                    candidates.push_back({dist, ti, di});
                }
            }
        }

        sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
            return a.dist < b.dist;
        });

        vector<bool> detUsed(detections.size(), false);
        vector<bool> trackUsed(tracks.size(), false);

        for (const auto& c : candidates) {
            if (trackUsed[c.trackIdx] || detUsed[c.detIdx]) continue;

            trackUsed[c.trackIdx] = true;
            detUsed[c.detIdx] = true;

            Mat measurement = (Mat_<float>(2, 1) << detections[c.detIdx].center.x,
                                                    detections[c.detIdx].center.y);
            tracks[c.trackIdx].kf.correct(measurement);
            tracks[c.trackIdx].box = detections[c.detIdx].box;
            tracks[c.trackIdx].missed = 0;
            tracks[c.trackIdx].hits += 1;
        }

        for (int di = 0; di < static_cast<int>(detections.size()); ++di) {
            if (detUsed[di]) continue;
            Track tr;
            tr.id = nextId++;
            tr.kf = makeKalman(detections[di].center.x, detections[di].center.y);
            tr.box = detections[di].box;
            tr.predicted = detections[di].center;
            tracks.push_back(tr);
        }

        tracks.erase(remove_if(tracks.begin(), tracks.end(), [&](const Track& tr) {
            return tr.missed > maxMissed;
        }), tracks.end());

        Mat vis = frame.clone();
        for (const auto& d : detections) {
            rectangle(vis, d.box, Scalar(120, 120, 120), 1);
        }

        for (const auto& tr : tracks) {
            rectangle(vis, tr.box, Scalar(0, 255, 255), 2);
            circle(vis, Point((int)tr.predicted.x, (int)tr.predicted.y), 5, Scalar(255, 0, 0), 2);
            Point est((int)tr.kf.statePost.at<float>(0), (int)tr.kf.statePost.at<float>(1));
            circle(vis, est, 5, Scalar(0, 0, 255), FILLED);

            string label = "ID " + to_string(tr.id);
            int tx = tr.box.x;
            int ty = max(25, tr.box.y - 8);
            putText(vis, label, Point(tx, ty), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(20, 20, 20), 3, LINE_AA);
            putText(vis, label, Point(tx, ty), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 1, LINE_AA);
        }

        putText(vis, "Gray: detections  Blue: Kalman prediction  Red: corrected estimate",
                Point(20, 35), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(15, 15, 15), 3, LINE_AA);
        putText(vis, "Gray: detections  Blue: Kalman prediction  Red: corrected estimate",
                Point(20, 35), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(255, 255, 255), 1, LINE_AA);

        writer.write(vis);
        imshow("Vehicle Tracking", vis);

        int key = waitKey(1);
        if (key == 27) break;
    }

    cap.release();
    writer.release();
    destroyAllWindows();

    cout << "Saved tracked output to: " << outputPath << endl;
    return 0;
}
