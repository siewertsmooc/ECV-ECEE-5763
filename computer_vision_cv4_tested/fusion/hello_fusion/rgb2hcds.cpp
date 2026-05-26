// Generated with ChatGPT 5.x Thinking and updated
//
// best: the currently selected best cross-camera match
//
// comp: compatibility - similar radius, similar relative images position, similar confidence
//
// For the Dempster-Shafer outputs:
//
// obj: belief mass for object present
//      here it means belief that this matched pair corresponds to the target object
//      higher obj = stronger support that the object is really there
//
// no: belief mass for no object
//     evidence against the object hypothesis
//     higher no = stronger support that this is not a valid object match
//
// unk: belief mass for unknown / uncertainty
//      neither strong yes nor strong no
//      this grows when the detector is unsure, weak, or conflicting
//
//
// comp = “do these two detections line up?”
// best = “which pair won?”
//
// obj = “how much do I believe the object is there?”
// no = “how much do I believe it is not there?”
// unk = “how unsure am I?”
//
// argument 1: first camera device
// argument 2: second camera device
// argument 3: number of extra candidate circles to show per camera
//
//
//
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

struct MassFunction {
    double object;
    double noObject;
    double unknown;
};

struct CircleCandidate {
    cv::Point2f center;
    float radius = 0.0f;
    double greenScore = 0.0;
    double edgeScore = 0.0;
    double confidence = 0.0;
    cv::Rect roi;
};

struct DetectionSet {
    std::vector<CircleCandidate> circles;
    cv::Mat debugMask;
};

struct BestPairResult {
    bool found = false;
    int idx1 = -1;
    int idx2 = -1;
    double compatibility = 0.0;
    MassFunction fused{0.0, 0.0, 1.0};
};

static double clamp01(double v) {
    return std::max(0.0, std::min(1.0, v));
}

static MassFunction makeMass(double confidence, double uncertainty = 0.25) {
    confidence = clamp01(confidence);
    uncertainty = clamp01(uncertainty);

    double remaining = 1.0 - uncertainty;
    return {
        confidence * remaining,
        (1.0 - confidence) * remaining,
        uncertainty
    };
}

static MassFunction combineDempster(const MassFunction& a, const MassFunction& b) {
    double K = a.object * b.noObject + a.noObject * b.object;
    if (K >= 0.999999) {
        return {0.0, 0.0, 1.0};
    }

    double norm = 1.0 / (1.0 - K);
    MassFunction out;
    out.object = norm * (
        a.object * b.object +
        a.object * b.unknown +
        a.unknown * b.object
    );
    out.noObject = norm * (
        a.noObject * b.noObject +
        a.noObject * b.unknown +
        a.unknown * b.noObject
    );
    out.unknown = norm * (a.unknown * b.unknown);
    return out;
}

static cv::Rect clampRect(const cv::Rect& r, const cv::Size& bounds) {
    int x = std::max(0, r.x);
    int y = std::max(0, r.y);
    int w = std::min(r.width, bounds.width - x);
    int h = std::min(r.height, bounds.height - y);
    if (w <= 0 || h <= 0) {
        return cv::Rect();
    }
    return cv::Rect(x, y, w, h);
}

static double scoreGreenFill(const cv::Mat& hsv, const cv::Point2f& center, float radius) {
    cv::Mat mask = cv::Mat::zeros(hsv.rows, hsv.cols, CV_8U);
    cv::circle(mask, center, cvRound(radius * 0.75f), cv::Scalar(255), -1);

    cv::Mat greenMask;
    cv::inRange(hsv, cv::Scalar(25, 60, 40), cv::Scalar(95, 255, 255), greenMask);

    int insidePixels = cv::countNonZero(mask);
    if (insidePixels <= 0) {
        return 0.0;
    }

    cv::Mat overlap;
    cv::bitwise_and(mask, greenMask, overlap);
    int greenPixels = cv::countNonZero(overlap);
    return clamp01(static_cast<double>(greenPixels) / static_cast<double>(insidePixels));
}

static double scoreCircleEdge(const cv::Mat& gray, const cv::Point2f& center, float radius) {
    cv::Mat edges;
    cv::Canny(gray, edges, 80, 160);

    int samples = 48;
    int hits = 0;
    for (int i = 0; i < samples; ++i) {
        double theta = 2.0 * CV_PI * static_cast<double>(i) / static_cast<double>(samples);
        int x = cvRound(center.x + radius * std::cos(theta));
        int y = cvRound(center.y + radius * std::sin(theta));
        if (x >= 0 && x < edges.cols && y >= 0 && y < edges.rows) {
            if (edges.at<unsigned char>(y, x) > 0) {
                ++hits;
            }
        }
    }
    return clamp01(static_cast<double>(hits) / static_cast<double>(samples));
}

static DetectionSet detectGreenCircles(const cv::Mat& frame) {
    DetectionSet out;
    if (frame.empty()) {
        return out;
    }

    const double scale = 0.5;
    cv::Mat small;
    cv::resize(frame, small, cv::Size(), scale, scale, cv::INTER_LINEAR);

    cv::Mat hsv, gray;
    cv::cvtColor(small, hsv, cv::COLOR_BGR2HSV);
    cv::cvtColor(small, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 1.5);

    cv::Mat greenMask;
    cv::inRange(hsv, cv::Scalar(25, 60, 40), cv::Scalar(95, 255, 255), greenMask);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(greenMask, greenMask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(greenMask, greenMask, cv::MORPH_CLOSE, kernel);
    out.debugMask = greenMask.clone();

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(greenMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area < 80.0) {
            continue;
        }

        cv::Rect box = cv::boundingRect(contour);
        int pad = 10;
        cv::Rect roi = clampRect(cv::Rect(box.x - pad, box.y - pad, box.width + 2 * pad, box.height + 2 * pad), small.size());
        if (roi.empty()) {
            continue;
        }

        std::vector<cv::Vec3f> circles;
        cv::HoughCircles(
            gray(roi),
            circles,
            cv::HOUGH_GRADIENT,
            1.2,
            std::max(12.0, roi.height / 6.0),
            100,
            16,
            4,
            std::max(roi.width, roi.height) / 2
        );

        for (const auto& c : circles) {
            CircleCandidate cand;
            cand.center = cv::Point2f((c[0] + static_cast<float>(roi.x)) / static_cast<float>(scale),
                                      (c[1] + static_cast<float>(roi.y)) / static_cast<float>(scale));
            cand.radius = c[2] / static_cast<float>(scale);
            cand.roi = cv::Rect(cvRound(cand.center.x - cand.radius),
                                cvRound(cand.center.y - cand.radius),
                                cvRound(2.0f * cand.radius),
                                cvRound(2.0f * cand.radius));

            cv::Point2f smallCenter(c[0] + static_cast<float>(roi.x), c[1] + static_cast<float>(roi.y));
            cand.greenScore = scoreGreenFill(hsv, smallCenter, c[2]);
            cand.edgeScore = scoreCircleEdge(gray, smallCenter, c[2]);

            double sizeScore = clamp01(static_cast<double>(c[2]) / 35.0);
            cand.confidence = clamp01(0.55 * cand.greenScore + 0.30 * cand.edgeScore + 0.15 * sizeScore);
            out.circles.push_back(cand);
        }
    }

    std::sort(out.circles.begin(), out.circles.end(), [](const CircleCandidate& a, const CircleCandidate& b) {
        return a.confidence > b.confidence;
    });

    const std::size_t maxKeep = 10;
    if (out.circles.size() > maxKeep) {
        out.circles.resize(maxKeep);
    }

    return out;
}

static double pairCompatibility(const CircleCandidate& a,
                                const CircleCandidate& b,
                                const cv::Size& sizeA,
                                const cv::Size& sizeB) {
    double maxRadius = std::max(1.0,
        std::max(static_cast<double>(a.radius), static_cast<double>(b.radius)));

    double radiusAgreement = 1.0 - clamp01(
        std::abs(static_cast<double>(a.radius) - static_cast<double>(b.radius)) / maxRadius
    );

    cv::Point2f na(
        a.center.x / static_cast<float>(std::max(1, sizeA.width)),
        a.center.y / static_cast<float>(std::max(1, sizeA.height))
    );
    cv::Point2f nb(
        b.center.x / static_cast<float>(std::max(1, sizeB.width)),
        b.center.y / static_cast<float>(std::max(1, sizeB.height))
    );

    double posDist = cv::norm(na - nb);
    double positionAgreement = 1.0 - clamp01(posDist / 0.75);
    double confidenceAgreement = 1.0 - clamp01(std::abs(a.confidence - b.confidence));
    double greenAgreement = 1.0 - clamp01(std::abs(a.greenScore - b.greenScore));

    return clamp01(
        0.35 * radiusAgreement +
        0.25 * positionAgreement +
        0.20 * confidenceAgreement +
        0.20 * greenAgreement
    );
}

static BestPairResult findBestPair(const std::vector<CircleCandidate>& cam1,
                                   const std::vector<CircleCandidate>& cam2,
                                   const cv::Size& size1,
                                   const cv::Size& size2) {
    BestPairResult best;
    double bestScore = -std::numeric_limits<double>::infinity();

    for (int i = 0; i < static_cast<int>(cam1.size()); ++i) {
        for (int j = 0; j < static_cast<int>(cam2.size()); ++j) {
            double compatibility = pairCompatibility(cam1[i], cam2[j], size1, size2);

            double conf1 = cam1[i].confidence * compatibility;
            double conf2 = cam2[j].confidence * compatibility;

            MassFunction m1 = makeMass(conf1, 0.25);
            MassFunction m2 = makeMass(conf2, 0.25);
            MassFunction fused = combineDempster(m1, m2);

            double score = fused.object - 0.25 * fused.unknown;
            if (score > bestScore) {
                bestScore = score;
                best.found = true;
                best.idx1 = i;
                best.idx2 = j;
                best.compatibility = compatibility;
                best.fused = fused;
            }
        }
    }

    return best;
}

static void drawCandidates(cv::Mat& frame,
                           const std::vector<CircleCandidate>& candidates,
                           const std::string& label,
                           int extraCandidatesToShow,
                           int bestIdx) {
    cv::putText(frame, label, cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

    int limit = static_cast<int>(candidates.size());
    if (extraCandidatesToShow >= 0) {
        limit = std::min(limit, extraCandidatesToShow + 1);
    }

    for (int i = 0; i < limit; ++i) {
        if (i == bestIdx) {
            continue;
        }

        const auto& c = candidates[static_cast<std::size_t>(i)];
        cv::circle(frame, c.center, cvRound(c.radius), cv::Scalar(0, 255, 255), 2);
        cv::circle(frame, c.center, 2, cv::Scalar(0, 0, 255), -1);

        std::string text = "#" + std::to_string(i) +
            " conf=" + cv::format("%.2f", c.confidence) +
            " green=" + cv::format("%.2f", c.greenScore);
        cv::putText(frame, text, cv::Point(cvRound(c.center.x + 5), cvRound(c.center.y - 5)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 255, 255), 1);
    }
}

int main(int argc, char** argv) {
    int camDevice1 = 0;
    int camDevice2 = 2;
    int extraCandidatesToShow = 9;

    if (argc >= 2) {
        camDevice1 = std::stoi(argv[1]);
    }
    if (argc >= 3) {
        camDevice2 = std::stoi(argv[2]);
    }
    if (argc >= 4) {
        extraCandidatesToShow = std::max(0, std::stoi(argv[3]));
    }

    cv::VideoCapture cam1(camDevice1);
    cv::VideoCapture cam2(camDevice2);

    cam1.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cam1.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cam2.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cam2.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cam1.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cam2.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));

    if (!cam1.isOpened() || !cam2.isOpened()) {
        std::cerr << "Failed to open camera inputs " << camDevice1 << " and " << camDevice2 << ".\n";
        return 1;
    }

    std::cout << "Using cameras " << camDevice1 << " and " << camDevice2
              << ", extra candidates shown per camera: " << extraCandidatesToShow << "\n";
    std::cout << "Press ESC to quit.\n";

    while (true) {
        cv::Mat frame1, frame2;
        cam1 >> frame1;
        cam2 >> frame2;
        if (frame1.empty() || frame2.empty()) {
            std::cerr << "Failed to grab frames.\n";
            break;
        }

        DetectionSet det1 = detectGreenCircles(frame1);
        DetectionSet det2 = detectGreenCircles(frame2);
        BestPairResult best = findBestPair(det1.circles, det2.circles, frame1.size(), frame2.size());

        int bestIdx1 = best.found ? best.idx1 : -1;
        int bestIdx2 = best.found ? best.idx2 : -1;
        drawCandidates(frame1, det1.circles, "Camera 1", extraCandidatesToShow, bestIdx1);
        drawCandidates(frame2, det2.circles, "Camera 2", extraCandidatesToShow, bestIdx2);

        if (best.found) {
            const auto& a = det1.circles[static_cast<std::size_t>(best.idx1)];
            const auto& b = det2.circles[static_cast<std::size_t>(best.idx2)];

            cv::circle(frame1, a.center, cvRound(a.radius), cv::Scalar(0, 255, 0), 3);
            cv::circle(frame2, b.center, cvRound(b.radius), cv::Scalar(0, 255, 0), 3);

            cv::putText(frame1, "BEST", cv::Point(cvRound(a.center.x), cvRound(a.center.y + a.radius + 18)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            cv::putText(frame2, "BEST", cv::Point(cvRound(b.center.x), cvRound(b.center.y + b.radius + 18)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        }

        cv::Mat combined;
        cv::hconcat(frame1, frame2, combined);

        std::string summary;
        if (best.found) {
            summary = "Dempster-Shafer Best pair: cam1 #" + std::to_string(best.idx1) +
                      " <-> cam2 #" + std::to_string(best.idx2) +
                      "  comp=" + cv::format("%.2f", best.compatibility) +
                      "  obj=" + cv::format("%.2f", best.fused.object) +
                      "  no=" + cv::format("%.2f", best.fused.noObject) +
                      "  unk=" + cv::format("%.2f", best.fused.unknown);
        } else {
            summary = "No matched green circles found.";
        }

        cv::putText(combined, summary, cv::Point(20, combined.rows - 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 255, 255), 2);

        cv::imshow("Green Hough Pair Dempster-Shafer", combined);

        int key = cv::waitKey(1);
        if (key == 27) {
            break;
        }
    }

    return 0;
}

