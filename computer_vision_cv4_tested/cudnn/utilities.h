#ifndef UTILITY_H
#define UTILITY_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/cuda.hpp>
#include <iostream>

using namespace cv;
using namespace std;
using namespace cv::dnn;

const float FACE_CONFIDENCE_THRESHOLD = 0.3;
const float CONFIDENCE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.3;

const int NETWORK_HEIGHT = 416;
const int NETWORK_WIDTH = 416;

const std::string face_cfg_file = "faces.cfg";
const std::string face_weights_file = "faces.weights";
const std::string person_cfg_file = "person.cfg";
const std::string person_weights_file = "person.weights";

extern cv::dnn::Net faceNet;
extern cv::dnn::Net personNet;

void configNetwork(cv::dnn::Net&);

void postProcess(cv::Mat&, const std::vector<cv::Mat>&, bool,bool);

void getBoxes(const std::vector<cv::Mat>&, std::vector<cv::Rect>&, const cv::Mat&, std::vector<int> &, std::vector<float>&);

void detectFaces(cv::Mat&, std::vector<cv::Mat>&);

void blurFaces(cv::Rect&, cv::Mat&);

void detectPeople(cv::Mat&, std::vector<cv::Mat>&);

void annotate(int, float, cv::Rect&, cv::Mat&, bool);

#endif