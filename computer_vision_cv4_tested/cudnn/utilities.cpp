#include "utilities.h"

cv::dnn::Net faceNet = cv::dnn::readNet(face_cfg_file, face_weights_file);
cv::dnn::Net personNet = cv::dnn::readNet(person_cfg_file,person_weights_file);

void detectFaces(cv::Mat &blob, std::vector<cv::Mat> &outs) {
    faceNet.setInput(blob);
    faceNet.forward(outs, faceNet.getUnconnectedOutLayersNames());
}

void detectPeople(cv::Mat &blob, std::vector<cv::Mat> &outs) {
    personNet.setInput(blob);
    personNet.forward(outs, personNet.getUnconnectedOutLayersNames());
}

void configNetwork(cv::dnn::Net &net){

    if(faceNet.empty() || personNet.empty()){
    std::cerr << "Could not load the neural networks. \nMake sure that the config and the weights are stored in the same directory as the executable.\nThe names need to be faces.cfg, faces.weights, person.cfg, person.weights"<<std::endl;
    exit(0);
    }
    // Check if OpenCV is built with CUDA support and set CUDA as preferable backend and target
    if (cuda::getCudaEnabledDeviceCount() > 0) {
        net.setPreferableBackend(DNN_BACKEND_CUDA);
        net.setPreferableTarget(DNN_TARGET_CUDA);
        cout << "Using CUDA for processing\n";
    } else {
        cerr << "CUDA not available on this device; using CPU.\n";
        net.setPreferableBackend(DNN_BACKEND_OPENCV);
        net.setPreferableTarget(DNN_TARGET_CPU);
    }
}

void getBoxes(const std::vector<cv::Mat>&outs, std::vector<cv::Rect> &boxes, const cv::Mat &frame,std::vector<int> &classIds,std::vector<float> &confidences) {

    for (size_t i = 0; i < outs.size(); ++i) {
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
            cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            cv::Point classIdPoint;
            double confidence;
            cv::minMaxLoc(scores, nullptr, &confidence, nullptr, &classIdPoint);
            if (confidence > CONFIDENCE_THRESHOLD && classIdPoint.x < 2){
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }

}

void postProcess(cv::Mat &frame, const std::vector<cv::Mat> &outs,bool faceProcess = false, bool driverView = false) {

    float confidence_threshold = faceProcess? FACE_CONFIDENCE_THRESHOLD : CONFIDENCE_THRESHOLD;

    std::vector<int> classIds;
    int classId;
    double confidence;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<int> indices;
    cv::Rect box;

    getBoxes(outs, boxes, frame, classIds, confidences);

    NMSBoxes(boxes, confidences, confidence_threshold,NMS_THRESHOLD,indices);

    for(int idx : indices){
        box = boxes[idx];
        if(faceProcess){
            blurFaces(box, frame);
        }
        else{
            classId = classIds[idx];
            confidence = confidences[idx];
            annotate(classId,confidence, box,frame, driverView);
        }
    } 
}

void blurFaces(cv::Rect &box, cv::Mat& frame){
    int left = box.x, top = box.y, right = box.x + box.width, bottom = box.y + box.height;

    cv::rectangle(frame, cv::Point(left,top),cv::Point(right,bottom), Scalar(0,255,0),3);
    if(left >= 0 && top >= 0 && right <= frame.cols && bottom <= frame.rows) {
        cv::Mat roi = frame(box);
        cv::GaussianBlur(roi, roi, cv::Size(31,31),13.0,13.0);
        roi.copyTo(frame(box));
    }
}

void annotate(int classId, float confidence,cv::Rect &box, cv::Mat& frame, bool driverView) {
    int left = box.x, top = box.y, right = box.x + box.width, bottom = box.y + box.height;

    cv::rectangle(frame, cv::Point(left,top),cv::Point(right,bottom), Scalar(0,255,0),3);
    std::string label = format("%.2f", confidence);
    std::string classP = classId? "Cyclist" : "Person";
    label = classP + ": " + label;
    cv::putText(frame, label, Point(left, top - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
    if(driverView) {
        putText(frame, "Slow Down!", Point(frame.cols / 3, 50), FONT_HERSHEY_SIMPLEX, 2, Scalar(0,0,255),4);

    }else{
        putText(frame, "Still Crossing!", Point(frame.cols / 3, 50), FONT_HERSHEY_SIMPLEX, 2, Scalar(0,255,255),4);
    }
    

}