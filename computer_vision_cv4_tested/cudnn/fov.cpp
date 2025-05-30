#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char** argv) {
    // Check for video file argument
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <VideoPath>\n";
        return -1;
    }

    // Open the video file
    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
        std::cout << "Error opening video file " << argv[1] << "\n";
        return -1;
    }

    // Create a window to display the frames
    cv::namedWindow("Field of View", cv::WINDOW_AUTOSIZE);

    // Variables to hold frame data
    cv::Mat frame;

    while (true) {
        // Read the current frame
        if (!cap.read(frame)) {
            break; // Exit loop if no more frames
        }

        // Dimensions of the frame
        int frameWidth = frame.cols;
        int frameHeight = frame.rows;



        // Calculate points for the field of view
        cv::Point leftcenter(frameWidth / 4, frameHeight);
        cv::Point rightcenter(3 * frameWidth / 4 , frameHeight);
        cv::Point leftPoint(frameWidth / 10 , frameHeight / 2);
        cv::Point rightPoint(9 * frameWidth / 10, frameHeight / 2);

        // Draw lines to represent the field of view
        cv::line(frame, leftcenter, leftPoint, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        cv::line(frame, rightcenter, rightPoint, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        cv::line(frame, leftPoint, rightPoint, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

        std::vector<cv::Point> polygon;
        polygon.push_back(leftcenter);
        polygon.push_back(rightcenter);
        polygon.push_back(rightPoint);
        polygon.push_back(leftPoint);

        // Create a mask with the same dimensions as the frame, initially all 0 (black)
        cv::Mat mask = cv::Mat::zeros(frame.size(), frame.type());

        // Fill the polygon with white color in the mask
        cv::fillConvexPoly(mask, polygon.data(), polygon.size(), cv::Scalar(255, 255, 255));

        // Apply the mask to the frame
        cv::Mat maskedFrame;
        frame.copyTo(maskedFrame, mask);
        

        // Display the frame
        cv::imshow("Field of View", maskedFrame);

        // Press 27 (ESC) to exit
        char c = (char)cv::waitKey(25);
        if (c == 27)
            break;
    }

    // Release the video capture and close any open windows
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
