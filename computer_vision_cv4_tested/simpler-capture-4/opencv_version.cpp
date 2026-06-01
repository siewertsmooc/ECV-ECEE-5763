#include <iostream>
#include <opencv2/core.hpp>

int main() {
    std::cout << "OpenCV version: " << cv::getVersionString() << std::endl;
    std::cout << "Major version: " << cv::getVersionMajor() << std::endl;
    std::cout << "Minor version: " << cv::getVersionMinor() << std::endl;
    std::cout << "Revision version: " << cv::getVersionRevision() << std::endl;
    return 0;
}
