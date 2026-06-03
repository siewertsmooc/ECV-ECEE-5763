#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <time.h>
#include <syslog.h>
#include "BackgroundRemover.h"
#include "SkinDetector.h"
#include "FaceDetector.h"
#include "FingerCount.h"

using namespace cv;
using namespace std;

int main(int, char**) {
	VideoCapture videoCapture(0);
	videoCapture.set(CAP_PROP_SETTINGS, 1);

	if (!videoCapture.isOpened()) {
		cout << "Can't find camera!" << endl;
		return -1;
	}

	Mat frame, frameOut, handMask, foreground, fingerCountDebug;

	BackgroundRemover backgroundRemover;
	SkinDetector skinDetector;
	FaceDetector faceDetector;
	FingerCount fingerCount;
	time_t startTime, curTime, endTime;
    time(&startTime);
    int numFramesCaptured = 0;
    double secElapsed;
    double curFPS;
    double averageFPS = 0.0;
	cout << "Frame Rate = " << curFPS << endl;
	setlogmask(LOG_UPTO (LOG_NOTICE));
	openlog("Log_tag", LOG_CONS | LOG_PID | LOG_NDELAY, LOG_LOCAL1);
	syslog(LOG_NOTICE, "Logging");
	while (true) {
		time(&startTime);
		videoCapture >> frame;
		frameOut = frame.clone();

		skinDetector.drawSkinColorSampler(frameOut);

		foreground = backgroundRemover.getForeground(frame);
		
		faceDetector.removeFaces(frame, foreground);
		handMask = skinDetector.getSkinMask(foreground);
		fingerCountDebug = fingerCount.findFingersCount(handMask, frameOut);
		imshow("output", frameOut);
		imshow("foreground", foreground);
		imshow("handMask", handMask);
		imshow("handDetection", fingerCountDebug);
		time(&endTime);
		/*double seconds = difftime(end, start);
		fps = num_frames/seconds;
		cout << "Frame rate per second = " << fps << endl;*/
		time(&curTime);
        double secElapsed = difftime(curTime, startTime);
        double curFPS = numFramesCaptured / secElapsed;
        cout << "FPS = " << curFPS << endl;
        cout << "secElapsed = " << secElapsed << " secs, numFramesCaptured = " << numFramesCaptured << endl;
        // compute running average of frames
        if (secElapsed > 0)
            averageFPS = (averageFPS * (numFramesCaptured - 1) + curFPS)
                                / numFramesCaptured;
		syslog(LOG_NOTICE, "Current FPS: %f\n", curFPS);
		int key = waitKey(400);

		if (key == 27) // esc
			break;
		else if (key == 98) // b
			backgroundRemover.calibrate(frame);
		else if (key == 115) // s
			skinDetector.calibrate(frame);
	}
	cout << "Average FPS = " << averageFPS << endl;
	syslog(LOG_NOTICE, "Finished running");
	closelog();
	return 0;
}
