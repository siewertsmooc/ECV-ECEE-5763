#include "utilities.h"


int main(int argc, char** argv) {

    if(argc != 2){
        std::cerr << "Usage: "<< argv[0] << " <video_file_path> "<<std::endl;
    }
    
    VideoCapture cap(argv[1]);
    if(!cap.isOpened()) {
        std::cerr <<"Could not open video"<<argv[1]<<std::endl;
        return -1;
    }

    configNetwork(personNet);
    configNetwork(faceNet);

    cv::Mat frame, blob;
    double fps_factor = 1.0;
    double video_fps = cap.get(cv::CAP_PROP_FPS);
    fps_factor = 30.0/ video_fps;
    double fps = 0.0;

    cv::namedWindow("Detect", cv::WINDOW_NORMAL); 
    cv::resizeWindow("Detect", 1280, 720); 

    struct timespec start, end;
    double seconds;
    string label;
    int frame_drop_limit = 100;
    int frame_count = 0;


    while (frame_drop_limit) {
        if(!cap.read(frame)) {
            frame_drop_limit--;
            cerr<<"Frame Dropped, Limit pending: "<<frame_drop_limit<<endl;
            continue;
        }

        //cv::resize(frame,frame,cv::Size(640,480),0,0,cv::INTER_LINEAR);
        
        clock_gettime(CLOCK_MONOTONIC, &start);
        vector<Mat> outs;

        blobFromImage(frame, blob, 1/255.0, Size(NETWORK_WIDTH, NETWORK_HEIGHT), Scalar(0, 0, 0), true, false);

        detectPeople(blob,outs);

        postProcess(frame, outs,false,false);

        detectFaces(blob, outs);

        postProcess(frame,outs,true,false);

        clock_gettime(CLOCK_MONOTONIC, &end);
        seconds = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        fps = fps_factor / seconds;

        // Display FPS on frame
        label = format("FPS: %.2f", fps);
        putText(frame, label, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);

        // imshow("Detect", frame);
        // if (waitKey(1) == 27) break; // stop if escape key is pressed

                // Save frame with proper numbering
        std::ostringstream filename;
        filename << "frames/frame_" << std::setfill('0') << std::setw(6) << frame_count++ << ".png";
        cv::imwrite(filename.str(), frame);
    }
    cap.release();
    destroyAllWindows();
    return 0;
}   