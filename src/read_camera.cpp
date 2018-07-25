#include <iostream>
#include <opencv2/opencv.hpp>
#include "camera.h"
#include <string.h>
#include "utils.h"
#include <glog/logging.h>

#define QUIT_KEY 'q'

using namespace std;
using namespace cv;

void process_camera(const CameraConfig &camera) {

    cout << "read camera: " << camera.identity() << endl;

    VideoCapture cap = camera.GetCapture();
    if (!cap.isOpened()) {
        LOG(ERROR) << "failed to open camera";
        return;
    }

    Mat frame, resized;

    do {
        cap >> frame;
        if (!frame.data) {
            LOG(ERROR) << "Capture video failed";
            exit(1);
        }

        resize(frame, resized, Size(1280, 720), 0, 0, INTER_NEAREST);
        imshow("window", resized);

    } while (QUIT_KEY != waitKey(1));
}

int main(int argc, char* argv[]) {

    google::InitGoogleLogging(argv[0]);

    const String keys =
        "{help h usage ? |                         | print this message   }"
        "{config         |config.toml              | camera config        }"
    ;

    CommandLineParser parser(argc, argv, keys);
    parser.about("camera face detector");
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    String config_path = parser.get<String>("config");
    LOG(INFO) << "config path: " << config_path;

    if (!parser.check()) {
        parser.printErrors();
        return 0;
    }

    vector<CameraConfig> cameras = LoadCameraConfig(config_path);

    process_camera(cameras[0]);

}
