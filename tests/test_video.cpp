#include <iostream>
#include <opencv2/opencv.hpp>
//#include <opencv2/tracking.hpp>
#include "mtcnn.h"
#include <cstdlib>
#include "face_attr.h"
#include "face_align.h"
#include <glog/logging.h>
#include <string.h>
#include <kcf/tracker.hpp>
#include "utils.h"

#define QUIT_KEY 'q'

using namespace std;
using namespace cv;


void test_video(const string &model_path, const CameraConfig &camera, string output_folder) {
    
    int detectionFrameInterval = 25; // nb of frames

    MTCNN mm(model_path);
    FaceAttr fa;
    fa.Load();
    FaceAlign align;

    prepare_output_folder(camera, output_folder);

    VideoCapture cap = camera.GetCapture();
    if (!cap.isOpened()) {
        LOG(ERROR) << "failed to open camera: " << camera.identity();
        return;
    }

    int frameCounter = 0;
    long faceId = 0;
    struct timeval  tv1,tv2;
    struct timezone tz1,tz2;

    vector<Bbox> finalBbox;
    Rect2d roi;
    vector<Ptr<Tracker>> trackers;
    vector<Rect2d> boxes;
    Mat frame;

    FileStorage fs;
    fs.open("kcf.yaml", FileStorage::READ);
    TrackerKCF::Params kcf_param;
    kcf_param.read(fs.root());

    namedWindow("face_detection", WINDOW_NORMAL);
    // resizeWindow("face_detection", 800, 600);

    do {
        finalBbox.clear();

        cap >> frame;
        if (!frame.data) {
            cerr << "end" << endl;
            break;
        }
        dlib::cv_image<dlib::bgr_pixel> cimg(frame);

        if (frameCounter % detectionFrameInterval == 0) {
            // start face detection
            gettimeofday(&tv1,&tz1);
            ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR2RGB, frame.cols, frame.rows);
            mm.detect(ncnn_img, finalBbox);
            gettimeofday(&tv2,&tz2);
            int total = 0;

            // update trackers' bounding boxes and create tracker for a new face
            for(vector<Bbox>::iterator it=finalBbox.begin(); it!=finalBbox.end();it++) {
                if((*it).exist) {
                    total++;

                    // get face bounding box
                    Bbox box = *it;
                    std::vector<double> qualities = fa.GetPoseQuality(cimg, box.x1, box.y1, box.x2, box.y2);
                    Rect2d detected_face(Point(box.x1, box.y1),Point(box.x2, box.y2));

                    // test whether is a new face
                    bool newFace = true;
                    unsigned i;
                    for (i=0;i<boxes.size();i++) {
                        Rect2d trackedFace = boxes[i];
                        if (overlap(detected_face, trackedFace)) {
                            newFace = false;
                            break;
                        }
                    }

                    if (newFace) {
                        // create a tracker if a new face is detected
                        Ptr<Tracker> tracker = TrackerKCF::create(kcf_param);
                        tracker->init(frame, detected_face);
                        tracker->id = faceId;
                        trackers.push_back(tracker);
                        boxes.push_back(detected_face);
                        cout << "frame " << frameCounter << ": start tracking face #" << tracker->id << endl;

                        saveFace(frame, box, faceId, output_folder);
                        
                        faceId++;
                    } else {
                        // update tracker's bounding box
                        trackers[i]->reset(frame, detected_face);
                    }
                }
            }

            cout << "frame " << frameCounter << ": detected " << total << " Persons. time eclipsed: " <<  getElapse(&tv1, &tv2) << " ms" << endl;

            // clean up trackers if the tracker doesn't follow a face
            for (unsigned i=0; i < trackers.size(); i++) {
                Ptr<Tracker> tracker = trackers[i];
                Rect2d trackedFace = boxes[i];

                bool isFace = false;
                for (vector<Bbox>::iterator it=finalBbox.begin(); it!=finalBbox.end();it++) {
                    Bbox box = *it;
                    if ((*it).exist) {
                        Bbox box = *it;
                        Rect2d detected_face(Point(box.x1, box.y1),Point(box.x2, box.y2));
                        if (overlap(detected_face, trackedFace)) {
                            isFace = true;
                            break;
                        }
                    } 
                }

                if (!isFace) {
                    /* clean up tracker */
                    cout << "frame " << frameCounter << ": stop tracking face #" << tracker->id << endl;
                    trackers.erase(trackers.begin() + i);
                    boxes.erase(boxes.begin() + i);
                    i--;
                }
            }
        }

        // update trackers
        for (int i = 0; i < trackers.size(); i++) {
            Ptr<Tracker> tracker = trackers[i];

            bool tracked = tracker->update(frame, boxes[i]);
            if (!tracked) {
                // delete tracker
                cout << "frame " << frameCounter << ": stop tracking face #" << tracker->id << endl;
                trackers.erase(trackers.begin() + i);
                boxes.erase(boxes.begin() + i);
                i--;
                continue;
            }

            Rect2d box = boxes[i];
            // draw tracked face
            rectangle( frame, box, Scalar( 255, 0, 0 ), 2, 1 );
            // show face id
            Point middleHighPoint = Point(box.x+box.width/2, box.y);
            putText(frame, to_string(tracker->id), middleHighPoint, FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
        }

        imshow("face_detection", frame);

        frameCounter++;

    } while (QUIT_KEY != waitKey(1));

    cap.release();
	destroyAllWindows();
}

int main(int argc, char* argv[]) {

    google::InitGoogleLogging(argv[0]);

    const String keys =
        "{help h usage ? |                         | print this message   }"
        "{model        |models/ncnn                | path to mtcnn model  }"
        "{config       |/opt/dev_keeper/keeper.toml| camera config        }"
        "{output       |/opt/dev_keeper/faces      | output folder        }"
    ;

    CommandLineParser parser(argc, argv, keys);
    parser.about("camera face detector");
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    String config_path = parser.get<String>("config");
    cout << "config path: " << config_path << endl;

    String model_path = parser.get<String>("model");
    String output_folder = parser.get<String>("output");
    if (!parser.check()) {
        parser.printErrors();
        return 0;
    }

    vector<CameraConfig> cameras = LoadCameraConfig(config_path);
    test_video(model_path, cameras[0], output_folder);
}
