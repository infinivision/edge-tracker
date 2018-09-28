#include <iostream>
#include <opencv2/opencv.hpp>
#include "camera.h"
#include "staple_tracker.hpp" // staple trakcer
#include "face_tracker.h"
#include "mtcnn.h"
#include <string.h>
#include <chrono>
#include <cstdlib>
#include "utils.h"
#include "face_attr.h"
#include "face_predict.h"
#include "face_pose_estimate.h"
#include "face_embed.h"
#include "face_age.h"
#include "vector_search.h"
#include "face_sample.h"
#include <glog/logging.h>
#include <thread>

#define QUIT_KEY 'q'

using namespace std;
using namespace cv;

/*
 * Decide whether the detected face is same as the tracking one
 * 
 * return true when:
 *   center point of one box is inside the other
 */

int detection_period = 20;
int min_score = 25;

bool overlap(Rect2d &box1, Rect2d &box2) {
    int x1 = box1.x + box1.width/2;
    int y1 = box1.y + box1.height/2;
    int x2 = box2.x + box2.width/2;
    int y2 = box2.y + box2.height/2;

    if ( x1 > box2.x && x1 < box2.x + box2.width &&
         y1 > box2.y && y1 < box2.y + box2.height &&
         x2 > box1.x && x2 < box1.x + box1.width &&
         y2 > box1.y && y2 < box1.y + box1.height ) {
        return true;
    }

    return false;
}

// prepare (clean output folder), output_folder argument will be changed!
void prepare_output_folder(const CameraConfig &camera, string &output_folder) {
    output_folder += "/" + camera.identity();
    string cmd = "rm -rf " + output_folder + "";
    system(cmd.c_str());

    cmd = "mkdir -p " + output_folder + "/original";
    int dir_err = system(cmd.c_str());
    if (-1 == dir_err) {
        LOG(ERROR) << "Error creating directory";
        exit(1);
    }

    cmd = "mkdir -p " + output_folder + "/detect";
    dir_err = system(cmd.c_str());
    if (-1 == dir_err) {
        LOG(ERROR) << "Error creating directory";
        exit(1);
    }

    cmd = "mkdir -p " + output_folder + "/tracker";
    dir_err = system(cmd.c_str());
    if (-1 == dir_err) {
        LOG(ERROR) << "Error creating directory";
        exit(1);
    }    
}

void process_camera(const string mtcnn_model_path, const CameraConfig &camera, string output_folder, \
                    const std::string ckdb_ip, const int ckdb_port, bool main_thread) {

    cout << "processing camera: " << camera.identity() << endl;

    prepare_output_folder(camera, output_folder);
    
    MTCNN mm(mtcnn_model_path);

    VideoCapture cap = camera.GetCapture();
    
    if (!cap.isOpened()) {
        cerr << "failed to open camera" << endl;
        return;
    }

    dbHandle db(ckdb_ip, ckdb_port, camera.identity());

    long frameCounter = 0;
    long faceId = 0;
    long thisFace = 0;

    #ifdef BENCH_EDGE
    struct timeval  tv1,tv2,tv3;
    #endif

    vector<Bbox> detected_bounding_boxes;
    vector<face_tracker> tracker_vec;
    vector<cv::Point2d> image_points;
    Mat frame;

    do {
        detected_bounding_boxes.clear();
        cap >> frame;
        if (!frame.data) {

            //if cap is constrcuct from a video file, function must return if there is no more frame
            if(camera.source_type==3) return;

            LOG(ERROR) << "Capture video failed: " << camera.identity() << ", opened: " << cap.isOpened();
            cap.release();

            LOG(ERROR) << "sleep for 5 seconds ...";
            std::this_thread::sleep_for(std::chrono::seconds(5));
            
            cap = camera.GetCapture();
            if (!cap.isOpened()) {
                LOG(ERROR) << "failed to open camera: " << camera.identity();
                return;
            }
            LOG(INFO) << "camera["<< camera.NO << "]" << "reopen camera: " << camera.identity();
            continue;
        }

        #ifdef VISUAL
        Mat show_frame;
        if(main_thread) show_frame = frame.clone();
        #endif

        // update tracker_vec
        for (size_t i = 0; i< tracker_vec.size(); i++) {
            tracker_vec[i].update(frame);
        }
        // clean up trackers whose tracking box is overlap
        for (size_t i = 0; i< tracker_vec.size(); i++) {
            for(size_t j = i+1; j< tracker_vec.size(); j++) {
                if(overlap(tracker_vec[i].box, tracker_vec[j].box)){
                    tracker_vec.erase(tracker_vec.begin() + j);
                    j--;
                }
            }
        }

        #ifdef VISUAL
        for (size_t i = 0; i< tracker_vec.size(); i++) {
            
            if(main_thread){
                Rect2d t_box = tracker_vec[i].box;
                rectangle( show_frame, t_box, Scalar( 255, 0, 0 ), 2, 1 );
                Point middleHighPoint = Point(t_box.x+t_box.width/2, t_box.y);
                putText(show_frame, to_string(tracker_vec[i].faceId), middleHighPoint, FONT_HERSHEY_SIMPLEX, 
                                                                        1, Scalar(255, 255, 255), 2);
            }
        }
        #endif
        if (frameCounter % detection_period == 0)
        {
            ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR2RGB, frame.cols, frame.rows);

            #ifdef BENCH_EDGE
            gettimeofday(&tv1,NULL);
            #endif
            mm.detect(ncnn_img, detected_bounding_boxes);
            #ifdef BENCH_EDGE
            gettimeofday(&tv2,NULL);
            LOG(INFO) << "mtcnn ncnn detected one frame, time eclipsed: " <<  getElapse(&tv1, &tv2) << " ms";
            #endif

            int total = 0;
            #ifdef SAVE_IMG
            // debug output
            Mat detect_frame=frame.clone();
            #endif
            // update tracker_vec's bounding boxes and push new tracker int tracker_vec for a new face
            for(vector<Bbox>::iterator it=detected_bounding_boxes.begin(); it!=detected_bounding_boxes.end();it++) {
                if((*it).exist) {
                    total++;
                    // get face bounding box
                    Bbox box = *it;
                    Rect2d detected_face(Point(box.x1, box.y1),Point(box.x2, box.y2));

                    bool newFace = true;
                    unsigned index;
                    for (index=0;index<tracker_vec.size();index++) {
                        if (overlap(detected_face, tracker_vec[index].box)) {
                            newFace = false;
                            break;
                        }
                    }
                    if (newFace) {
                        // create a new tracker if a new face is detected
                        tracker_vec.push_back(face_tracker(faceId,frame,detected_face));
                        LOG(INFO) << "camera["<< camera.NO << "]" << " start tracking face " << tracker_vec[index].faceId << ",tracker index " << index;

                        thisFace = faceId;
                        faceId++;
                    } else {
                        // update tracker's bounding box
                        thisFace = tracker_vec[index].faceId;
                        tracker_vec[index].update_by_dectect(frame,detected_face);
                        LOG(INFO) << "camera["<< camera.NO << "]" << " update tracking face " << tracker_vec[index].faceId <<", tracker index " << index;
                    }

                    // calculate score of face
                    Mat face(frame, detected_face);
                    double score = GetVarianceOfLaplacianSharpness(face);
                    LOG(INFO) << "camera["<< camera.NO << "]" <<" frame[" << frameCounter << "]faceId[" << thisFace 
                              << "], LaplacianSharpness score: " << score;
                    if(score>min_score) {
                        vector<mx_float> face_vec;
                        imgFormConvert(face,face_vec);
                        int infer_age = proc_age(face, face_vec, tracker_vec[index], camera);
                        cv::Mat world_coordinate;
                        if (infer_age >= child_age_min) {
                            // prepare image points for pnp
                            image_points.clear();
                            for(int j =0;j<5;j++){
                                cv::Point2d point(box.ppoint[j],box.ppoint[j+5]);
                                image_points.push_back(point);
                            }

                            bool front_side = compute_coordinate(frame, image_points, camera, world_coordinate, 
                                                                infer_age, frameCounter, thisFace);
                            int new_id=-1;
                            if(front_side) {
                                if(newFace || (!newFace && (tracker_vec[index].reid == -1)) ) {
                                    new_id = proc_embeding(face ,face_vec, tracker_vec[index], camera, frameCounter, thisFace);
                                    #ifdef SAVE_IMG
                                    if(new_id!=-1){
                                        string cmd = "mkdir -p " + output_folder + "/" + to_string(new_id);
                                        system(cmd.c_str());
                                        string output = output_folder + "/" + to_string(new_id) + "/face_" + to_string(thisFace) + "_"+ to_string(frameCounter) + ".jpg";
                                        imwrite(output,face);
                                    }
                                    #endif
                                }
                            } else 
                                LOG(INFO) << "camera["<< camera.NO << "]" <<" frame[" << frameCounter << "]faceId[" << thisFace
                                        << "], pose is skew, don't make face embedding";
                            db.insert(frameCounter, thisFace, index, front_side, score, infer_age, new_id, world_coordinate);
                        } else if(infer_age != -1)
                                LOG(INFO) << "camera["<< camera.NO << "]" <<" frame[" << frameCounter << "]faceId[" << thisFace
                                        << "], can't compute coordinate for child age less than " << child_age_min;
                    } else 
                        LOG(WARNING) << camera.ip <<" frame[" << frameCounter << "]faceId[" << thisFace 
                                     << "], video frame is blur";

                    std::vector<cv::Point2d> detect_points;
                    for(int j =0;j<5;j++){
                        cv::Point2d point(box.ppoint[j],box.ppoint[j+5]);
                        detect_points.push_back(point);
                    }

                    #ifdef SAVE_IMG
                    {
                        // debug output
                        // draw detected face                    
                        rectangle( detect_frame, detected_face, Scalar( 255, 0, 0 ), 2, 1 );
                        Point middleHighPoint = Point(detected_face.x+detected_face.width/2, detected_face.y);
                        putText(detect_frame, to_string(thisFace), middleHighPoint, FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
                        
                        for(auto point: detect_points){
                            drawMarker(detect_frame, point,  cv::Scalar(0, 255, 0), cv::MARKER_CROSS, 10, 1);
                        }                        
                    }
                    #endif
                    #ifdef VISUAL
                    // draw detected face
                    if(main_thread){
                        rectangle( show_frame, detected_face, Scalar( 255, 0, 0 ), 2, 1 );
                        Point middleHighPoint = Point(detected_face.x+detected_face.width/2, detected_face.y);
                        putText(show_frame, to_string(thisFace), middleHighPoint, FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);

                        for(auto point: detect_points){
                            drawMarker(show_frame, point,  cv::Scalar(0, 255, 0), cv::MARKER_CROSS, 10, 1);
                        }
                    }
                    #endif
                }
            }
            #ifdef SAVE_IMG
            // debug output
            string output = output_folder + "/original/" + to_string(frameCounter) + ".jpg";
            imwrite(output,frame);
            output = output_folder + "/detect/"  + to_string(frameCounter) + ".jpg";
            imwrite(output,detect_frame);
            #endif
            #ifdef BENCH_EDGE
            gettimeofday(&tv3,NULL);
            LOG(INFO) << "detected " << total << " Persons. time eclipsed: " <<  getElapse(&tv1, &tv3) << " ms";
            #endif
            // clean up trackers if the tracker doesn't follow a face
            for (size_t i = 0; i < tracker_vec.size(); i++) {
                bool isFace = false;
                for (vector<Bbox>::iterator it=detected_bounding_boxes.begin(); it!=detected_bounding_boxes.end();it++) {
                    if ((*it).exist) {
                        Bbox box = *it;
                        Rect2d detected_face(Point(box.x1, box.y1),Point(box.x2, box.y2));
                        if (overlap(detected_face, tracker_vec[i].box)) {
                            isFace = true;
                            break;
                        }
                    }
                }
                if (!isFace) 
                {
                    LOG(INFO) << "camera["<< camera.NO << "]" << "stop tracking face " << tracker_vec[i].faceId << ", tracker index " << i;
                    #ifdef SAVE_IMG
                    // debug output
                    string output = output_folder + "/tracker/face_" + to_string(tracker_vec[i].faceId) + "_" + to_string(frameCounter) + ".jpg";
                    Rect2d t_face = tracker_vec[i].box;
                    Rect2d frame_rect = Rect2d(0, 0, frame.size().width, frame.size().height);
                    Rect2d roi = t_face & frame_rect;
                    if(roi.area() > 0){
                        Mat tracker_face(frame,roi);
                        imwrite(output,tracker_face);
                    }
                    #endif
                    tracker_vec.erase(tracker_vec.begin() + i);
                    i--;
                }
            }
        }

        frameCounter++;
        #ifdef VISUAL
        if(main_thread){
            Mat small_show_frame;
            if(show_frame.cols>1500)
                resize(show_frame,small_show_frame,frame.size()/2);
            else
                small_show_frame = show_frame;
            
            imshow("window" + camera.ip , small_show_frame);
            if(QUIT_KEY == waitKey(1)) break;
        }
        #endif
        google::FlushLogFiles(google::GLOG_INFO);

    } while (true);
    // } while (QUIT_KEY != waitKey(1));
}

int main(int argc, char* argv[]) {

    const String keys =
        "{help h usage ?     |                    | print this message }"
        "{config             |config.toml         | local_id config file }"
        "{camera             |camera              | folder for camera meta info }"
        "{output             |output              | folder for image sample outputh} "
        "{log                |log                 | folder for log }"
    ;

    CommandLineParser parser(argc, argv, keys);
    parser.about("camera face tracker detector and recognizor");
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    String config_path = parser.get<String>("config");
    cout << "config file: " << config_path << endl;
    String camera_folder = parser.get<String>("camera");
    String output_folder = parser.get<String>("output");
    String log_folder = parser.get<String>("log");

    if (!parser.check()) {
        parser.printErrors();
        return 0;
    }

    google::InitGoogleLogging("multi-camera");
    FLAGS_log_dir = log_folder.c_str();
//    FLAGS_logtostderr = true;

    std::string mtcnn_model_path;
    std::string ckdb_ip;
    int ckdb_port;
    std::string pnp_conf_file;
    std::string mx_model_conf_file;

    try {
        auto g = cpptoml::parse_file(config_path);
        // parse tracker conf
        auto staple_file = g->get_qualified_as<std::string>("tracker.config_file").value_or("staple.yaml");
        auto wisdom_file = g->get_qualified_as<std::string>("tracker.wisdom_file").value_or("wisdom");
        face_tracker::staple_init(wisdom_file,staple_file);
        // parse detector conf
        mtcnn_model_path =  g->get_qualified_as<std::string>("detect.mtccn_model_path").value_or("../models/ncnn");
        detection_period = g->get_qualified_as<int>("detect.period").value_or(20);
        min_score = g->get_qualified_as<int>("detect.min_score").value_or(25);
        // parse db conf
        ckdb_ip = g->get_qualified_as<std::string>("db.ip").value_or("172.19.0.105");
        ckdb_port = g->get_qualified_as<int>("db.port").value_or(9000);
        // parse other conf
        pnp_conf_file = g->get_qualified_as<std::string>("global.pnp_config").value_or("3dpnp.toml");
        mx_model_conf_file = g->get_qualified_as<std::string>("global.mxmodel_config").value_or("mxModel.toml");

    }
    catch (const cpptoml::parse_exception& e) {
        std::cerr << "Failed to parse config.toml: " << e.what() << std::endl;
        exit(1);
    }

    vector<CameraConfig> cameras = LoadCameraConfig(camera_folder);

    CameraConfig main_camera = cameras[cameras.size()-1];
    cameras.pop_back();

    read3D_conf(pnp_conf_file);
    LoadMxModelConf(mx_model_conf_file);
    LoadEmbedConf(mx_model_conf_file);
    LoadAgeConf(mx_model_conf_file);
    LoadVecSearchConf(mx_model_conf_file);

    for (CameraConfig camera: cameras) {
        thread t {process_camera, mtcnn_model_path, camera, output_folder, ckdb_ip, ckdb_port, false};
        t.detach();

    }

    process_camera(mtcnn_model_path, main_camera, output_folder, ckdb_ip, ckdb_port, true);

}
