#include <iostream>
#include <opencv2/opencv.hpp>
//#include <opencv2/tracking.hpp>
#include "camera.h"
#include "mtcnn.h"
#include <string.h>
#include <chrono>
#include <cstdlib>
//#include "tracker.hpp" // use optimised tracker instead of OpenCV version of KCF tracker
#include "staple_tracker.hpp" // staple trakcer
#include "utils.h"
#include "face_attr.h"
#include "face_align.h"
#include "face_predict.h"
#include <glog/logging.h>
#include <thread>

#define QUIT_KEY 'q'

using namespace std;
using namespace cv;

#ifdef BENCH_EDGE
long  sum_t_infer_embed  = 0;
long  infer_count_embed  = 0;
long  sum_t_infer_age  = 0;
long  infer_count_age  = 0;
#endif

extern int min_score;
bool compute_coordinate( const cv::Mat im, const std::vector<cv::Point2d> & image_points, const CameraConfig & camera, \
                         cv::Mat & world_coordinate, int age, int sex, int frameCount,int faceId);
void read3D_conf();

FaceAlign faceAlign = FaceAlign();

/*
 * Decide whether the detected face is same as the tracking one
 * 
 * return true when:
 *   center point of one box is inside the other
 */
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
    cmd = "mkdir -p " + output_folder + "/tracker";
    dir_err = system(cmd.c_str());
    if (-1 == dir_err) {
        LOG(ERROR) << "Error creating directory";
        exit(1);
    }    
}

void process_camera(const string model_path, const CameraConfig &camera, string output_folder, bool mainThread) {

    cout << "processing camera: " << camera.identity() << endl;

    prepare_output_folder(camera, output_folder);
    
    MTCNN mm(model_path);
    FaceAttr fa;
    fa.Load();
    FaceAlign align;

    VideoCapture cap = camera.GetCapture();
    if (!cap.isOpened()) {
        cerr << "failed to open camera" << endl;
        return;
    }

    int frameCounter = 0;
    long faceId = 0;
    long thisFace = 0;
    struct timeval  tv1,tv2;
    struct timeval  ts;
    struct timezone tz1,tz2;

    vector<Bbox> detected_bounding_boxes;
    Rect2d roi;
    vector<STAPLE_TRACKER *>  trackers;
    vector<Rect2d> tracker_boxes;
    vector<long int> reids;
    Mat frame;

    FileStorage fs;
    fs.open("staple.yaml", FileStorage::READ);
    staple_cfg staple_cfg;
    staple_cfg.read(fs.root());

    // namedWindow("window", WINDOW_NORMAL);
    std::vector<cv::Point2d> image_points;
    do {
        detected_bounding_boxes.clear();
        bool enable_detection = false;
        cap >> frame;
        if (!frame.data) {
            cerr << "Capture video failed" << endl;
            continue;
        }

        #ifdef VISUAL
        Mat show_frame = frame.clone();
        #endif

        string log = "frame #" + to_string(frameCounter) + ", tracking faces: ";
        // update trackers
        for (int i = 0; i < trackers.size(); i++) {
            STAPLE_TRACKER *tracker = trackers[i];
            tracker_boxes[i] = tracker->tracker_staple_update(frame);
            tracker->tracker_staple_train(frame,false);
            log += "#" + to_string(trackers[i]->id) + " ";
        }
        //LOG(INFO) << log;
        
        if (frameCounter % camera.detection_period == 0)
        {
            enable_detection = true;
            ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR2RGB, frame.cols, frame.rows);

            gettimeofday(&tv1,&tz1);
            mm.detect(ncnn_img, detected_bounding_boxes);
            gettimeofday(&tv2,&tz2);
            int total = 0;

            // update trackers' bounding boxes and create tracker for a new face
            for(vector<Bbox>::iterator it=detected_bounding_boxes.begin(); it!=detected_bounding_boxes.end();it++) {
                if((*it).exist) {
                    total++;
                    // get face bounding box
                    Bbox box = *it;
                    Rect2d detected_face(Point(box.x1, box.y1),Point(box.x2, box.y2));

                    bool newFace = true;
                    unsigned i;
                    for (i=0;i<tracker_boxes.size();i++) {
                        if (overlap(detected_face, tracker_boxes[i])) {
                            newFace = false;
                            break;
                        }
                    }
                    if (newFace) {
                        // create a new tracker if a new face is detected
                        STAPLE_TRACKER * tracker = new STAPLE_TRACKER(staple_cfg);
                        tracker->tracker_staple_initialize(frame,detected_face);
                        tracker->tracker_staple_train(frame,true);
                        tracker->id = faceId;
                        trackers.push_back(tracker);
                        tracker_boxes.push_back(detected_face);
                        reids.push_back(-1);
                        LOG(INFO) << "start tracking face " << tracker->id << ",tracker i " << trackers.size()-1;

                        thisFace = faceId;
                        faceId++;
                    } else {
                        // update tracker's bounding box
                        STAPLE_TRACKER * tracker = trackers[i];
                        long id_ = tracker->id;
                        LOG(INFO) << "update tracking face " << tracker->id <<",tracker i " << i;
                        delete tracker;
                        tracker = new STAPLE_TRACKER(staple_cfg);
                        tracker->id = id_;
                        thisFace = id_;
                        tracker->tracker_staple_initialize(frame,detected_face);
                        tracker->tracker_staple_train(frame,true);
                        trackers[i] = tracker;
                    }

                    // calculate score of face
                    Mat face(frame, detected_face);
                    bool front_side =false;
                    double score = fa.GetVarianceOfLaplacianSharpness(face);
                    LOG(INFO) << camera.ip <<" frame[" << frameCounter << "]faceId[" << thisFace 
                              << "], LaplacianSharpness score: " << score;
                    if(score>min_score){
                        image_points.clear();
                        for(int i =0;i<5;i++){
                            cv::Point2d point(box.ppoint[i],box.ppoint[i+5]);
                            image_points.push_back(point);
                        }
                        std::vector<mx_float> face_vec;            
                        std::vector<float> age_vec;
                        imgFormConvert(face,face_vec);

                        #ifdef BENCH_EDGE
                        struct timeval  tv_age;
                        gettimeofday(&tv_age,NULL);
                        long t_ms1_age = tv_age.tv_sec * 1000 * 1000 + tv_age.tv_usec;
                        #endif
                        int g = 0;
                        Infer(age_hd,face_vec, age_vec);
                        if(age_vec[0]>age_vec[1]){
                            LOG(INFO) << "target gender: female";
                            g = 0;
                        }
                            
                        else{
                            LOG(INFO) << "target gender: male";
                            g = 1;
                        }

                        int age=0;
                        for(size_t i = 2; i<age_vec.size()-1; i+=2){
                            if(age_vec[i]<age_vec[i+1])
                                age++;
                        }
                        LOG(INFO) << "target age: " << age;

                        #ifdef BENCH_EDGE
                        gettimeofday(&tv_age,NULL);
                        long t_ms2_age = tv_age.tv_sec * 1000 * 1000 + tv_age.tv_usec;
                        infer_count_age++;
                        if(infer_count_age>1){
                            sum_t_infer_age += t_ms2_age-t_ms1_age;
                            LOG(INFO) << "face infer age performance: [" << (sum_t_infer_age/1000.0 ) / (infer_count_age-1) << "] mili second latency per time";
                        }                            
                        #endif

                        cv::Mat world_coordinate;
                        front_side = compute_coordinate(frame, image_points, camera, world_coordinate, age, g,frameCounter, thisFace);
                        
                        if(front_side && newFace){
                            std::vector<float> face_embed_vec;

                            #ifdef BENCH_EDGE
                            struct timeval  tv;
                            gettimeofday(&tv,NULL);
                            long t_ms1 = tv.tv_sec * 1000 * 1000 + tv.tv_usec;
                            #endif

                            Infer(embd_hd,face_vec,face_embed_vec);

                            #ifdef BENCH_EDGE
                            gettimeofday(&tv,NULL);
                            long t_ms2 = tv.tv_sec * 1000 * 1000 + tv.tv_usec;
                            infer_count_embed++;
                            if(infer_count_embed>1){
                                sum_t_infer_embed += t_ms2-t_ms1;
                                LOG(INFO) << "face infer embeding performance: [" << (sum_t_infer_embed/1000.0 ) / (infer_count_embed-1) << "] mili second latency per time";
                            }
                            #endif

                            int new_id;
                            new_id = proc_embd_vec(face_embed_vec, camera, frameCounter, thisFace);

                            #ifdef SAVE_IMG
                            // debug output
                            string cmd = "mkdir -p " + output_folder + "/" + to_string(new_id);
                            system(cmd.c_str());
                            #endif

                            if(reids[i]==-1){
                                reids[i] = new_id;
                                LOG(INFO) << camera.ip <<" frame[" << frameCounter << "]faceId[" << thisFace
                                          << "], reid: " << new_id;
                            } else {
                                if(reids[i]!=new_id){
                                    LOG(WARNING) << camera.ip <<" frame[" << frameCounter << "]faceId[" << thisFace
                                             << "], tracker reid change from["<<reids[i] << "],to["<<new_id<<"]";
                                    reids[i]=new_id;
                                }
                                else
                                    LOG(INFO) << camera.ip <<" frame[" << frameCounter << "]faceId[" << thisFace
                                              << "], reid: " << new_id;
                            }

                            {   
                                #ifdef SAVE_IMG
                                // debug output
                                Mat frame2=frame.clone();
                                rectangle( frame2, detected_face, Scalar( 255, 0, 0 ), 2, 1 );
                                // show face id
                                Point middleHighPoint = Point(detected_face.x+detected_face.width/2, detected_face.y);
                                putText(frame2, to_string(thisFace), middleHighPoint, FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
                                for(auto point: image_points){
                                    drawMarker(frame2, point,  cv::Scalar(0, 255, 0), cv::MARKER_CROSS, 10, 1);
                                }
                                string output = output_folder + "/face_" + to_string(thisFace) + "_" + to_string(frameCounter) + ".jpg";
                                imwrite(output,frame2);

                                output = output_folder + "/" + to_string(new_id) + "/face_" + to_string(thisFace) + "_"+ to_string(frameCounter) + ".jpg";
                                imwrite(output,face);
                                #endif
                            }
                            // PrintOutputResult(face_embed_vec);
                        }
                        if(front_side==false)
                            LOG(INFO) << camera.ip <<" frame[" << frameCounter << "]faceId[" << thisFace
                                      << "], pose is skew, don't make face embedding";
                        else {
                            gettimeofday(&ts,NULL);
                            long int ts_ms = ts.tv_sec * 1000 + ts.tv_usec / 1000;
                            // to do: push the coordinate reid timestamp info into the time series database
                            // (world_coordinate, reid[i], ts_ms)
                        }
                    }
                    else 
                        LOG(WARNING) << camera.ip <<" frame[" << frameCounter << "]faceId[" << thisFace 
                                     << "], video frame is blur";
                    #ifdef VISUAL
                    // draw detected face
                    if(mainThread){
                        rectangle( show_frame, detected_face, Scalar( 255, 0, 0 ), 2, 1 );
                        // show face id
                        Point middleHighPoint = Point(detected_face.x+detected_face.width/2, detected_face.y);
                        putText(show_frame, to_string(thisFace), middleHighPoint, FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
                        for(auto point: image_points){
                            drawMarker(show_frame, point,  cv::Scalar(0, 255, 0), cv::MARKER_CROSS, 10, 1);
                        }
                    }
                    #endif
                    #ifdef SAVE_IMG
                    // debug output
                    string output = output_folder + "/original/"+to_string(thisFace) + "_" + to_string(frameCounter) + ".jpg";
                    imwrite(output,frame);
                    #endif
                }
            }
            //LOG(INFO) << "trackers size after decect: " << trackers.size();
            LOG(INFO) << "detected " << total << " Persons. time eclipsed: " <<  getElapse(&tv1, &tv2) << " ms";
            // clean up trackers if the tracker doesn't follow a face
            for (unsigned i=0; i < trackers.size(); i++) {
                STAPLE_TRACKER *tracker = trackers[i];
                Rect2d tracker_box = tracker_boxes[i];
                bool isFace = false;
                for (vector<Bbox>::iterator it=detected_bounding_boxes.begin(); it!=detected_bounding_boxes.end();it++) {
                    if ((*it).exist) {
                        Bbox box = *it;

                        Rect2d detected_face(Point(box.x1, box.y1),Point(box.x2, box.y2));
                        if (overlap(detected_face, tracker_boxes[i])) {
                            isFace = true;                                                          
                            break;
                        }
                    }
                }
                if (!isFace) 
                {
                    LOG(INFO) << "stop tracking face " << tracker->id <<",tracker i " << i;
                    #ifdef SAVE_IMG
                    // debug output
                    string output = output_folder + "/tracker/face_" + to_string(trackers[i]->id) + "_" + to_string(frameCounter) + ".jpg";
                    Rect2d t_face = tracker_boxes[i];
                    Rect2d frame_rect = Rect2d(0, 0, frame.size().width, frame.size().height);
                    Rect2d roi = t_face & frame_rect;
                    if(roi.area() > 0){
                        Mat tracker_face(frame,roi);
                        imwrite(output,tracker_face);
                    }

                    output = output_folder + "/tracker/" + to_string(trackers[i]->id) + "_" + to_string(frameCounter) + ".jpg";
                    imwrite(output,frame);
                    #endif

                    delete tracker;
                    trackers.erase(trackers.begin() + i);
                    tracker_boxes.erase(tracker_boxes.begin() + i);
                    reids.erase(reids.begin() + i);
                    i--;
                }
            }
        }

        frameCounter++;
        #ifdef VISUAL

        if(enable_detection && mainThread) {
            Mat small_show_frame;
            if(show_frame.cols>1500)
                resize(show_frame,small_show_frame,frame.size()/2);
            else
                small_show_frame = show_frame;
            
            // LOG(INFO) << "camera ip: "<< camera.ip;
            imshow("window" + camera.ip , small_show_frame);
            if(QUIT_KEY == waitKey(1)) break;
        }

        #endif
        google::FlushLogFiles(google::GLOG_INFO);

    } while (true);
    // } while (QUIT_KEY != waitKey(1));
}

int main(int argc, char* argv[]) {

    google::InitGoogleLogging("multi-camera-trakcing");
    FLAGS_log_dir = "./";
//    FLAGS_logtostderr = true;
    read3D_conf();
    LoadMxModelConf();
    const String keys =
        "{help h usage ? |                         | print this message   }"
        "{model        |models/ncnn                | path to mtcnn model  }"
        "{config       |config.toml                | camera config        }"
        "{output       |/Users/moon/Pictures/faces | output folder        }"
    ;

    CommandLineParser parser(argc, argv, keys);
    parser.about("camera face detector");
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    String config_path = parser.get<String>("config");
    LOG(INFO) << "config path: " << config_path;

    String model_path = parser.get<String>("model");
    String output_folder = parser.get<String>("output");
    if (!parser.check()) {
        parser.printErrors();
        return 0;
    }

    vector<CameraConfig> cameras = LoadCameraConfig(config_path);

    // LOG(INFO) << "detection period: " << camera.detection_period;

    CameraConfig demo = cameras[cameras.size()-1];
    cameras.pop_back();

    for (CameraConfig camera: cameras) {
        // start processing video
        thread t {process_camera, model_path, camera, output_folder,false};
        t.detach();
 
    }

    process_camera(model_path,demo,output_folder,true);
/*
    while(true) {
        std::this_thread::sleep_for(chrono::seconds(1));
    }
*/
}
