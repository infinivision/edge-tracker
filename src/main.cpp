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
extern int child_age_min;
bool compute_coordinate( const cv::Mat im, const std::vector<cv::Point2d> & image_points, const CameraConfig & camera, \
                         cv::Mat & world_coordinate, int age, int frameCount,int faceId);
void read3D_conf();

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

int proc_age(vector<mx_float> face_vec, face_tracker & target) {
    if(target.infer_age_count < n_age_sample) {
        int age=0;
        if(age_enable){
            std::vector<float> age_vec;
            #ifdef BENCH_EDGE
            struct timeval  tv_age;
            gettimeofday(&tv_age,NULL);
            long t_ms1_age = tv_age.tv_sec * 1000 * 1000 + tv_age.tv_usec;
            #endif
            Infer(age_hd,face_vec, age_vec);
            for(size_t j = 2; j<age_vec.size()-1; j+=2){
                if(age_vec[j]<age_vec[j+1])
                    age++;
            }
            LOG(INFO) << "target infer age: " << age;
            #ifdef BENCH_EDGE
            gettimeofday(&tv_age,NULL);
            long t_ms2_age = tv_age.tv_sec * 1000 * 1000 + tv_age.tv_usec;
            infer_count_age++;
            if(infer_count_age>1){
                sum_t_infer_age += t_ms2_age-t_ms1_age;
                LOG(INFO) << "face infer age performance: [" << (sum_t_infer_age/1000.0 ) / (infer_count_age-1) << "] mili second latency per time";
            }
            #endif
        } else 
            age = 20;
        target.infer_age_count++;
        target.age_sum += age;
    }
    return target.age_sum / target.infer_age_count;
}


int proc_embeding(vector<mx_float> face_vec, face_tracker & target, 
                 const CameraConfig & camera, int frameCounter, int thisFace) {

    vector<float> face_embed_vec;

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

    int new_id = proc_embd_vec(face_embed_vec, camera, frameCounter, thisFace);
    target.reid = new_id;
    return new_id;
}

void process_camera(const string model_path, const CameraConfig &camera, string output_folder, const String video_file, bool mainThread) {

    cout << "processing camera: " << camera.identity() << endl;

    prepare_output_folder(camera, output_folder);
    
    MTCNN mm(model_path);

    VideoCapture cap = camera.GetCapture();

    if(video_file!="none")
        cap = VideoCapture(video_file);
    
    if (!cap.isOpened()) {
        cerr << "failed to open camera" << endl;
        return;
    }

    int frameCounter = 0;
    long faceId = 0;
    long thisFace = 0;
    struct timeval  tv1,tv2,tv3;
    struct timeval  ts;
    struct timezone tz1,tz2;

    vector<Bbox> detected_bounding_boxes;
    vector<face_tracker> tracker_vec;
    vector<cv::Point2d> image_points;
    Mat frame;

    do {
        detected_bounding_boxes.clear();
        cap >> frame;
        if (!frame.data) {

            if(video_file!="none"){
                std::cout << "video file read over!\n" ;
                exit(0);
            }

            LOG(ERROR) << "Capture video failed: " << camera.identity() << ", opened: " << cap.isOpened();
            cap.release();

            LOG(ERROR) << "sleep for 5 seconds ...";
            std::this_thread::sleep_for(std::chrono::seconds(5));
            
            cap = camera.GetCapture();
            if (!cap.isOpened()) {
                LOG(ERROR) << "failed to open camera: " << camera.identity();
                return;
            }
            LOG(INFO) << "reopen camera: " << camera.identity();
            continue;
        }

        #ifdef VISUAL
        Mat show_frame = frame.clone();
        #endif

        // update tracker_vec
        for (size_t i = 0; i< tracker_vec.size(); i++){
            tracker_vec[i].update(frame);

            // to do
            // need to detect duplicated tracking face

            #ifdef VISUAL
            Rect2d t_box = tracker_vec[i].box;
            rectangle( show_frame, t_box, Scalar( 255, 0, 0 ), 2, 1 );
            Point middleHighPoint = Point(t_box.x+t_box.width/2, t_box.y);
            putText(show_frame, to_string(tracker_vec[i].faceId), middleHighPoint, FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
            #endif
        }
        
        if (frameCounter % camera.detection_period == 0)
        {
            ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR2RGB, frame.cols, frame.rows);

            #ifdef BENCH_EDGE
            gettimeofday(&tv1,&tz1);
            #endif
            mm.detect(ncnn_img, detected_bounding_boxes);
            #ifdef BENCH_EDGE
            gettimeofday(&tv2,&tz2);
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
                        LOG(INFO) << "start tracking face " << tracker_vec[index].faceId << ",tracker index " << index;

                        thisFace = faceId;
                        faceId++;
                    } else {
                        // update tracker's bounding box
                        thisFace = tracker_vec[index].faceId;
                        tracker_vec[index].update_by_dectect(frame,detected_face);
                        LOG(INFO) << "update tracking face " << tracker_vec[index].faceId <<", tracker index " << index;
                    }

                    // calculate score of face
                    Mat face(frame, detected_face);
                    double score = GetVarianceOfLaplacianSharpness(face);
                    LOG(INFO) << camera.ip <<" frame[" << frameCounter << "]faceId[" << thisFace 
                              << "], LaplacianSharpness score: " << score;
                    if(score>min_score) {
                        vector<mx_float> face_vec;
                        imgFormConvert(face,face_vec);
                        int infer_age = proc_age(face_vec, tracker_vec[index]);
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
                            if(front_side && newFace) {
                                int new_id = proc_embeding(face_vec, tracker_vec[index], camera, frameCounter, thisFace);
                                #ifdef SAVE_IMG
                                string cmd = "mkdir -p " + output_folder + "/" + to_string(new_id);
                                system(cmd.c_str());
                                string output = output_folder + "/" + to_string(new_id) + "/face_" + to_string(thisFace) + "_"+ to_string(frameCounter) + ".jpg";
                                imwrite(output,face);
                                #endif
                            }
                            if(front_side){
                                gettimeofday(&ts,NULL);
                                long int ts_ms = ts.tv_sec * 1000 + ts.tv_usec / 1000;
                                // to do: push the coordinate reid timestamp info into the time series database
                                // (world_coordinate, reid, ts_ms)
                            } else 
                                LOG(INFO) << camera.ip <<" frame[" << frameCounter << "]faceId[" << thisFace
                                        << "], pose is skew, don't make face embedding";
                        } else 
                                LOG(INFO) << camera.ip <<" frame[" << frameCounter << "]faceId[" << thisFace
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
                    // debug output
                    // draw detected face                    
                    rectangle( detect_frame, detected_face, Scalar( 255, 0, 0 ), 2, 1 );
                    Point middleHighPoint = Point(detected_face.x+detected_face.width/2, detected_face.y);
                    putText(detect_frame, to_string(thisFace), middleHighPoint, FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
                    
                    for(auto point: detect_points){
                        drawMarker(detect_frame, point,  cv::Scalar(0, 255, 0), cv::MARKER_CROSS, 10, 1);
                    }
                    #endif
                    #ifdef VISUAL
                    // draw detected face
                    if(mainThread){

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
                    LOG(INFO) << "stop tracking face " << tracker_vec[i].faceId << ", tracker index " << i;
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
        if( mainThread) {
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
        "{model        |../models/ncnn             | path to mtcnn model  }"
        "{config       |config.toml                | camera config        }"
        "{output       |output                     | output folder        }"
        "{video-file   |none                       | use video file instead of camera stream }"
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
    String video_file = parser.get<String>("video-file");
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
        thread t {process_camera, model_path, camera, output_folder, video_file, false};
        t.detach();
 
    }

    process_camera(model_path,demo,output_folder,video_file,true);
/*
    while(true) {
        std::this_thread::sleep_for(chrono::seconds(1));
    }
*/
}
