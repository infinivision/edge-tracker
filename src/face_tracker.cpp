#include "face_tracker.h" // staple trakcer

face_tracker::face_tracker(int faceId_, Mat frame, Rect2d roi , algo_type t_) {
    box = roi;
    faceId = faceId_;
    reid = -1;
    age_sum = 0;
    infer_age_count = 0;
    sample_count_type1 = 0;
    sample_count_type2 = 0;
    last_disappear_frame = -1;
    t = t_;
    staple = nullptr;

    if(t == STAPLE){
        staple = new STAPLE_TRACKER(s_cfg);
        staple->tracker_staple_initialize(frame,roi);
        staple->tracker_staple_train(frame,true);
    }
}

face_tracker::~face_tracker(){
    if(t == STAPLE){
        if(staple!=nullptr)
            delete staple;
    }
}

face_tracker::face_tracker(face_tracker && tracker){
    box             = tracker.box;
    faceId          = tracker.faceId;
    reid            = tracker.reid;
    age_sum         = tracker.age_sum;
    infer_age_count = tracker.infer_age_count;
    sample_count_type1 = tracker.sample_count_type1;
    sample_count_type2 = tracker.sample_count_type2;
    last_disappear_frame = tracker.last_disappear_frame;
    t               = tracker.t;
    staple          = tracker.staple;

    tracker.staple = nullptr;
}

face_tracker & face_tracker::operator=(face_tracker && tracker){
    box             = tracker.box;    
    faceId          = tracker.faceId;
    reid            = tracker.reid;
    age_sum         = tracker.age_sum;
    infer_age_count = tracker.infer_age_count;
    sample_count_type1 = tracker.sample_count_type1;
    sample_count_type2 = tracker.sample_count_type2;
    last_disappear_frame = tracker.last_disappear_frame;
    t               = tracker.t;
    staple          = tracker.staple;

    tracker.staple = nullptr;

    return *this;
}

void face_tracker::update(Mat & frame){
    if(t == STAPLE) {
        box = staple->tracker_staple_update(frame);
        staple->tracker_staple_train(frame,false);
    }
}

void face_tracker::update_by_dectect(Mat & frame, Rect2d roi) {

    if(t == STAPLE) {
        if(staple != nullptr)
            delete staple;
        staple = new STAPLE_TRACKER(s_cfg);
        staple->tracker_staple_initialize(frame,roi);
        staple->tracker_staple_train(frame,true);
        box = roi;
    }

}

void face_tracker::staple_init(std::string wisdom_file, std::string cfg_file){

    if(!STAPLE_TRACKER::importWisdom(wisdom_file)){
        std::cout << "import wisdom file for staple failed, exit! \n";
        exit(0);
    }

    FileStorage fs;
    fs.open(cfg_file, FileStorage::READ);
    staple_cfg cfg;
    cfg.read(fs.root());
    s_cfg = cfg;

}

staple_cfg face_tracker::s_cfg = staple_cfg();