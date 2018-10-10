#ifndef __FACE_TRACKER__
#define __FACE_TRACKER__

//#include <opencv2/tracking.hpp>
//#include "tracker.hpp" // use optimised tracker instead of OpenCV version of KCF tracker
#include "staple_tracker.hpp" // staple trakcer
#include <opencv2/opencv.hpp>

using namespace cv;

// class face_tracker is wrapper class for multi tracking algorithm, such as staple,kcf;
// face_tracker also collect addition info such as face id , reid, age of face and so on.

class face_tracker {

public:
    enum algo_type {
        STAPLE
    };

    face_tracker(int faceId_, Mat frame, Rect2d roi , algo_type t_ = STAPLE);
    face_tracker(face_tracker && tracker);
    face_tracker & operator=(face_tracker && tracker);
    ~face_tracker();

    void update(Mat & frame);

    void update_by_dectect(Mat & frame, Rect2d roi);

    static void staple_init(std::string wisdom_file, std::string cfg_file);

    Rect2d box;
    int faceId;
    int reid;
    int age_sum;
    int infer_age_count;
    int sample_count_type1;
    int sample_count_type2;
    int last_disappear_frame;

private:
    algo_type t;
    STAPLE_TRACKER * staple;

    static staple_cfg s_cfg;

    face_tracker(const face_tracker &);
    face_tracker & operator=(const face_tracker &);
    
};

#endif