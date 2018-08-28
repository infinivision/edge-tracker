#include "camera.h"
#include <dirent.h>
#include <face_align.h>
#include <glog/logging.h>
#include <iostream>
#include "mtcnn.h"
#include <sys/stat.h>
#include <sys/types.h>
#include "utils.h"
#include <vector>

using namespace std;

FaceAlign faceAlign = FaceAlign();

float getElapse(struct timeval *tv1,struct timeval *tv2)
{
    float t = 0.0f;
    if (tv1->tv_sec == tv2->tv_sec)
        t = (tv2->tv_usec - tv1->tv_usec)/1000.0f;
    else
        t = ((tv2->tv_sec - tv1->tv_sec) * 1000 * 1000 + tv2->tv_usec - tv1->tv_usec)/1000.0f;
    return t;
}

std::string get_current_time() {
    char buffer[26];
    char full[30];
    int millisec;
    struct tm* tm_info;
    struct timeval tv;

    gettimeofday(&tv, NULL);

    millisec = tv.tv_usec % 1000;
    tm_info = localtime(&tv.tv_sec);

    strftime(buffer, 26, "%Y-%m-%d.%H-%M-%S", tm_info);
    sprintf(full, "%s.%03d", buffer, millisec);
    std::string output = full;

    return output;
}

int trave_dir(std::string& path, std::vector<std::string>& file_list)
{
    DIR *d; //声明一个句柄
    struct dirent* dt; //readdir函数的返回值就存放在这个结构体中
    struct stat sb;    
    
    if(!(d = opendir(path.c_str())))
    {
        std::cout << "Error opendir: " << path << std::endl;
        return -1;
    }

    while((dt = readdir(d)) != NULL)
    {
        // std::cout << "file name: " << dt->d_name << std::endl;
        std::string file_name = dt->d_name;
        if(file_name[0] == '.') {
            continue;
        }
        // if(strncmp(dt->d_name, ".", 1) == 0 || strncmp(dt->d_name, "..", 2) == 0) {
        //     continue;
        // }

        std::string file_path = path + "/" + dt->d_name;
        // std::cout << "file path: " << file_path << std::endl;

        if(stat(file_path.c_str(), &sb) < 0) {
            std::cout << "Error stat file: " << file_path << std::endl;
            return -1;
        }

        if (S_ISDIR(sb.st_mode)) {
            // is a directory
            // std::cout << "directory: " << file_path << std::endl;
            trave_dir(file_path, file_list);
        } else {
            // is a regular file
            // std::cout << "file: " << file_path << std::endl;            
            file_list.push_back(file_path); 
        }
    }

    // close directory
    closedir(d);
    return 0;
}

const vector<string> split(const string& s, const char& c) {
    string buff = "";
    vector<string> v;

    for (auto n:s) {
        if (n != c) {
            buff += n;
        } else {
            if (n == c && buff != "") {
                v.push_back(buff);
                buff = "";
            }
        }
    }
    if (buff != "") {
        v.push_back(buff);
    }

    return v;
}

/*
 * Decide whether the detected face is same as the tracking one
 *
 * return true when:
 *   center point of one box is inside the other
 */
bool overlap(const cv::Rect2d &box1, const cv::Rect2d &box2) {
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

    string cmd = "mkdir -p " + output_folder + "/original";
    int dir_err = system(cmd.c_str());
    if (-1 == dir_err) {
        LOG(ERROR) << "Error creating directory";
        exit(1);
    }

    cmd = "mkdir -p " + output_folder + "/aligned";
    dir_err = system(cmd.c_str());
    if (-1 == dir_err) {
        LOG(ERROR) << "Error creating directory";
        exit(1);
    }
}

/*
 * write face to the output folder
 */
void saveFace(const cv::Mat &frame, const Bbox &box, long faceId, string outputFolder) {

    string current_time = get_current_time();

    cv::Rect2d roi(cv::Point(box.x1, box.y1),Point(box.x2, box.y2));
    cv::Mat cropped(frame, roi);
    string output = outputFolder + "/original/" + current_time + ".jpg";
    if ( imwrite(output, cropped) ) {
        LOG(INFO) << "\tsave face #" << faceId << " to " << output;
        cout << "save face #" << faceId << " to " << output << endl;
    } else {
        LOG(ERROR) << "\tfail to save face #" << faceId << " to " << output;
    }

    //namedWindow("output", WINDOW_NORMAL);

    std::vector<cv::Point2f> points;
    for(int num=0;num<5;num++) {
        Point2f point(box.ppoint[num], box.ppoint[num+5]);
        points.push_back(point);
    }

    Mat image = faceAlign.Align(frame, points);

    if (image.empty()) {
        /* empty image means unable to align face */
        return;
    }

    output = outputFolder + "/" + current_time + ".jpg";
    if ( imwrite(output, image) ) {
        LOG(INFO) << "\tsave face #" << faceId << " to " << output;
        cout << "save face #" << faceId << " to " << output << endl;
        LOG(INFO) << "\tmtcnn score: " << box.score;
    } else {
        LOG(ERROR) << "\tfail to save face #" << faceId << " to " << output;
    }

    output = outputFolder + "/aligned/" + current_time + ".jpg";
    imwrite(output, image);

    //imshow("output", image);
}

// int main() {
//     std::string time = get_current_time();

//     std::cout << time << std::endl;

//     return 0;
// }
