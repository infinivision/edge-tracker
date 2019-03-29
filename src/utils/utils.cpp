#include <algorithm>
#include "camera.h"
#include <dirent.h>
#include <glog/logging.h>
#include <iostream>
#include "mtcnn.h"
#include <sys/stat.h>
#include <sys/types.h>
#include "utils.h"
#include <vector>

using namespace std;

#define MAX_DETECTED_FACE_SIZE 160

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

// prepare (clean output folder)
// args output_folder will append camera ip address
void prepare_output_folder(const CameraConfig &camera, string &output_folder) {
    output_folder += "/" + camera.identity();
}

/*
 * write face to the output folder
 */
void saveFace(const cv::Mat &frame, const Bbox &box, long faceId, string outputFolder) {

    string current_time = get_current_time();

    cv::Rect2d roi(cv::Point(box.x1, box.y1),Point(box.x2, box.y2));

    // crop a 1.5x larger face bounding box
    float factor = 1.5; // make the face roi 1.5x larger
    resizeBoundingBox(frame, roi, factor);
    cv::Mat image(frame, roi);
    cv::Mat resized_image; // resize image if long side is larger than MAX_DETECTED_FACE_SIZE

    // resize image to long side MAX_DETECTED_FACE_SIZE
    if (image.rows > image.cols) {
        if (image.rows > MAX_DETECTED_FACE_SIZE) {
            int new_rows = MAX_DETECTED_FACE_SIZE;
            int new_cols = floor(image.cols * MAX_DETECTED_FACE_SIZE / image.rows);
            cv::resize(image, resized_image, cv::Size(new_cols, new_rows));
        } else {
            resized_image = image;
        }

    } else if (image.cols > MAX_DETECTED_FACE_SIZE) {
        int new_cols = MAX_DETECTED_FACE_SIZE;
        int new_rows = floor(image.rows * MAX_DETECTED_FACE_SIZE / image.cols);
        cv::resize(image, resized_image, cv::Size(new_cols, new_rows));
    } else {
        resized_image = image;
    }

    string output = outputFolder + "/" + current_time + ".jpg";
    if ( imwrite(output, resized_image) ) {
        LOG(INFO) << "\tsave face #" << faceId << " to " << output;
        cout << "save face #" << faceId << " to " << output << endl;
        LOG(INFO) << "\tmtcnn score: " << box.score;
    } else {
        LOG(ERROR) << "\tfail to save face #" << faceId << " to " << output;
    }
}

// scale the roi to a factor, inside an image
void resizeBoundingBox(const cv::Mat &frame, cv::Rect2d &roi, float factor) {
    float center_x = roi.x + roi.width / 2.0;
    float center_y = roi.y + roi.height / 2.0;
    int new_x = floor(center_x - factor * roi.width / 2);
    int new_y = floor(center_y - factor * roi.height / 2);
    int new_width = ceil(factor * roi.width);
    int new_height = ceil(factor * roi.height);

    if (new_x < 0) new_x = 0;
    if (new_y < 0) new_y = 0;
    if (new_x + new_width > frame.cols) new_width = frame.cols - new_x;
    if (new_y + new_height > frame.rows) new_height = frame.rows - new_y;

    roi.x = new_x;
    roi.y = new_y;
    roi.width = new_width;
    roi.height = new_height;
}

// int main() {
//     std::string time = get_current_time();

//     std::cout << time << std::endl;

//     return 0;
// }
