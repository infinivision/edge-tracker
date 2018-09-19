#include "face_sample.h"
#include <glog/logging.h>
#include <sys/time.h>
#include <ctime>

dbHandle::dbHandle(std::string db_server, std::string camera_ip)
try:
    client(ClientOptions().SetHost(db_server)),
    camera_id(camera_ip)
{
    std::cout << "init client for click house server: " << db_server << std::endl;
}
catch (clickhouse::ServerException & exception) {
    LOG(ERROR) << "click house client init exception: " << exception.what() << std::endl;
    exit(0);
}
catch(std::system_error & exception){
    LOG(ERROR) << "click house client init exception, system error: " << exception.what() << std::endl;
    exit(0);
}

void dbHandle::insert(long frameCounter, long faceId, unsigned tracker_index, \
                    bool front_side, float score, int age, int reid, cv::Mat & coordinate) {

    try{

        Block block;

        auto camera_id_ = std::make_shared<ColumnString>();
        camera_id_->Append(camera_id);
        block.AppendColumn("camera_id" , camera_id_);

        auto frameCounter_ = std::make_shared<ColumnUInt64>();
        frameCounter_->Append(frameCounter);
        block.AppendColumn("frame_count" , frameCounter_);
        
        auto faceId_ = std::make_shared<ColumnUInt64>();
        faceId_->Append(faceId);
        block.AppendColumn("face_id" , faceId_);

        auto tracker_index_ = std::make_shared<ColumnInt8>();
        tracker_index_->Append(tracker_index);
        block.AppendColumn("tracker_index" , tracker_index_);

        auto front_side_ = std::make_shared<ColumnUInt8>();
        front_side_->Append(front_side);
        block.AppendColumn("front_side" , front_side_);

        auto score_ = std::make_shared<ColumnFloat32>();
        score_->Append(score);
        block.AppendColumn("score", score_);

        auto age_ = std::make_shared<ColumnInt8>();
        age_->Append(age);
        block.AppendColumn("age", age_);

        auto reid_ = std::make_shared<ColumnInt32>();
        reid_->Append(reid);
        block.AppendColumn("reid", reid_);

        auto sample_date = std::make_shared<ColumnDate>();
        sample_date->Append(std::time(nullptr));
        block.AppendColumn("sample_date", sample_date);

        struct timeval  ts;
        gettimeofday(&ts,NULL);
        long int ts_ms = ts.tv_sec * 1000 + ts.tv_usec / 1000;
        LOG(INFO) << "time stamp " << ts_ms;
        auto time_stamp = std::make_shared<ColumnUInt64>();
        time_stamp->Append(ts_ms);
        block.AppendColumn("time_stamp", time_stamp);

        auto x = coordinate.at<double>(0,0);
        auto y = coordinate.at<double>(1,0);
        auto z = coordinate.at<double>(2,0);
        auto x_ = std::make_shared<ColumnFloat32>();
        x_->Append(x);
        block.AppendColumn("coordinate-x", x_);
        auto y_ = std::make_shared<ColumnFloat32>();
        y_->Append(y);
        block.AppendColumn("coordinate-y", y_);
        auto z_ = std::make_shared<ColumnFloat32>();
        z_->Append(z);
        block.AppendColumn("coordinate-z", z_);

        client.Insert("tracker.sample", block);

        // std::cout << "insert one row into table tracker.sample\n";
    } 
    catch (clickhouse::ServerException & exception) {
        LOG(ERROR) << "click house access exception " << exception.what() << std::endl;
        exit(0);
    }
    catch(std::system_error & exception){
        LOG(ERROR) << "click house access exception, system error: " << exception.what() << std::endl;
        exit(0);
    }
}
