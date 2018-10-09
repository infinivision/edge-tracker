clickhouse-client --host 172.19.0.104

CREATE DATABASE IF NOT EXISTS tracker

CREATE TABLE IF NOT EXISTS tracker.sample ( \
	`camera_id`				String,	\
	`frame_count`			UInt64,	\
	`face_id`					UInt64,	\
	`tracker_index`		Int8,		\
	`face_pose_type`	UInt8,	\
	`score`						Float32,\
	`age`   					Int8,		\
	`reid`						Int32,	\
	`sample_date`			Date,		\
	`time_stamp`			UInt64,	\
	`coordinate-x` 		Float32,\
	`coordinate-y` 		Float32,\
	`coordinate-z` 		Float32 \
) ENGINE = MergeTree(sample_date, time_stamp, (sample_date,time_stamp), 8192)


// DROP TABLE IF EXISTS tracker.sample
// DROP DATABASE IF EXISTS tracker

select * from tracker.sample

docker run -itd --name ck-svr --rm  --network host -v /root/dev_keeper/ckdb/data:/var/lib/clickhouse -v /root/dev_keeper/ckdb/clickhouse_config.xml:/etc/clickhouse-server/config.xml  yandex/clickhouse-server 

