clickhouse-client --host 172.19.0.104

CREATE DATABASE IF NOT EXISTS tracker

CREATE TABLE IF NOT EXISTS tracker.coordinate ( \
	`camera_id`				String,	\
	`face_id`					UInt64,	\
	`frame_count`			UInt64,	\
	`tracker_index`		Int8,		\
	`face_pose_type`	UInt8,	\
	`score`						Float32,\
	`age`   					Int8,		\
	`sample_date`			Date,		\
	`time_stamp`			UInt64,	\
	`coordinate-x` 		Float32,\
	`coordinate-y` 		Float32,\
	`coordinate-z` 		Float32 \
) ENGINE = MergeTree(sample_date, time_stamp, (sample_date,time_stamp), 8192)

CREATE TABLE IF NOT EXISTS tracker.reid ( \
	`camera_id`				String,	\
	`face_id`					UInt64,	\
	`reid`						Int32,	\
	`sample_date`			Date,		\
	`time_stamp`			UInt64	\
) ENGINE = MergeTree(sample_date, time_stamp, (sample_date,time_stamp), 8192)

// DROP TABLE IF EXISTS tracker.coordinate
// DROP TABLE IF EXISTS tracker.reid
// DROP DATABASE IF EXISTS tracker

select reid,point_num from (	\
		select reid,count(1) as point_num from tracker.reid \
					any inner join tracker.coordinate \
					using camera_id,face_id	 \
					where sample_date='2018-10-11'	\
					group by reid ) \
		where point_num > 3 \
		order by point_num
				
select time_stamp, `coordinate-x` from tracker.coordinate	\
			any inner join tracker.reid	\
			using camera_id,face_id	 \
			where    sample_date='2018-10-11' \
					 and reid = 4 \
			order by time_stamp
	

docker run -itd --name ck-svr --rm  --network host -v /root/dev_keeper/ckdb/data:/var/lib/clickhouse -v /root/dev_keeper/ckdb/clickhouse_config.xml:/etc/clickhouse-server/config.xml  yandex/clickhouse-server 

