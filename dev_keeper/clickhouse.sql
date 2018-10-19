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

// trace map
// eliminate noise point by select correct reid
select reid,point_num from (	\
		select reid,count(1) as point_num from tracker.coordinate \
					any inner join  tracker.reid\
					using camera_id,face_id	 \
					where sample_date='2018-10-11'	\
					group by reid ) \
		where point_num > 10 \
		order by point_num
				
select time_stamp, `coordinate-x`, `coordinate-y` from tracker.coordinate	\
			any inner join tracker.reid	\
			using camera_id,face_id	 \
			where    sample_date='2018-10-11' \
					 and reid = 1 \
			order by time_stamp
	

// heat map
// eliminate noise point by select correct reid
select reid,point_num from (	\
		select reid,count(1) as point_num from tracker.coordinate \
					any inner join  tracker.reid\
					using camera_id,face_id	 \
					where sample_date='2018-10-11'	\
					group by reid ) \
		where point_num > 10 \
		order by reid
		
// coordinate metric:  decimetre		
select x, y, count(1) from ( \
	select cast(`coordinate-x`/100 as int) as x,cast(`coordinate-y`/100 as int) as y  from tracker.coordinate	\
			any inner join tracker.reid	\
			using camera_id,face_id	 \
			where    sample_date='2018-10-11' \
					 and reid in ( 1,6,7,8,12,16,21 ) \
	) group by x, y order by x,y

// coordinate metric:  metre		
select x, y, count(1) from ( \
	select cast(`coordinate-x`/1000 as int) as x,cast(`coordinate-y`/1000 as int) as y  from tracker.coordinate	\
			any inner join tracker.reid	\
			using camera_id,face_id	 \
			where    sample_date='2018-10-11' \
					 and reid in ( 1,6,7,8 ) \
	) group by x, y order by x,y
	
docker run -itd --name ck-svr --rm  --network host -v /root/dev_keeper/ckdb/data:/var/lib/clickhouse -v /root/dev_keeper/ckdb/clickhouse_config.xml:/etc/clickhouse-server/config.xml  yandex/clickhouse-server 
