#!/bin/sh

g++ -v -std=c++14 src/main.cpp src/utils.cpp src/mtcnn.cpp src/face_attr.cpp src/face_align.cpp src/camera.cpp src/image_quality.cpp -o bin/main -pthread -fopenmp -Iinclude -I/usr/local/include/ncnn -I/usr/local/include/tracker -I/usr/local/include/opencv -I/usr/include/libpng12 -L/usr/local/share/OpenCV/3rdparty/lib -Wl,-Bstatic -lopencv_photo -lopencv_shape -lopencv_superres -lopencv_video -lopencv_calib3d -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc -lopencv_core -ltegra_hal -lncnn -ltrackerKCF -ldlib -Wl,-Bdynamic -ldl -lz -ljpeg -ltiff -lwebp -ljasper -lpng -lavformat-ffmpeg -lavcodec-ffmpeg -lavutil-ffmpeg -lswscale-ffmpeg -lglog -lfftw3f

echo "build end"
