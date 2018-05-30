# MTCNN NCNN Implementation
-----------------------------
the application can be run on X86 and ARM platform with Neon optimization.

# How to build
## clone the repo
```sh
git clone --recursive https://github.com/infinivision/mtcnn_ncnn.git
```

## build ncnn
* ncnn as submodule for the main repo
* to support mtcnn. I update the ncnn code. [the repo addr](https://github.com/infinivision/ncnn.git) 
* for the build and install please refer to the ncnn [wiki page](https://github.com/Tencent/ncnn/wiki/how-to-build)

## copy the ncnn lib and include headers to the main repo folder
```sh
# by default the ncnn was installed in ncnn/build/install folder
mkdir -p lib/ncnn include/ncnn
cp 3rdparty/ncnn/build/install/lib/libncnn.a lib/ncnn/
cp 3rdparty/ncnn/build/install/include/* include/ncnn/
```

## build the mtcnn main repo
```sh
mkdir -p build
cd build
cmake ..
make
```

# how to run
```sh
bin/export 32 64 70 80 90 100
bin/main models/ncnn <camera_ip>
```

