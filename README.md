# edge tracker
-----------------------------
the application can be run on X86 and ARM platform with Neon optimization.

# How to build
## clone the repo
```sh
git clone --recursive https://github.com/infinivision/edge-tracker.git
```

## build ncnn
* ncnn as submodule for the main repo to support mtcnn [the repo addr](https://github.com/Tencent/ncnn.git) 
* for the build and install please refer to the ncnn [wiki page](https://github.com/Tencent/ncnn/wiki/how-to-build)

## copy the ncnn lib and include headers to the main repo folder
```sh
# by default the ncnn was installed in ncnn/build/install folder
mkdir -p lib/ncnn include/ncnn
cp 3rdparty/ncnn/build/install/lib/libncnn.a lib/ncnn/
cp 3rdparty/ncnn/build/install/include/* include/ncnn/
```
## install eigen
```sh
wget http://bitbucket.org/eigen/eigen/get/3.3.4.tar.gz
tar xzvf 3.3.4.tar.gz
mv eigen/Eigen /usr/local/include/
```

## install fftw
```sh
apt install libfftw3-dev
```

## install cpptoml

```sh
git clone git@github.com:skystrife/cpptoml.git
mkdir build
cmake ..
make
make install
```

## build the edge-tracker

```sh
mkdir -p build
cd build
cmake ..
make
```

# run edge tracker
```sh
# activate OpemMP threads
set OMP_NUM_THREADS=2
bin/export 32 64 70 80 90 100
# use taskset to set cpu affinity 
taskset -c 4,5 main --model=<model_path> --config=<config_file> --output=<face_folder>
```

