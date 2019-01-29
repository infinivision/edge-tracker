cmake_minimum_required(VERSION 3.1)

project(edge-tracker)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
message(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "build type not defined")
    set(CMAKE_BUILD_TYPE release CACHE STRING "Choose the type of build" FORCE)
endif()

if(CMAKE_BUILD_TYPE STREQUAL debug)
    message(STATUS "debug mode")
    ADD_DEFINITIONS(-D__DEBUG__)
else()
    message(STATUS "release mode")
ENDIF()

set(CMAKE_CXX_STANDARD 11)

find_package(OpenMP)
if(OpenMP_CXX_FOUND OR OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    if(APPLE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -openmp")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	endif()
else()
    message(STATUS "OpenMP Not Found")
endif()

find_package(PkgConfig)
pkg_check_modules( OpenCV REQUIRED opencv)
if (OpenCV_FOUND)
    message(STATUS "  OpenCV_LIBS: ${OpenCV_LIBRARIES}")
    message(STATUS "  OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "  OpenCV_LIBRARY_DIRS: ${OpenCV_LIBRARY_DIRS}")
    include_directories(${OpenCV_INCLUDE_DIRS})
    link_directories(${OpenCV_LIBRARY_DIRS})
else()
    message(FATAL_ERROR "  opencv Not Found!")
endif()

pkg_check_modules(EIGEN REQUIRED eigen3)
if(EIGEN_FOUND)
    message(STATUS "  eigen include: ${EIGEN_INCLUDE_DIRS}")
    include_directories(${EIGEN_INCLUDE_DIRS})
endif()

pkg_check_modules(FFTW REQUIRED fftw3)
if(FFTW_FOUND)
    message(STATUS "  fftw include: ${FFTW_INCLUDE_DIRS}")
    include_directories(${FFTW_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "  fftw not found! tracker depends on it")
endif()

pkg_check_modules(DLIB REQUIRED dlib)
if(DLIB_FOUND)
    message(STATUS "  dlib include: ${DLIB_INCLUDE_DIRS}")
    message(STATUS "  dlib libraries: ${DLIB_LIBRARIES}")
    message(STATUS "  dlib library dir: ${DLIB_LIBRARY_DIRS}")
    include_directories(${DLIB_INCLUDE_DIRS})
    link_directories(${DLIB_LIBRARY_DIRS})
endif()

pkg_check_modules(NCNN REQUIRED ncnn)
if(NCNN_FOUND)
    message(STATUS "  ncnn include: ${NCNN_INCLUDE_DIRS}")
    message(STATUS "  ncnn libraries: ${NCNN_LIBRARIES}")
    include_directories(${NCNN_INCLUDE_DIRS})
endif()

pkg_check_modules(GLOG REQUIRED glog)
if(GLOG_FOUND)
    message(STATUS "  glog include: ${GLOG_INCLUDE_DIRS}")
    include_directories(${GLOG_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "  glog not found! edge-tracker needs it for logging")
endif()

pkg_check_modules(GFLAGS REQUIRED gflags)
if(GFLAGS_FOUND)
    message(STATUS "  gflags include: ${GFLAGS_INCLUDE_DIRS}")
    include_directories(${GFLAGS_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "  gflags not found! glog depends on it")
endif()

pkg_check_modules(KCF REQUIRED tracker-kcf)
if(KCF_FOUND)
    message(STATUS "  kcf include: ${KCF_INCLUDE_DIRS}")
    include_directories(${KCF_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "  KCF tracker not found!")
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/3rdparty/kcf/trackerKCF)
include_directories(${PROJECT_SOURCE_DIR}/3rdparty/staple/staple)
include_directories(${PROJECT_SOURCE_DIR}/3rdparty/cpptoml/include)
#include_directories(../dlib)
add_subdirectory(${PROJECT_SOURCE_DIR}/3rdparty/kcf/trackerKCF)
#add_subdirectory(${PROJECT_SOURCE_DIR}/3rdparty/staple/staple)

link_directories(
  /usr/local/lib
#  ${PROJECT_SOURCE_DIR}/lib/ncnn
)

add_executable(main src/main.cpp src/utils/utils.cpp src/utils/time_utils.cpp src/mtcnn.cpp src/face_attr.cpp src/face_align.cpp src/camera.cpp src/image_quality.cpp)
target_link_libraries(main ncnn trackerKCF ${OpenCV_LIBRARIES} fftw3f ${DLIB_LIBRARIES} glog)
set_target_properties(main
        PROPERTIES 
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
)

add_executable(export src/export.cpp)
target_link_libraries(export fftw3f)
set_target_properties(export PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

if (EDGE_BUILD_TESTS)
    add_executable(test-face-align tests/test_face_align.cpp src/utils/time_utils.cpp src/utils/utils.cpp src/mtcnn.cpp src/face_align.cpp src/camera.cpp)
    target_link_libraries(test-face-align ncnn ${OpenCV_LIBRARIES} ${DLIB_LIBRARIES} glog)
    set_target_properties(test-face-align
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
    )

    add_executable(test-face-quality tests/test_face_quality.cpp src/utils/time_utils.cpp src/utils/utils.cpp src/face_attr.cpp src/image_quality.cpp src/face_align.cpp src/camera.cpp)
    target_link_libraries(test-face-quality ${OpenCV_LIBRARIES} ${DLIB_LIBRARIES} glog)
    set_target_properties(test-face-quality
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
    )

    add_executable(test_video tests/test_video.cpp src/utils/time_utils.cpp src/utils/utils.cpp src/mtcnn.cpp src/face_attr.cpp src/face_align.cpp src/camera.cpp)
    target_link_libraries(test_video ncnn trackerKCF trackerStaple ${OpenCV_LIBRARIES} fftw3f ${DLIB_LIBRARIES} glog)
    set_target_properties(test_video
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
    )

    add_executable(read-camera tests/read_camera.cpp src/utils/time_utils.cpp src/utils/utils.cpp src/face_align.cpp src/camera.cpp)
    target_link_libraries(read-camera ${OpenCV_LIBRARIES} ${DLIB_LIBRARIES} glog)
    set_target_properties(read-camera
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
    )

    add_executable(test-get-time tests/test_get_time.cpp src/utils/time_utils.cpp)
    target_link_libraries(test-get-time)
    set_target_properties(test-get-time
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
    )
else()
	message(STATUS "Not building tests")
endif(EDGE_BUILD_TESTS)

