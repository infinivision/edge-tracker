cmake_minimum_required(VERSION 2.8.10)

project(mtcnn_ncnn)

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

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenMP)
if(OpenMP_CXX_FOUND OR OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package( OpenCV REQUIRED )

if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "opencv Not Found!")
else()
    message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
    message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
endif (NOT OpenCV_FOUND)

message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/include/ncnn)
include_directories(${PROJECT_SOURCE_DIR}/3rdparty/kcf/trackerKCF)
include_directories(${PROJECT_SOURCE_DIR}/3rdparty/staple/staple)
add_subdirectory(${PROJECT_SOURCE_DIR}/3rdparty/kcf/trackerKCF)
add_subdirectory(${PROJECT_SOURCE_DIR}/3rdparty/staple/staple)
add_subdirectory(../dlib dlib_build)

link_directories(
  ${OpenCV_Install}/lib
  ${PROJECT_SOURCE_DIR}/lib/ncnn
)

add_executable(main src/main.cpp src/utils.cpp src/mtcnn.cpp src/face_attr.cpp src/face_align.cpp src/camera.cpp)
target_link_libraries(main ncnn trackerKCF trackerStaple ${OpenCV_LIBS} fftw3f dlib::dlib glog)
set_target_properties(main
        PROPERTIES 
#        ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib"
#        LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
)

add_executable(test-face-align src/test_face_align.cpp src/utils.cpp src/mtcnn.cpp src/face_align.cpp)
target_link_libraries(test-face-align ncnn ${OpenCV_LIBS} dlib::dlib glog)
set_target_properties(test-face-align
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
)

add_executable(test-face-quality src/test_face_quality.cpp src/utils.cpp src/face_attr.cpp)
target_link_libraries(test-face-quality ${OpenCV_LIBS} dlib::dlib)
set_target_properties(test-face-quality
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
)

add_executable(test_video src/test_video.cpp src/utils.cpp src/mtcnn.cpp src/face_attr.cpp src/face_align.cpp)
target_link_libraries(test_video ncnn trackerKCF trackerStaple ${OpenCV_LIBS} fftw3f dlib::dlib glog)
set_target_properties(test_video
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
)

#add_executable(compare src/tracker_compare.cpp src/utils.cpp src/mtcnn.cpp src/face_attr.cpp src/face_align.cpp)
#target_link_libraries(compare ncnn trackerKCF trackerStaple ${OpenCV_LIBS} fftw3f dlib::dlib)
#set_target_properties(compare
#        PROPERTIES
#        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
#)

add_executable(export src/export.cpp)
target_link_libraries(export fftw3f)
set_target_properties(export PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

