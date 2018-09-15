cmake_minimum_required(VERSION 2.8.10)

project(local_edge_tracker)

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "build type not defined")
    set(CMAKE_BUILD_TYPE release CACHE STRING "Choose the type of build" FORCE)
endif()

if(DEV)
        add_definitions("-Wall -g -O0")
endif()

if(BENCH_EDGE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBENCH_EDGE")
endif()

if(VISUAL)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVISUAL")
endif()

if(SAVE_IMG)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSAVE_IMG")
endif()

if(CMAKE_BUILD_TYPE STREQUAL debug)
    message(STATUS "debug mode")
    ADD_DEFINITIONS(-D__DEBUG__)
else()
    message(STATUS "release mode")
ENDIF()

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if(NOT APPLE)
    find_package(OpenMP REQUIRED)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

message(STATUS "ncnn include path: ${NCNN_PATH}/include")
message(STATUS "ncnn link path: ${NCNN_PATH}/lib")
include_directories(${NCNN_PATH}/include)
link_directories(${NCNN_PATH}/lib)

message(STATUS "clickhouse include path: ${CLICKHOUSE_PATH}")
message(STATUS "clickhouse link path: ${CLICKHOUSE_PATH}/build/clickhouse")
include_directories(${CLICKHOUSE_PATH})
link_directories(${CLICKHOUSE_PATH}/build/clickhouse)

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
include_directories(${PROJECT_SOURCE_DIR}/3rdparty/mxnet/include)
include_directories(${PROJECT_SOURCE_DIR}/3rdparty/kcf/trackerKCF)
include_directories(${PROJECT_SOURCE_DIR}/3rdparty/staple/staple)

if (APPLE)
    include_directories(/opt/local/include)
    link_directories(/opt/local/lib)
endif()

add_subdirectory(${PROJECT_SOURCE_DIR}/3rdparty/kcf/trackerKCF)
add_subdirectory(${PROJECT_SOURCE_DIR}/3rdparty/staple/staple)

link_directories(
  ${OpenCV_Install}/lib
  ${PROJECT_SOURCE_DIR}/3rdparty/mxnet/lib
)

add_executable(main src/main.cpp src/face_tracker.cpp src/utils.cpp src/mtcnn.cpp src/face_attr.cpp 
                    src/camera.cpp src/face_pose_estimate.cpp src/face_predict.cpp 
                    src/vector_search.cpp src/face_embed.cpp src/face_age.cpp src/face_sample.cpp)
target_link_libraries(main ncnn trackerKCF trackerStaple mxnet ${OpenCV_LIBS} 
                           fftw3f  glog  curl idn2 psl ssl crypto z 
                           clickhouse-cpp-lib cityhash-lib lz4-lib)

set_target_properties(main
        PROPERTIES 
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
)

add_executable(test_infer_service src/test_infer_service.cpp)
target_link_libraries(test_infer_service curl idn2 psl ssl crypto z)
set_target_properties(test_infer_service 
        PROPERTIES 
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
)

add_executable(test_click_house src/test_click_house.cpp)
target_link_libraries(test_click_house clickhouse-cpp-lib cityhash-lib lz4-lib)
set_target_properties(test_click_house 
        PROPERTIES 
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
)

add_executable(export src/export.cpp)
target_link_libraries(export fftw3f)
set_target_properties(export 
        PROPERTIES 
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
)

add_executable(pnp src/test_solvePnP.cpp src/utils.cpp src/mtcnn.cpp src/camera.cpp)
target_link_libraries(pnp ncnn ${OpenCV_LIBS} glog)
set_target_properties(pnp
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
)

add_executable(face-predict src/face_predict_example.cpp src/face_predict.cpp)
target_link_libraries(face-predict mxnet ${OpenCV_LIBS} glog)
set_target_properties(face-predict 
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
)
