cmake_minimum_required(VERSION 3.1)

project(edge-tracker)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

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

find_package( OpenCV REQUIRED )

if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "  opencv Not Found!")
else()
    message(STATUS "  OpenCV_LIBS: ${OpenCV_LIBS}")
    message(STATUS "  OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
endif (NOT OpenCV_FOUND)

find_package(PkgConfig)

pkg_check_modules(TRACKER_KCF REQUIRED tracker-kcf)
if(TRACKER_KCF_FOUND)
	message(STATUS "  tracker-kcf include: ${TRACKER_KCF_INCLUDE_DIRS}")
	message(STATUS "  tracker-kcf libraries: ${TRACKER_KCF_LIBRARIES}")
    include_directories(${TRACKER_KCF_INCLUDE_DIRS})
endif()

pkg_check_modules(DLIB REQUIRED dlib)
if(DLIB_FOUND)
    message(STATUS "  dlib include: ${DLIB_INCLUDE_DIRS}")
    message(STATUS "  dlib libraries: ${DLIB_LIBRARIES}")
    include_directories(${DLIB_INCLUDE_DIRS})
endif()

pkg_check_modules(NCNN REQUIRED ncnn)
if(NCNN_FOUND)
    message(STATUS "  ncnn include: ${NCNN_INCLUDE_DIRS}")
    message(STATUS "  ncnn libraries: ${NCNN_LIBRARIES}")
    include_directories(${NCNN_INCLUDE_DIRS})
endif()

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
#include_directories(${PROJECT_SOURCE_DIR}/include/ncnn)
#include_directories(${PROJECT_SOURCE_DIR}/3rdparty/kcf/trackerKCF)
include_directories(${PROJECT_SOURCE_DIR}/3rdparty/staple/staple)
include_directories(${PROJECT_SOURCE_DIR}/3rdparty/cpptoml/include)
#include_directories(../dlib)
#add_subdirectory(${PROJECT_SOURCE_DIR}/3rdparty/kcf/trackerKCF)
#add_subdirectory(${PROJECT_SOURCE_DIR}/3rdparty/staple/staple)

link_directories(
  /usr/local/lib
#  ${PROJECT_SOURCE_DIR}/lib/ncnn
)

add_executable(main src/main.cpp src/utils/utils.cpp src/utils/time_utils.cpp src/mtcnn.cpp src/face_attr.cpp src/face_align.cpp src/camera.cpp src/image_quality.cpp)
target_link_libraries(main ncnn trackerKCF ${OpenCV_LIBS} fftw3f ${DLIB_LIBRARIES} glog)
set_target_properties(main
        PROPERTIES 
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
)

add_executable(export src/export.cpp)
target_link_libraries(export fftw3f)
set_target_properties(export PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

if (EDGE_BUILD_TESTS)
    add_executable(test-face-align tests/test_face_align.cpp src/utils/time_utils.cpp src/utils/utils.cpp src/mtcnn.cpp src/face_align.cpp src/camera.cpp)
    target_link_libraries(test-face-align ncnn ${OpenCV_LIBS} ${DLIB_LIBRARIES} glog)
    set_target_properties(test-face-align
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
    )

    add_executable(test-face-quality tests/test_face_quality.cpp src/utils/time_utils.cpp src/utils/utils.cpp src/face_attr.cpp src/image_quality.cpp src/face_align.cpp src/camera.cpp)
    target_link_libraries(test-face-quality ${OpenCV_LIBS} ${DLIB_LIBRARIES} glog)
    set_target_properties(test-face-quality
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
    )

    add_executable(test_video tests/test_video.cpp src/utils/time_utils.cpp src/utils/utils.cpp src/mtcnn.cpp src/face_attr.cpp src/face_align.cpp src/camera.cpp)
    target_link_libraries(test_video ncnn trackerKCF trackerStaple ${OpenCV_LIBS} fftw3f ${DLIB_LIBRARIES} glog)
    set_target_properties(test_video
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
    )

    add_executable(read-camera tests/read_camera.cpp src/utils/time_utils.cpp src/utils/utils.cpp src/face_align.cpp src/camera.cpp)
    target_link_libraries(read-camera ${OpenCV_LIBS} ${DLIB_LIBRARIES} glog)
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

