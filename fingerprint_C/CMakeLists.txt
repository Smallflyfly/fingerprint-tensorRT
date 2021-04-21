cmake_minimum_required(VERSION 3.19)
project(fingerprint_C)

set(CMAKE_CXX_STANDARD 14)


#set(OpenCV_DIR "D:\\opencv\\build\\install")
#set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)
#message(${OpenCV_LIBS})
#find_package(OpenCV REQUIRED)
#include_directories(${OpenCV_INCLUDE_DIRS})
#target_link_libraries(fingerprint_C_TensorRT ${OpenCV_LIBS})

find_package(OpenCV REQUIRED)
#message("${OpenCV}")
#include_directories(${OpenCV_INCLUDE_DIRS})
#target_link_libraries(fingerprint_C ${OpenCV_LIBS})

#include_directories(C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include)
#include_directories(D:\\TensorRT-7.0.0.11\\include)
#include_directories(D:\\TensorRT-7.0.0.11\\samples\\common\\windows)
#include_directories(D:\\TensorRT-7.0.0.11\\samples\\common)
#include_directories(D:\\TensorRT-7.0.0.11\\bin)
#include_directories(C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.2\\lib\\x64)

#set(TENSORRT_LIB_PATH "D:\\TensorRT-7.0.0.11\\lib")
#file(GLOB LIBS "${TENSORRT_LIB_PATH}/*.lib")
#set(CUDA_LIBRARIES "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.2\\lib\\x64\\*.lib")
#find_package(CUDA 10.2 REQUIRED)
#message("CUDA_LIBRARIES:${CUDA_LIBRARIES}")

set(SAMPLES_COMMON_SOURCES ./common/logger.cpp)
add_executable(fingerprint_C ${SAMPLES_COMMON_SOURCES} main.cpp)

target_link_libraries(fingerprint_C ${LIBS} ${CUDA_LIBRARIES})
target_link_libraries(fingerprint_C ${OpenCV_LIBS} ${TENSORRT_LIB})
include_directories(./common/)
include_directories(/home/TensorRT-7.2.2.3/samples/common)
include_directories(/home/TensorRT-7.2.2.3/include)
set(TENSORRT_LIB_PATH /home/TensorRT-7.2.2.3/lib)
file(GLOB LIBS ${TENSORRT_LIB_PATH}/*.so)
# add CUDA
find_package(CUDA 11.0 REQUIRED)
message(“CUDA_LIBRARIES:${CUDA_LIBRARIES}”)
message(“CUDA_INCLUDE_DIRS:${CUDA_INCLUDE_DIRS}”)
include_directories(${CUDA_INCLUDE_DIRS})


target_link_libraries(fingerprint_C ${LIBS} ${CUDA_LIBRARIES})



