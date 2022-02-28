get_directory_property(hasParent PARENT_DIRECTORY)
if(hasParent)
    cuda_add_library(l_detection yolov5.cpp myCudaResize.cu yololayer.cu)
    target_link_libraries(l_detection nvinfer cudart)
else()
    cmake_minimum_required(VERSION 3.10)
    project(yolov5_demo)
    find_library(/home/cza/jetson-inference/build/aarch64/lib NAMES jetson-utils REQUIRED)
    find_package(CUDA REQUIRED)
    link_directories(/usr/local/cuda/lib64)

    cuda_add_library(myplugins SHARED yololayer.cu)
    target_link_libraries(myplugins nvinfer cudart)

    cuda_add_library(l_detection yolov5.cpp myCudaResize.cu)
    cuda_add_executable(yolov5_demo yolov5_demo.cpp)
    target_link_libraries(yolov5_demo l_detection jetson-utils myplugins)

    
endif()