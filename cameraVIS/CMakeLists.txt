get_directory_property(hasParent PARENT_DIRECTORY)
if(hasParent)
    cuda_add_library(l_cameraVI LiteGstCamera.cpp)
    target_link_libraries(l_cameraVI jetson-utils gobject-2.0 glib-2.0 gstreamer-1.0 gstapp-1.0)
else()
    cmake_minimum_required(VERSION 3.10)
    project(cameraVISTest)
    find_library(/home/cza/jetson-inference/build/aarch64/lib NAMES jetson-utils REQUIRED)
    find_package(CUDA REQUIRED)
    include_directories(/usr/include/gstreamer-1.0 /usr/include/glib-2.0 /usr/include/libxml2 /usr/lib/aarch64-linux-gnu/gstreamer-1.0/include /usr/lib/x86_64-linux-gnu/gstreamer-1.0/include/ /usr/lib/aarch64-linux-gnu/glib-2.0/include/ /usr/lib/x86_64-linux-gnu/glib-2.0/include/)

    cuda_add_executable(cameraVISTest cameraVISTest.cpp LiteGstCamera.cpp)

    target_link_libraries(cameraVISTest jetson-utils gobject-2.0 glib-2.0 gstreamer-1.0 gstapp-1.0)

endif()

