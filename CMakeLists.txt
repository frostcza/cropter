cmake_minimum_required(VERSION 2.8)
project( cropter )

# opencv, CUDA, openmp
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# jetson-utils
find_library(/home/cza/jetson-inference/build/aarch64/lib NAMES jetson-utils REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
file(MAKE_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
file(MAKE_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})

include_directories(/usr/include/gstreamer-1.0 /usr/lib/aarch64-linux-gnu/gstreamer-1.0/include /usr/include/glib-2.0 
					/usr/include/libxml2 /usr/lib/aarch64-linux-gnu/glib-2.0/include/ )

include_directories( ${PROJECT_BINARY_DIR} ${PROJECT_SOURCE_DIR})

link_directories(${PROJECT_SOURCE_DIR}/cameraIR/lib)

# Export compile
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(modules)
add_subdirectory(cameraIR)
add_subdirectory(cameraVIS)
add_subdirectory(RIFT)
add_subdirectory(detection)
add_subdirectory(fusion)
add_subdirectory(sht)
add_subdirectory(test)