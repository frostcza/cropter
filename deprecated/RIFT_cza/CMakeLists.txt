get_directory_property(hasParent PARENT_DIRECTORY)
if(hasParent)
    cuda_add_library(l_rift rift_no_rotation_invariance.cpp phase.cpp bfprt.cpp)
    target_link_libraries(l_rift ${OpenCV_LIBS})
else()
    cmake_minimum_required(VERSION 3.10)
    project(RIFT_demo)
    find_package(OpenCV REQUIRED)
    find_package(CUDA REQUIRED)
    find_package(OpenMP REQUIRED)
    
    if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()

    include_directories(/usr/local/lib/)
    cuda_add_library(l_rift rift_no_rotation_invariance.cpp phase.cpp bfprt.cpp)
    
    cuda_add_executable(RIFT_demo RIFT_demo.cpp)
    target_link_libraries(RIFT_demo l_rift)
    target_link_libraries(RIFT_demo ${OpenCV_LIBS})
endif()



