cmake_minimum_required(VERSION 3.10)
project(bfprtTest)
add_library(l_bfprt bfprt.cpp)
add_executable(bfprtTest bfprtTest.cpp)
target_link_libraries(bfprtTest l_bfprt)



