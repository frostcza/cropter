cmake_minimum_required(VERSION 3.10)
project(gpioTest)
add_executable(gpioTest ModuleGPIOTest.cpp)
target_link_libraries(gpioTest pthread JetsonGPIO jetson-utils)
