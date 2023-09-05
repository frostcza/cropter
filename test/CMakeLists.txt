# # version 1
# cuda_add_executable(test1 test1.cpp)
# target_link_libraries(test1 l_modules jetson-utils)

# version 2
cuda_add_library(l_integrate integrate.cpp)
cuda_add_executable(test2 test2_integrate.cpp)
target_link_libraries(test2 l_integrate l_modules jetson-utils)

# temper rectify tool
cuda_add_executable(test3 test3_temper_rectify.cpp)
target_link_libraries(test3 l_modules jetson-utils)