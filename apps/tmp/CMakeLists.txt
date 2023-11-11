# Set the minimum version of CMake that can be used
# To find the cmake version run
# $ cmake --version
cmake_minimum_required(VERSION 3.5)

# Set the project name
project (tmp)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -march=native")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
find_package(gflags REQUIRED)

# Add an executable
add_executable(tmp main.cpp)
target_link_libraries(tmp gflags)
