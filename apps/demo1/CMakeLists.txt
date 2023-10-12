# Set the minimum version of CMake that can be used
# To find the cmake version run
# $ cmake --version
cmake_minimum_required(VERSION 3.5)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=sapphirerapids")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

# Set the project name
project (demo1)

find_package(gflags REQUIRED)

# Add an executable
add_executable(demo1 main.cpp tc.cpp test_amx.cpp test_avx512.cpp test_mem.cpp)
target_link_libraries(demo1 gflags)
