# Set the minimum version of CMake that can be used
# To find the cmake version run
# $ cmake --version
cmake_minimum_required(VERSION 3.5)

# Set the project name
project (perf-sample)

find_package(gflags REQUIRED)

include_directories(/home/taosy/repo/github/linux/tools/lib/perf/include)
link_directories(lib64)

add_executable(counting
  counting.c
)

add_executable(sampling
  sampling.c
)
link_directories(${CMAKE_SOURCE_DIR}/lib)

target_link_libraries(counting perf)
target_link_libraries(sampling perf)
