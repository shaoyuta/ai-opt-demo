# Set the minimum version of CMake that can be used
# To find the cmake version run
# $ cmake --version
cmake_minimum_required(VERSION 3.5)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=sapphirerapids")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Set the project name
project (demo1)

find_package(gflags REQUIRED)

SET(SOURCES 
    main.cpp tc.cpp
    test_amx.cpp
    test_simd_common.cpp
    test_simd_sse.cpp
    test_simd_avx.cpp
    test_avx512.cpp
    test_mem.cpp
    test_cpu.cpp
    test_omp.cpp
)

# Add an executable
add_executable(demo1 ${SOURCES})
#target_link_libraries(demo1 gflags_static -static)  #static link
target_link_libraries(demo1 gflags)  #dynamic link
