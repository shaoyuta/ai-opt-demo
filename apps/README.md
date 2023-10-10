# Issues
## Centos8
### gcc 11
- sudo dnf install gcc-toolset-11-toolchain
- scl list-collections
- scl enable gcc-toolset-11 bash
- cmake -D CMAKE_C_COMPILER=/opt/rh/gcc-toolset-11/root/bin/gcc -D CMAKE_CXX_COMPILER=/opt/rh/gcc-toolset-11/root/bin/g++ ../apps/
- ```SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=sapphirerapids")```