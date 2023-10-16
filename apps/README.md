# Issues
## Centos8
### compiler incompatibility
#### using gcc 11 from toolset
- sudo dnf install gcc-toolset-11-toolchain
- scl list-collections
- scl enable gcc-toolset-11 bash
- cmake -D CMAKE_C_COMPILER=/opt/rh/gcc-toolset-11/root/bin/gcc -D CMAKE_CXX_COMPILER=/opt/rh/gcc-toolset-11/root/bin/g++ ../apps/
- ```SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=sapphirerapids")```

#### using gcc 12 from conda
 - conda install gcc=12.3.0

### libgflags issue
### version of gflags is too low, no static file
- solution: 
    - conda install gflags (conda can provide >2.2 )
    - build and install gflags from source code 