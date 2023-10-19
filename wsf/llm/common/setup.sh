conda create -n pt-39 python=3.9 -y
conda activate pt-39


# install required packages
conda install -y -c conda-forge  gcc=12.3 gxx=12.3 cxx-compiler make cmake gperftools 
conda install -y cmake ninja mkl mkl-include libxml2 numpy packaging

# install pytorch dev version
conda pip install https://download.pytorch.org/whl/nightly/cpu/torch-2.2.0.dev20230913%2Bcpu-cp39-cp39-linux_x86_64.whl


# install ipex: llvm
#cd ~/llm/XXXXX/
mkdir -p llvm-project && cd llvm-project
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.6/cmake-16.0.6.src.tar.xz
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.6/llvm-16.0.6.src.tar.xz
tar xvf cmake-16.0.6.src.tar.xz
tar xvf llvm-16.0.6.src.tar.xz
mv cmake-16.0.6.src cmake
mv llvm-16.0.6.src llvm
mkdir -p build && cd build
cmake ../llvm -DCMAKE_INSTALL_PREFIX=${PWD}/_install/llvm -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=X86 -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"
make install -j
ln -s ${PWD}/_install/llvm/bin/llvm-config ${CONDA_PREFIX}/bin/llvm-config-13

# install ipex: ipex
# cd ~/llm/XXXXX/cd 
git clone --branch llm_feature_branch https://github.com/intel-innersource/frameworks.ai.pytorch.ipex-cpu && \
cd frameworks.ai.pytorch.ipex-cpu && \
git reset 714b390 --hard && \
git submodule sync && git submodule update --init --recursive 
DNNL_GRAPH_BUILD_COMPILER_BACKEND=1 CXXFLAGS="${CXXFLAGS} -D__STDC_FORMAT_MACROS" python setup.py install

# install packages required by llama2
conda install datasets
python -m pip install transformers==4.31.0
python -m pip install cpuid accelerate datasets sentencepiece protobuf==3.20.3


