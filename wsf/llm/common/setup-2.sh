#!/bin/bash -e

set -ex

# Create workspace and create link
# mkdir -p ~/llm/1-c4709ac180e
# ln -s ../common ~/llm/common
# ln -s ../workspace ~/llm/1-c4709ac180e/workspace

#conda create -n pt-39 python=3.9 -y
#conda activate pt-39

#exit 1


# install required packages
conda install -y -c "conda-forge/label/cf202003" gperftools 
conda install -y cmake ninja mkl mkl-include

# install pytorch dev version
pip install torch==2.1 --index-url https://download.pytorch.org/whl/cpu


# install ipex: ipex
python -m pip install intel_extension_for_pytorch==2.1

# install packages required by llama2
conda install datasets
python -m pip install transformers==4.31.0
python -m pip install cpuid accelerate datasets sentencepiece protobuf==3.20.3

# install deepspeed: build torch-ccl
git clone -b ccl_torch_dev_0905 https://github.com/intel/torch-ccl.git && \
cd torch-ccl && \
git reset --hard 40d6141 && \
git submodule sync && git submodule update --init --recursive && \
python setup.py install

# install deepspeed: build DEEPSPEED
git clone https://github.com/delock/DeepSpeedSYCLSupport && \
cd DeepSpeedSYCLSupport && \
git checkout gma/run-opt-branch && \
git reset --hard f15e6d4 && \
python -m pip install -r requirements/requirements.txt && \
python setup.py install

# install deepspeed: build oneccl
git clone  https://github.com/oneapi-src/oneCCL.git && \
cd oneCCL/ && \
git reset --hard b4c31ba && \
mkdir -p build && cd build/ &&
cmake .. && make -j && make -j install 
