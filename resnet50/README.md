- orig_repo: https://github.com/1pikachu/examples/blob/main/imagenet/README.md
- Dataset: https://ubit-artifactory-or.intel.com/artifactory/list/dpgpaivsoworkloads-or-local/dlboost/dataset/pytorch/resnet50_${INTERNAL_DATASET_VER}.tar

- regu00.py: rn50 + validate + cpu
- regu01.py: regu00.py + cuda
- regu02.py: regu01.py + hpu

- chap04.py: rn50 + {benchmark} + {cpu} + {dummy} + {num_warmup, num_iter}
- chap03.py: rn50 + {benchmark} + {cpu, cuda} + {dummy} + {num_warmup, num_iter}
- chap02.py: rn50 + {validate, benchmark} + {cpu, cuda} + {dummy} + {num_warmup, num_iter, precision, batch-size}
- chap01.py: rn50 + {validate, train, benchmark} + {cpu, cuda, mps } + {dummy, real} + {num_warmup, num_iter, precision, batch-size} + distributed + {channels_last, profile, ipex, jit, quantized_engine}
- chap05.py: rn50 + {validate, train, benchmark} + {cpu} + {dummy, real} + {num_warmup, num_iter, precision, batch-size} + {channels_last, profile, ipex, jit, quantized_engine}

- chap01.py:
  - train: ```python3 chap01.py --num_warmup 10 --num_iter 50 --data=/home/taosy/resnet50 ```
  - evaluate: ```python3 chap01.py --num_warmup 10 --num_iter 50 --data=/home/taosy/datasets/resnet50 --evaluate```
  - use pretrained model: ```--pretrained```