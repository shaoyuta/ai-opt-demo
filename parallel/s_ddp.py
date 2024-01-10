import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
from torch.nn.parallel import DistributedDataParallel as DDP


def run(rank, size):
    """ Distributed function to be implemented later. """
    torch.manual_seed(42)
    m = torch.nn.Linear(10, 1)
    ddp = DDP(m)
    inp = torch.rand(10)
    out = ddp(inp)
    out.sum().backward()
    print ('OUTPUT: {}'.format(out))
    print ('GRADIENT: {}'.format(m.weight))


def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 5
    processes = []
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()