'''
A Demo app: resnet50
Assume: dummy data, precision=float32, device=cpu, batch-size=64, evaluate
Sample cmd: python3 <file>.py --num_warmup 10 --num_iter 50
'''
# pylint: disable=consider-using-f-string,multiple-statements

import argparse
import time
import torch
from torchvision import datasets, models, transforms

parser = argparse.ArgumentParser(description='PyTorch ImageNet Demo')
parser.add_argument('--num_iter', type=int, default=-1, help='num_iter')
parser.add_argument('--num_warmup', type=int, default=-1, help='num_warmup')


def main():
    '''
    Function main
    '''
    args = parser.parse_args()
    main_worker(args)


def main_worker(args):
    '''
    Function main worker
    '''
    # create model
    print("=> using pre-trained model '{}'".format("resnet50"))
    model = models.__dict__["resnet50"](pretrained=True)

    val_dataset = datasets.FakeData(
        50000, (3, 224, 224), 1000, transforms.ToTensor())
    val_sampler = None
    val_loader = torch.utils.data.DataLoader(  # type: ignore
        val_dataset, batch_size=64, shuffle=False,
        num_workers=4, pin_memory=True, sampler=val_sampler)
    validate(val_loader, model, args)


def validate(val_loader, model, args):
    '''
    Function validate
    '''
    def run_validate(loader):
        with torch.no_grad():
            total_time = 0.0
            total_sample = 0
            for i, (images, _) in enumerate(loader):
                if args.num_iter > 0 and i >= args.num_iter:
                    break

                # input to device
                elapsed = time.time()

                # compute output
                model(images)

                elapsed = time.time() - elapsed
                print("Iteration: {}, inference time: {} sec.".format(
                    i, elapsed), flush=True)
                if i >= args.num_warmup:
                    total_time += elapsed
                    total_sample += 64

            throughput = total_sample / total_time
            latency = total_time / total_sample * 1000
            print('inference latency: %f ms' % latency)
            print('inference Throughput: %3f images/s' % throughput)

    # switch to evaluate mode
    model.eval()
    run_validate(val_loader)
    return 0


if __name__ == '__main__':
    main()
