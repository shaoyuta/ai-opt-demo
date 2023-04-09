'''
cuda is ok
python3 chap03.py    --num_warmup 10 --num_iter 50
'''
import argparse
import time
import torch
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
#parser.add_argument('--gpu', default=None, type=int,
#                    help='GPU id to use.')
parser.add_argument('--num_iter', type=int, default=-1, help='num_iter')
parser.add_argument('--num_warmup', type=int, default=-1, help='num_warmup')

def main():
    args = parser.parse_args()
    main_worker(args)


def main_worker(args):
    # create model
    print("=> using pre-trained model '{}'".format("resnet50"))
    model = models.__dict__["resnet50"](pretrained=True)

    if not torch.cuda.is_available() :
        print('using CPU, this will be slow')
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
  
   
    # Data loading code
    print("=> Dummy data is used!")
    val_dataset = datasets.FakeData(50000, (3, 224, 224), 1000, transforms.ToTensor())

    val_sampler = None

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=64, shuffle=False,
        num_workers=4, pin_memory=True, sampler=val_sampler)

    validate(val_loader, model, args)

    return

def validate(val_loader, model, args):
    def run_validate(loader):
        with torch.no_grad():
            total_time = 0.0
            total_sample = 0
            for i, (images, target) in enumerate(loader):
                if args.num_iter > 0 and i >= args.num_iter: break

                # input to device
                elapsed = time.time()
                if torch.cuda.is_available():
                    target = target.cuda(non_blocking=True)

                # compute output
                model(images)
                if torch.cuda.is_available(): torch.cuda.synchronize()
                elapsed = time.time() - elapsed
                print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
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
