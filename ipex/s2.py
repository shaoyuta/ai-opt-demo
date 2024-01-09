import torch
from torch import nn
#import intel_extension_for_pytorch as ipex
import argparse
from intel_extension_for_pytorch.cpu._auto_kernel_selection import (
    _enable_tpp,
    _disable_tpp,
    _using_tpp,
    _using_dnnl,
    _enable_dnnl,
    _disable_dnnl,
)

parser = argparse.ArgumentParser("S2", add_help=False)
parser.add_argument(
    "--dtype",
    type=str,
    choices=["float32", "bfloat16"],
    default="bfloat16",
    help="bfloat16, float32",
)
parser.add_argument("--N", default=100, type=int, help="num iter")
parser.add_argument("--nof-layers", default=1024, type=int, help="num of layers")
parser.add_argument("--sn", default=4096, type=int, help="num of layers")
parser.add_argument("--ipex", action="store_true")
parser.add_argument("--jit", action="store_true")
parser.add_argument("--profile", action="store_true")
parser.add_argument("--use-dnnl", action="store_true")
args = parser.parse_args()
print(args)

amp_enabled = True if args.dtype != "float32" else False

# import ipex
if args.ipex:
    import intel_extension_for_pytorch as ipex

    torch._C._jit_set_texpr_fuser_enabled(False)
    try:
        ipex._C.disable_jit_linear_repack()
    except Exception:
        pass

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_dummy_stack = nn.Sequential(
        )
        for _ in range(args.nof_layers):
            self.linear_dummy_stack.append(nn.Linear(args.sn, args.sn,bias=False))

    def forward(self, x):
        logits = self.linear_dummy_stack(x)
        return logits

m=Model().eval()

inputs = torch.randn(4,3, args.sn).to(dtype=torch.bfloat16)

if args.ipex:
    _disable_tpp()
    _disable_dnnl()
    if not args.use_dnnl:
        _enable_tpp()
    else:
        _enable_dnnl()
    m = ipex.optimize(m,dtype=torch.bfloat16,inplace=True)

if args.jit:
    m = torch.jit.trace(
        m,
        example_inputs=inputs,
        strict=False,
        check_trace=False,
    )
    m = torch.jit.freeze(m)

def trace_handler(prof):
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))

#print(m)


with torch.inference_mode(), torch.no_grad(), torch.cpu.amp.autocast(
        enabled=amp_enabled
    ):
    # do several runs:
    for _ in range(args.N):
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            on_trace_ready=trace_handler,
            ) as prof:
            m(inputs)