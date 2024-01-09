import torch
from torch import nn
import intel_extension_for_pytorch as ipex

USE_JIT=False
USE_IPEX=True
data_type=torch.float

linear_layer = nn.Linear(in_features=4096,out_features=17, bias=True).to(dtype=data_type).eval()


if USE_IPEX:
    linear_layer = ipex.optimize(linear_layer.eval(), dtype=data_type, inplace=True)

input=torch.zeros(1,4096).to(dtype=data_type)

if USE_JIT:
    trace_model = torch.jit.trace(
                        linear_layer,
                        example_inputs=input,
                        strict=False,
                        check_trace=False,
                    )
    trace_model = torch.jit.freeze(trace_model)
else:
    trace_model=linear_layer


def trace_handler(prof):
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU],
    on_trace_ready=trace_handler,
    ) as prof:
    for _ in range(6):
        output=trace_model(input)