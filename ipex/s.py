import torch
import intel_extension_for_pytorch as ipex

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, a):
        b = torch.conv2d(a, torch.randn(1, 1, 1, 1)) # not fusible
        x = torch.mul(b, b)                          # fusible
        y = torch.sin(x)                             # fusible
        z = torch.mul(y, y)                          # fusible
        return z

m=Model().eval()
inputs = torch.randn(1, 1, 128, 128)
sm = torch.jit.script(m)

# do several runs:
for _ in range(2):
    sm(inputs)