import torch
from torch._functorch.aot_autograd import aot_module_simplified

class M(torch.nn.Module):
    def forward(self, x):
        y = x.sin()
        return (y * y).sum(),

m = M()
x = torch.randn(4, requires_grad=True)

compiled = aot_module_simplified(m, args=(x,), fw_compiler=lambda g, _: g)
print(compiled(x))
