import torch
import torch._dynamo as dynamo

def f(x):
    if x.sum() > 0:
        return x * 2
    return x - 2

opt_f = dynamo.optimize("eager")(f)
opt_f(torch.randn(4))

exp = torch._dynamo.explain(f, torch.randn(4))
