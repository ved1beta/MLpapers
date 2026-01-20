import math
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.tensor.parallel import parallelize_module

dist.init_process_group(backend="nccl", init_method="env://")
torch.cuda.set_device(dist.get_rank())

class SimpleMLP(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))
    
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

model = SimpleMLP(hidden_size=1024).cuda()

tp_model = parallelize_module(
    model,
    device=torch.cuda.current_device(),
    parallelize_plan={
        "fc1": ColwiseParallel(),
        "fc2": RowwiseParallel(),
    }
)

class ved_TP(nn.Module):
    def __init__(self,
        d_model: int,           # Model dimension (e.g., 512)
        num_heads: int,         # Total attention heads (e.g., 8)
        rank: int,              # This process's rank
        world_size: int,        # Total number of GPUs
):
        super().__init__()
    
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.rank = rank
        self.world_size = world_size

        self.num_heads_local = num_heads // world_size
        self.dk = d_model // num_heads 
        self.d_local = self.num_heads_local * self.dk



        self.q_linear = nn.Linear(d_model, self.d_local)
        self.k_linear = nn.Linear(d_model, self.d_local)
        self.v_linear = nn.Linear(d_model, self.d_local)

        self.out_linear = nn.Linear(self.d_local, d_model)

    def forward(self, x, mask=None):
        batch_size , seq_len , _ = x.shape
    
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)

        atten_score = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            atten_score = atten_score.masked_fill(mask == 0, -1e9)

        atten_prob = torch.softmax(atten_score, dim=-1)
        atten_output = torch.matmul(atten_prob, V)

        atten_output = atten_output.transpose(1, 2).contiguous()
        atten_output = atten_output.view(batch_size, seq_len, self.d_local)

        output = self.out_linear(atten_output)

        return output