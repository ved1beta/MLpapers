import torch 
import torch.nn as nn
import torch.distributed as dist

class ved_PP(nn.modules):
    def __init__(self, 
                 stage_id: int,
                 num_stages: int,
                 
                 layers: nn.ModuleList,
                 
                 num_microbatches:int,
                 
                 rank: int,
                 world_size: int):
        super().__init__()

        self.stage_id = stage_id 
        self.num_stages = num_stages
        self.layers = layers    
        self.num_microbatches = num_microbatches
        self.rank = rank
        self.world_size = world_size

        self.layers = layers

        self.prev_rank = rank - 1 if stage_id > 0 else None
        self.next_rank = rank + 1 if stage_id < num_stages - 1 else None

        self.is_first_stage = (stage_id == 0)
        self.is_last_stage = (stage_id == num_stages - 1)

        

        self.saved_inputs = {}
        self.saved_outputs = {}




