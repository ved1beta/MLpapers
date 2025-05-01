import torch
from torch import Tensor, nn

import math
from typing import Tuple, Type

from .common import MLPBlock

class TwoWayTransformer(nn.Module):
    def __init__(self , 
                 depth: int , 
                 embedding_dim:int, 
                 num_heads:int , 
                 mlp_dim : int , 
                 activation :Type[nn.Module]= nn.ReLU, 
                 attention_downsample_rate : int = 2):
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append( 
                TwoWayTransformer(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )
        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)
