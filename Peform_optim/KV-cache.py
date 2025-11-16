import torch
import torch.nn as nn
import math

class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention implementation.
    All attention heads share the same key and value projections,
    while each head maintains its own query projection.
    """
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size

        self.q_proj = nn.Linear(hidden_size, hidden_size)

        self.k_proj = nn.Linear(hidden_size, self.head_dim)
        self.v_proj = nn.Linear(hidden_size, self.head_dim)
        

        self.o_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.size()
        

        query = self.q_proj(hidden_states)
        
        # Key/Value shapes: [batch, seq_len, head_dim] (single head)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        

        # [batch, seq_len, hidden_size] -> [batch, num_heads, seq_len, head_dim]
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Reshape key/value for single head (with head dimension for broadcasting)
        # [batch, seq_len, head_dim] -> [batch, 1, seq_len, head_dim]
        key = key.view(batch_size, seq_len, 1, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, 1, self.head_dim).transpose(1, 2)
        
        # Key: [batch, 1, seq_len, head_dim] -> broadcasts to match query
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        

        if attention_mask is not None:
            scores += attention_mask * -1e9

        attn_probs = torch.softmax(scores, dim=-1)

        attn_output = torch.matmul(attn_probs, value)
        
        # [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, hidden_size]
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        
        return self.o_proj(attn_output)
    
    def demonstrate_backward_pass(self):
        """Show gradient accumulation in MQA"""
        torch.manual_seed(42)
        

        hidden_size, num_heads = 256, 8
        batch_size, seq_len = 2, 50
        
        mqa = MultiQueryAttention(hidden_size, num_heads)
        hidden = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
        target = torch.randn(batch_size, seq_len, hidden_size)
        
        output = mqa(hidden)
        loss = torch.nn.functional.mse_loss(output, target)
        
        loss.backward()
    
