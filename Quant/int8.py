import torch
import torch.nn as nn
import numpy as np

class Int8Quantizer:
    
    def __init__(self, symmetric=True):
        self.symmetric = symmetric
    
    def quantize(self, tensor):

        if self.symmetric:
            return self._symmetric_quantize(tensor)
        else:
            return self._asymmetric_quantize(tensor)
    
    def _symmetric_quantize(self, tensor):
        max_val = torch.max(torch.abs(tensor))
        
        scale = max_val / 127.0
        
        quantized = torch.round(tensor / scale)
        quantized = torch.clamp(quantized, -127, 127).to(torch.int8)
        
        return quantized, scale, torch.tensor(0)
    
    def _asymmetric_quantize(self, tensor):
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)
        
        scale = (max_val - min_val) / 255.0
        zero_point = torch.round(-min_val / scale)
        zero_point = torch.clamp(zero_point, 0, 255)
        
        quantized = torch.round(tensor / scale + zero_point)
        quantized = torch.clamp(quantized, 0, 255).to(torch.uint8)
        
        return quantized, scale, zero_point
    
    def dequantize(self, quantized, scale, zero_point):

        return (quantized.float() - zero_point) * scale


class QuantizedLinear(nn.Module):
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.register_buffer('quantized_weight', torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer('weight_scale', torch.tensor(1.0))
        self.register_buffer('weight_zero_point', torch.tensor(0))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.quantizer = Int8Quantizer(symmetric=True)
    
    def quantize_weights(self, weight):
        q_weight, scale, zero_point = self.quantizer.quantize(weight)
        self.quantized_weight = q_weight
        self.weight_scale = scale
        self.weight_zero_point = zero_point
    
    def forward(self, x):
        q_x, x_scale, x_zero_point = self.quantizer.quantize(x)
        
        output = torch.matmul(q_x.float(), self.quantized_weight.float().t())
        
        output = output * (x_scale * self.weight_scale)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


def quantize_model(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            q_layer = QuantizedLinear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None
            )
            
            q_layer.quantize_weights(module.weight.data)
            
            if module.bias is not None:
                q_layer.bias.data = module.bias.data.clone()
            
            setattr(model, name, q_layer)
        else:
            quantize_model(module)
    
    return model