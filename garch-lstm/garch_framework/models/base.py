import torch
import torch.nn as nn

class BaseGARCHModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        raise NotImplementedError
        
    def get_params(self):
        return {k: v.detach().cpu().numpy() for k, v in self.state_dict().items()}