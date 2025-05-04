import torch
import torch.nn as nn

class BaseGARCHModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        raise NotImplementedError
        
    def get_params(self):
        return torch.concatenate([v.detach().cpu().flatten() for v in self.state_dict().values()]).numpy()