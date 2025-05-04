from .base import BaseGARCHModel
import torch
import torch.nn as nn

class FIGARCHModel(BaseGARCHModel):
    def __init__(self, truncation_size, lr = 1e-3):
        super().__init__()
        self.truncation_size = truncation_size
        self.lr = lr


        self.conv = nn.Conv1d(
            in_channels=1, 
            out_channels=1, 
            kernel_size=self.truncation_size,
            bias=False
        )
        
        # Инициализация весов как единицы
        with torch.no_grad():
            self.conv.weight.data = torch.ones_like(self.conv.weight.data)
    
    
    def forward(self, x):
        # x: [batch_size, truncation_size]
        x = x.unsqueeze(1)  # [batch_size, 1, truncation_size]
        # Conv1d с kernel_size=truncation_size даст 1 выходной элемент
        out = self.conv(x)  # [batch_size, 1, 1]
        return out.squeeze(-1).squeeze(-1)  # [batch_size]