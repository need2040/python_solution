from .base import BaseGARCHModel
import torch
import torch.nn as nn

class GJRGARCHModel(BaseGARCHModel):
    def __init__(self, init_omega=0.05, init_alpha=0.1, init_gamma = 0.15 , init_beta=0.7):
        super().__init__()
        self.model =nn.Sequential(
            nn.Linear(4,1, bias=False)
        )
        self._initialize_weights(init_omega, init_alpha, init_gamma, init_beta)

    def _initialize_weights(self, omega, alpha, gamma, beta):

        init_weights = torch.tensor([omega, alpha, gamma, beta], dtype=torch.float32)
        
        with torch.no_grad():
            self.model[0].weight.copy_(init_weights.unsqueeze(0))

    def forward (self, x):
        return self.model(x).squeeze(-1)