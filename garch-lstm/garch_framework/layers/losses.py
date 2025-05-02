import torch
import torch.nn as nn

class CorrectedNLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred_var, target_eps_squared):
        loss = 0.5 * torch.log(pred_var) + (target_eps_squared / (2 * pred_var))
        return loss.mean()

class CorrectedTLoss(nn.Module):
    def __init__(self, nu):
        super().__init__()
        self.nu = nu
        
    def forward(self, pred_var, target_eps_squared):
        term = 1 + (target_eps_squared / ((self.nu - 2) * pred_var))
        loss = 0.5 * torch.log(pred_var) + 0.5 * (self.nu + 1) * torch.log(term)
        return loss.mean()