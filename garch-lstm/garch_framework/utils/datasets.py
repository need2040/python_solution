import torch
from torch.utils.data import Dataset
import numpy as np

class FIGARCHDataset(Dataset):
    def __init__(self, eps, vol, truncation_size, scale=100):
        super().__init__()
        self.scale = scale

        self.eps = torch.tensor(eps, dtype=torch.float32)
        self.vol = torch.tensor(vol, dtype = torch.float32)

        self.eps_scaled = torch.square(self.eps) * self.scale
        self.vol_scaled = torch.square(self.vol) * self.scale

        self.truncation_size = truncation_size +1 

    def __len__(self):
        return len(self.vol) - self.truncation_size

    def __getitem__(self, idx):
        # Окно квадратов остатков и соответствующая волатильность
        t = idx + self.truncation_size -1

        return (
            self.eps_scaled[t - self.truncation_size+1: t],  
            self.vol_scaled[t]  
        )




class GARCHDataset(Dataset):
    def __init__(self, residuals, volatility, scale=100):
        self.scale = scale

        self.residuals = torch.tensor(residuals, dtype=torch.float32)
        self.volatility = torch.tensor(volatility, dtype=torch.float32)

        self.residuals_scaled = torch.square(self.residuals) * self.scale 
        self.volatility_scaled = torch.square(self.volatility) * self.scale
        

        self.inputs = torch.column_stack([
            torch.ones_like(self.residuals_scaled)*self.scale,   # 
            self.residuals_scaled,
            self.volatility_scaled
        ])
        


    def __len__(self):
        return len(self.inputs)-1 
    

    def __getitem__(self, index):
        return (
            self.inputs[index],
            self.residuals_scaled[index+1]
        )


class GJRGARCHDataset(Dataset):
    def __init__(self, residuals, volatility, scale=100):
        self.scale = scale

        self.residuals = torch.tensor(residuals, dtype=torch.float32)
        self.volatility = torch.tensor(volatility, dtype=torch.float32)

        self.residuals_scaled = torch.square(self.residuals) * self.scale 
        self.volatility_scaled = torch.square(self.volatility) * self.scale
        

        self.inputs = torch.column_stack([
            torch.ones_like(self.residuals_scaled)*self.scale,   # 
            self.residuals_scaled,
            torch.where(self.residuals < 0,1,0) * self.residuals_scaled,
            self.volatility_scaled
        ])
        


    def __len__(self):
        return len(self.inputs)-1 
    

    def __getitem__(self, index):
        return (
            self.inputs[index],
            self.residuals_scaled[index+1]
        )
