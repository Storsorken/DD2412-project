import numpy as np
import torch
import torch.nn as nn

class Dropout2dCustom(nn.Module):
    #ugly hack, MC-dropout only
    def __init__(self, p=0.5):
        super(Dropout2dCustom, self).__init__()
        self.p = p
        self.frozen_mask = None
        self.shape = None

    def forward(self, x):
        if not self.training or self.p == 0:
            return x*self.frozen_mask

        v = torch.rand((x.shape[0], x.shape[1]))
        self.shape = x.shape
        mask = v > self.p
        mask = mask.int()
        mask = mask.unsqueeze(-1).unsqueeze(-1).expand(*x.shape)

        # Scale the remaining values to maintain expectation
        return x * mask / (1 - self.p)
    
    def set_frozen_mask(self,):
        # Used to subsample a model using dropout 
        # so that it can be applied over several datpoints
        v = torch.rand((1, self.shape[1]))> self.p
        mask = v.repeat(self.shape[0], 1)
        mask = mask.int()
        self.frozen_mask = mask.unsqueeze(-1).unsqueeze(-1).expand(*self.shape) / (1 - self.p)

    


""" A = np.array([[1, 2, 3, 2],
            [4, 5, 6, 4],
            [7, 8, 9, 1]])

# Create a 3D array of shape (n, 4, 4) indicating whether pairs are equal
equality_array = (A[:, :, None] == A[:, None, :]).astype(int)

print(equality_array.sum(axis = 0)) """

drop = nn.Dropout2d(p=0.1)
drop = Dropout2dCustom()
x = torch.ones(3, 4, 2, 3)
out = drop(x)
drop.set_frozen_mask()
print(out)
drop.eval()
out = drop(x)
print(out)
