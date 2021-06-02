# -*- coding: utf-8 -*-
# NN for formula structure: G_{[k_1,k_2]}^w tx>=b
import torch
import torch.nn as nn
import torch.nn.functional as F


class TL_NN(nn.Module):
    def __init__(self, T, M):
        super(TL_NN,self).__init__()
        self.t = torch.nn.Parameter(1e-5 * torch.ones(M,1), requires_grad=True)
        A = torch.rand((T,1),requires_grad=True)
        self.A = torch.nn.Parameter(A)
        self.b = torch.nn.Parameter(torch.randn(1,1), requires_grad=True)
    def forward(self,x):
         r_a = torch.matmul(x, self.t) - self.b
         self.r_a =  r_a.reshape((-1,16)).t()
         self.A_abs = torch.abs(self.A)
         self.x_sftmin = F.softmin(self.r_a, 0)
         self.A_sm =  self.A_abs / torch.sum(self.A_abs)
         self.A_sm_nonzr = self.A_sm
         self.wsx = self.A_sm_nonzr * self.x_sftmin * self.r_a
         self.weisum = torch.sum(self.A_sm_nonzr * self.x_sftmin, 0)
         self.xrtn = torch.sum(self.wsx, 0) / self.weisum
         return self.xrtn
