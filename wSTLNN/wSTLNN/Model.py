# -*- coding: utf-8 -*-
# NN for formula structure: G_{[k_1,k_2]}^w tx>=b
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TL_NN(nn.Module):
    def __init__(self, T, M, n):
        super(TL_NN,self).__init__()
        self.t = torch.nn.Parameter(1e-5 * torch.ones(M,1), requires_grad=True)  # coefficients for inputs (s) (5, 1) (number of inputs for each node)
        A = torch.rand((T,1),requires_grad=True)
        self.A = torch.nn.Parameter(A)  # weights (16, 1)
        self.b = torch.nn.Parameter(torch.randn(1,1), requires_grad=True)  # bias for time
        self.n = n
        self.t2 = torch.nn.Parameter(1e-5 * torch.ones(n+1,1), requires_grad=True)
        self.b2 = torch.nn.Parameter(torch.randn(1,1), requires_grad=True)


    def forward(self,x):
        # print(np.shape(x))
        r_a = torch.matmul(x, self.t) - self.b  # [numData, 16, n+1, 1]
        self.r_a = r_a.reshape(-1, 30, self.n+1).permute((1, 0, 2))
        grph_sftmax = F.softmax(self.r_a, 2)  # [16, numData, n+1] - output shape
        grph_sftmin = F.softmin(self.r_a, 2)  # [16, numData, n+1] - output shape
        self.grph_sftmax = torch.matmul(grph_sftmax, self.t2) - self.b2  # [16, numData, 1]
        self.grph_sftmin = torch.matmul(grph_sftmin, self.t2) - self.b2  # [16, numData, 1]
        # print(np.shape(self.r_a))
        self.r_a = torch.matmul(self.r_a, self.t2) - self.b2
        self.r_a = self.r_a.reshape(30, -1)
        self.grph_sftmax = self.grph_sftmax.reshape((30, -1))  # [16, numData] = output shape
        self.grph_sftmin = self.grph_sftmin.reshape((30, -1))  # [16, numData] = output shape
        # print(np.shape(self.grph_sftmax))
        self.tmp_sftmin = F.softmin(self.grph_sftmax, 0)  # [16, numData]
        self.tmp_sftmax = F.softmax(self.grph_sftmin, 0)  # [16, numData]
        row, column = np.shape(self.tmp_sftmax)
        self.x_cond = self.tmp_sftmax
        # print(np.shape(self.tmp_sftmax))
        self.x_sftmax = torch.max(self.tmp_sftmin, self.tmp_sftmax * -1)   # negation on softmax
        self.A_abs = torch.abs(self.A)
        self.A_sm = self.A_abs / torch.sum(self.A_abs)
        self.A_sm_nonzr = self.A_sm
        self.wsx = self.A_sm_nonzr * self.x_sftmax * self.r_a
        # print(np.shape(self.wsx))
        self.weisum = torch.sum(self.A_sm_nonzr * self.x_sftmax, 0)
        self.xrtn = torch.sum(self.wsx, 0) / self.weisum
        # print(np.shape(self.xrtn))
        print([self.t, self.A, self.b, self.t2, self.b2])
        return self.xrtn
