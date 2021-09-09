# -*- coding: utf-8 -*-
# NN for formula structure: G_{[k_1,k_2]}^w tx>=b
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TL_NN(nn.Module):
    def __init__(self, T, M, n, k):
        super(TL_NN,self).__init__()
        self.t = torch.nn.Parameter(1e-5 * torch.ones(M,1), requires_grad=True)
        A1 = torch.rand((7,1),requires_grad=True)
        self.A1 = torch.nn.Parameter(A1)
        A2 = torch.rand((8,1),requires_grad=True)
        self.A2 = torch.nn.Parameter(A2)
        self.b = torch.nn.Parameter(torch.randn(1,1), requires_grad=True)  # bias for time
        self.n = n
        A_n = torch.rand((1, n+1), requires_grad=True)
        self.A_n = torch.nn.Parameter(A_n)
        self.k1 = 1 if k[0] >= 0 else -1
        self.k2 = 1 if k[1] >= 0 else -1
        self.k3 = 1 if k[2] >= 0 else -1
        self.k4 = 1 if k[3] >= 0 else -1
        self.T = T

    def forward(self,x):
        # zeros1 = torch.zeros((5, 1))
        zeros2 = torch.zeros((8, 1))
        zeros3 = torch.zeros((7, 1))
        # zeros4 = torch.zeros((10, 1))

        A1 = torch.cat((self.A1, zeros2), dim=0)
        A2 = torch.cat((zeros3, self.A2), dim=0)
        self.A = torch.max(torch.abs(A1), torch.abs(A2))

        r_a = torch.matmul(x, self.t) - self.b
        self.r_a = r_a.reshape(-1, 15, self.n + 1)

        self.A_abs1 = torch.abs(A1)
        self.A_sm1 = self.A_abs1 / torch.sum(self.A_abs1)

        self.A_abs2 = torch.abs(A2)
        self.A_sm2 = self.A_abs2 / torch.sum(self.A_abs2)

        self.A_n_abs = torch.abs(self.A_n)
        self.A_n_sm = self.A_n_abs / torch.sum(self.A_n_abs)

        self.tmp_sftmax1 = self.k3 * F.softmax(self.r_a, 1)
        self.tmp_sftmax2 = self.k4 * F.softmax(self.r_a, 1)

        self.grph_sftmax1 = self.k1 * F.softmax(self.tmp_sftmax1 * self.A_sm1, 2)
        self.grph_sftmax2 = self.k2 * F.softmax(self.tmp_sftmax2 * self.A_sm2, 2)

        self.grph_sftmax1 = self.grph_sftmax1 * self.A_n_sm
        self.grph_sftmax2 = self.grph_sftmax2 * self.A_n_sm

        self.x_max = torch.max(self.grph_sftmax1, self.grph_sftmax2 * -1)

        self.wsx = self.x_max * self.r_a

        self.weisum = torch.sum(torch.sum(self.x_max, 2), 1)
        self.xrtn = torch.sum(torch.sum(self.wsx, 2), 1) / self.weisum

        return self.xrtn

    def print_properties(self):
        print("\ninput weights:")
        print(self.t)
        print("\ntime weights:")
        print(self.A)
        print("\nbias1:")
        print(self.b)
        print("\nneighbor weights:")
        print(self.A_n)
        print("\nk values:")
        print([self.k1, self.k2, self.k3, self.k4])
