# -*- coding: utf-8 -*-
# NN for formula structure: G_{[k_1,k_2]}^w tx>=b
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class GTL_NN(nn.Module):
    def __init__(self, T, M, n, k):
        super(GTL_NN, self).__init__()
        self.t = torch.nn.Parameter(1e-5 * torch.ones(M,1), requires_grad=True)
        A1 = 0.5 * torch.ones((15,1),requires_grad=True)
        self.A1 = torch.nn.Parameter(A1)
        A2 = 0.5 * torch.ones((15,1),requires_grad=True)
        self.A2 = torch.nn.Parameter(A2)
        self.b = torch.nn.Parameter(1e-5*torch.randn(1,1), requires_grad=True)  # bias for time
        self.n = n
        A_n = F.softmax(torch.ones((1, self.n+1), requires_grad=True), 1)
        self.A_n = torch.nn.Parameter(A_n)
        self.k1 = 1 if k[0] >= 0 else -1
        self.k2 = 1 if k[1] >= 0 else -1
        self.k3 = 1 if k[2] >= 0 else -1
        self.k4 = 1 if k[3] >= 0 else -1
        A_disj = 0.5 * torch.ones((2,1,1), requires_grad=True)
        self.A_disj = torch.nn.Parameter(A_disj)
        self.T = T


    def forward(self,x):
        # print(np.shape(x))

        zeros1 = torch.zeros((5, 1))
        zeros2 = torch.zeros((15, 1))
        zeros3 = torch.zeros((15, 1))
        zeros4 = torch.zeros((10, 1))
        A1 = torch.cat((self.A1, zeros2), dim=0)
        A2 = torch.cat((zeros3, self.A2), dim=0)
        self.A = torch.max(torch.abs(A1), torch.abs(A2))

        A1 = torch.cat((self.A1, zeros2), dim=0)
        A2 = torch.cat((zeros3, self.A2), dim=0)
        self.A = torch.max(torch.abs(A1), torch.abs(A2))

        r_a = torch.matmul(x, self.t) - self.b
        self.r_a = r_a.reshape(-1, 1, 30, self.n + 1)

        self.A_abs1 = torch.abs(A1)
        self.A_sm1 = self.A_abs1 / torch.sum(self.A_abs1)

        self.A_abs2 = torch.abs(A2)
        self.A_sm2 = self.A_abs2 / torch.sum(self.A_abs2)

        self.A_n_abs = torch.abs(self.A_n)
        self.A_n_sm = self.A_n_abs / torch.sum(self.A_n_abs)

        self.A_disj_abs = torch.abs(self.A_disj)
        self.A_disj_sm = self.A_disj_abs / torch.sum(self.A_disj_abs)

        self.tmp_sftmax1 = self.k3 * F.softmax(self.r_a, 1)
        self.tmp_sftmax2 = self.k4 * F.softmax(self.r_a, 1)

        self.grph_sftmax1 = self.k1 * F.softmax(self.tmp_sftmax1 * self.A_sm1, 2)
        self.grph_sftmax2 = self.k2 * F.softmax(self.tmp_sftmax2 * self.A_sm2, 2)

        self.grph_sftmax1 = self.grph_sftmax1 * self.A_n_sm
        self.grph_sftmax2 = self.grph_sftmax2 * self.A_n_sm

        self.grph_sftmax1 = self.grph_sftmax1.reshape(-1, 1, 30, self.n + 1)
        self.grph_sftmax2 = self.grph_sftmax2.reshape(-1, 1, 30, self.n + 1)

        self.x_sftmax = F.softmax(torch.cat((self.grph_sftmax1, self.grph_sftmax2 * -1), 1), 1) * self.A_disj_sm

        self.wsx = self.x_sftmax * self.r_a

        self.weisum = torch.sum(torch.sum(torch.sum(self.x_sftmax, 3), 2), 1)
        self.xrtn = torch.sum(torch.sum(torch.sum(self.wsx, 3), 2),1) / self.weisum

        return self.xrtn

    def print_properties(self):
        print("\ninput predicates:")
        print(self.t)
        print("\ntime weights1:")
        print(self.A1 / torch.sum(self.A1, 0))
        print("\ntime weights2:")
        print(self.A2 / torch.sum(self.A2, 0))
        print("\nbias1:")
        print(self.b)
        print("\nneighbor weights:")
        print(self.A_n / torch.sum(self.A_n[:, 1:], 1))
        print("\nk values:")
        print([self.k1, self.k2, self.k3, self.k4])
        print("\ndisjunction weights:")
        print(self.A_disj / torch.sum(self.A_disj, 0))