# -*- coding: utf-8 -*-
# NN for formula structure: G_{[k_1,k_2]}^w tx>=b
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TL_NN(nn.Module):
    def __init__(self, T, M, n, k):
        super(TL_NN,self).__init__()
        self.t = torch.nn.Parameter(1e-5 * torch.ones(M,1), requires_grad=True)  # coefficients for inputs (s) (number of inputs for each node)
        A1 = torch.rand((15,1),requires_grad=True)
        self.A1 = torch.nn.Parameter(A1)
        A2 = torch.rand((15,1),requires_grad=True)
        self.A2 = torch.nn.Parameter(A2)
        self.b = torch.nn.Parameter(torch.randn(1,1), requires_grad=True)  # bias for time
        self.n = n
        A_n = torch.rand((1, n+1), requires_grad=True) * 1e-5
        self.A_n = torch.nn.Parameter(A_n)
        if k[0] >= 0:
            self.k1 = 1
        else:
            self.k1 = -1

        if k[1] >= 0:
            self.k2 = 1
        else:
            self.k2 = -1

        if k[2] >= 0:
            self.k3 = 1
        else:
            self.k3 = -1

        if k[3] >= 0:
            self.k4 = 1
        else:
            self.k4 = -1
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
        self.r_a = r_a.reshape(-1, 30, self.n + 1)

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
        # self.grph_sftmax1 = torch.matmul(grph_sftmax1, self.t2) - self.b2
        # self.grph_sftmax2 = torch.matmul(grph_sftmax2, self.t2) - self.b2
        # # print(np.shape(self.r_a))
        # self.r_a = torch.matmul(self.r_a, self.t2) - self.b2
        # print(np.shape(self.r_a))
        # self.r_a = self.r_a.reshape(30, -1)
        # self.grph_sftmax1 = self.grph_sftmax1.reshape((30, -1))
        # self.grph_sftmax2 = self.grph_sftmax2.reshape((30, -1))
        # print(np.shape(self.grph_sftmax1))
        self.grph_sftmax1 = self.grph_sftmax1 * self.A_n_sm
        self.grph_sftmax2 = self.grph_sftmax2 * self.A_n_sm
        # print(np.shape(self.tmp_sftmax1))   # negation on softmax
        # self.grph_sftmax1 = self.grph_sftmax1.reshape(-1, 15, self.n + 1, 1)
        # self.grph_sftmax2 = self.grph_sftmax2.reshape(-1, 15, self.n + 1, 1)
        self.x_max = torch.max(self.grph_sftmax1, self.grph_sftmax2 * -1)

        # self.A_sm_nonzr, indices = torch.max(torch.cat((self.A_sm1, self.A_sm2), 1), 1)
        # self.A_sm_nonzr = self.A_sm_nonzr.reshape(15, 1)
        # print(np.shape(self.A_sm_nonzr))
        # print(np.shape(self.x_max))
        # print(np.shape(self.r_a))
        # print(np.shape(self.A_n_sm))
        self.wsx = self.x_max * self.r_a
        # self.wsx = torch.max(self.wsx1, self.wsx2 * -1)
        # print (np.shape(self.wsx1))

        # self.weisum = torch.sum(self.A_sm_nonzr * self.x_max, 0)
        # self.xrtn = torch.sum(self.wsx, 0) / self.weisum

        self.weisum = torch.sum(torch.sum(self.x_max, 2), 1)
        self.xrtn = torch.sum(torch.sum(self.wsx, 2), 1) / self.weisum
        # self.weisum2 = torch.sum(torch.sum(self.A_sm2 * self.tmp_sftmax2 * self.A_n_sm, 2), 1)
        # self.xrtn2 = torch.sum(torch.sum(self.wsx2, 2), 1) / self.weisum2
        # self.xrtn1 = self.xrtn1.reshape(-1, 1)
        # self.xrtn2 = self.xrtn2.reshape(-1, 1)
        # self.xrtn = torch.max(torch.cat((self.xrtn1, self.xrtn2 * -1), dim=1), 1)
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

        return self.xrtn