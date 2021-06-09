# -*- coding: utf-8 -*-

# Code for classification on occupancy data with wSTL-NN
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from torch.autograd import Variable
import time
from Model import TL_NN

torch.manual_seed(223626)
np.random.seed(223626)


# test accuracy
def Test_accu(tl_nn, X_test, y_test):
    Xtest_pred = tl_nn(Variable(torch.Tensor(X_test)))
    Xpred_sign = torch.sign(Xtest_pred).detach().numpy()
    accu = np.sum(Xpred_sign == y_test) / len(y_test)
    return Xpred_sign, accu


# load data
def Read_data(train_path, test_path):
    df = pd.read_csv(train_path)
    data_test = pd.read_csv(test_path)
    return df, data_test


# dataframe to data and label
def Dataframe_to_Xy(df, data_len):
    labelcol = df['Occupancy'].to_list()
    c1 = df.iloc[:, 1:6].to_numpy()
    X = []
    y = []

    i = 0
    while i < len(labelcol):
        cur_label = labelcol[i]
        label_dur = labelcol[i:i + data_len]
        dif_sign = len(set(label_dur))
        if dif_sign == 1 and {cur_label} == set(label_dur):
            X.append(c1[i:i + data_len, :].astype(float))
            y.append(cur_label)
        i = i + data_len
    return X, y


# Training and test data, label
def GetXy(train_path, test_path, data_len):
    train_df, test_df = Read_data(train_path, test_path)
    X, y = Dataframe_to_Xy(train_df, data_len)
    Xtest, ytest = Dataframe_to_Xy(test_df, data_len)
    return X, y, Xtest, ytest


def SlideXy(X, y, Xtest, ytest):
    X, y = X[:-1], y[:-1]
    Xtest, ytest = Xtest[:-1], ytest[:-1]
    return X, y, Xtest, ytest


def Xtoarray(X, Xtest):
    X_stack = np.stack(X, 0)
    Xtest = np.stack(Xtest, 0)
    return X_stack, Xtest


def Changelabel(ytest, y, X):
    ytest0_pos = [i for i in range(len(ytest)) if ytest[i] == 0]
    ytest = np.array(ytest)
    ytest[ytest0_pos] = -1
    ytest = ytest.tolist()
    y1_pos = [i for i in range(len(y)) if y[i] == 1]
    y0_pos = [i for i in range(len(y)) if y[i] == 0]
    y = np.array(y)
    y[y0_pos] = -1
    y = y.tolist()
    y1_len = len(y1_pos)
    y0_len = len(y0_pos)
    y10dif_len = y0_len - y1_len
    for i in range(y10dif_len):
        idx = np.random.randint(y1_len)
        newx = X[y1_pos[idx]] + np.random.randn(16, 5)
        X = np.concatenate((X, newx.reshape((1, 16, 5))))
        y.append(1)
    y = np.array(y)
    return X, y, ytest


# Split data into training and test
def Splitdata(train_paths, test_paths, data_len):
    X_train_tot = []
    y_train_tot = []
    Xtest_tot = []
    ytest_tot = []
    n = len(train_paths) - 1  # number of neighbors
    for i in range(len(train_paths)):  # append for each node
        X, y, Xtest, ytest = GetXy(train_paths[i], test_paths[i], data_len)
        X, y, Xtest, ytest = SlideXy(X, y, Xtest, ytest)
        X, y, ytest = Changelabel(ytest, y, X)
        (N, T, M) = np.shape(X)
        num = list(range(N))
        random.shuffle(num)
        X_perm, y_perm = X[np.array(num), :, :], y[np.array(num)]
        X, y = X_perm, y_perm
        X_train, y_train = X, y
        X_total = np.concatenate((X_train, Xtest))
        y_total = np.concatenate((y_train, np.array(ytest)))
        N_new = len(X_total)
        N_train = int(0.8 * N_new)
        X_train, y_train = X_total[:N_train, :, :], y_total[:N_train]
        Xtest, ytest = X_total[N_train:, :, :], y_total[N_train:]
        X_train_tot.append(X_train)
        y_train_tot.append(y_train)
        Xtest_tot.append(Xtest)
        ytest_tot.append(ytest)
    # print(X_train_tot)
    # print(np.shape(X_train_tot))
    X_train_tot = np.reshape(X_train_tot, (N_train, T, n+1, M))  # (batch, time, node, input)
    # print(np.shape(X_train_tot))
    # print(np.shape(ytest))
    y_train_tot = np.transpose(np.reshape(y_train_tot, (-1, )))
    Xtest_tot = np.reshape(Xtest_tot, (N_new-N_train, T, n+1, M))
    ytest_tot = np.transpose(np.reshape(ytest_tot, (-1, )))

    # y_train_tot = np.reshape(y_train_tot, ())
    # print(X_train_tot)
    # print(np.shape(Xtest))
    return X_train_tot, y_train_tot, Xtest_tot, ytest_tot, T, M, n


def Plotdata(X, y):
    plt.figure()
    for i in range(len(y)):
        if y[i] == -1:
            plt.plot(X[i, :], 'g')
        else:
            plt.plot(X[i, :], 'r')
    plt.show()


# Evaluate classification performance
def Evaluate_measure(tp, fp, tn, fn):
    sensitivity = tp / (tp + fn)
    specificity = tn / (fp + tn)
    positive_pred_value = tp / (tp + fp)
    negative_pred_value = tn / (tn + fn)
    auc = (sensitivity + specificity) / 2
    return sensitivity, specificity, positive_pred_value, negative_pred_value, \
           auc


def main():
    train_paths = ["occupancy_data/datatraining.txt"]
    test_paths = ["occupancy_data/datatest.txt"]
    data_len = 16
    X_train, y_train, Xtest, ytest, T, M, n = Splitdata(train_paths, test_paths, data_len)
    train_size = len(X_train)
    # Plotdata(X_train, y_train)
    tl_nn = TL_NN(T, M, n)
    learning_rate = 0.001
    optimizer = torch.optim.RMSprop(tl_nn.parameters(), lr=learning_rate)
    batch_size = 16
    Epoch = 10
    loss_iter = []
    accu_iter = []
    Perfor_iter = []
    for epoch in range(Epoch):
        for d_i in range(train_size // batch_size + 1):
            rand_idx = np.random.randint(0, train_size, batch_size)
            X_bt = Variable(torch.Tensor(X_train[rand_idx, :, :, :]))
            y_bt = Variable(torch.LongTensor(y_train[rand_idx,]))
            X_btpred = tl_nn(X_bt)
            loss_tlnn = torch.sum(torch.exp(-y_bt * X_btpred))
            optimizer.zero_grad()
            loss_tlnn.backward()
            optimizer.step()
            loss_iter.append(loss_tlnn.detach().numpy())

            if d_i % 2 == 0:
                pred_sign, test_accu = Test_accu(tl_nn, Xtest, ytest)
                print(pred_sign)
                print(ytest)
                accu_iter.append(test_accu)
                tp, fp, tn, fn = 0, 0, 0, 0
                for i in range(len(pred_sign)):
                    pdlb = pred_sign[i]
                    aclb = ytest[i]
                    if pdlb == 1 and aclb == 1:
                        tp += 1
                    elif pdlb == 1 and aclb == -1:
                        fp += 1
                    elif pdlb == -1 and aclb == 1:
                        fn += 1
                    elif pdlb == -1 and aclb == -1:
                        tn += 1
                Perfor_iter.append([tp, fp, tn, fn])

    #sensitivity, specificity, positive_pred_value, negative_pred_value, auc \
    #    = Evaluate_measure(tp, fp, tn, fn)
    print('Accuracy is {}%'.format(test_accu))


if __name__ == "__main__":
    main()


