# -*- coding: utf-8 -*-

# Code for classification on occupancy data with wSTL-NN
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from torch.autograd import Variable
import time
from COVIDModel import GTL_NN
from COVIDOpModel import GTL_NN_Operators
import time
start_time = time.time()


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
    labelcol = df['Output'].to_list()
    c1 = df.iloc[:, 1:5].to_numpy()
    X = []
    y = []

    i = 0
    while i <= len(labelcol) - data_len:
        cur_label = labelcol[i]
        # label_dur = labelcol[i:i + data_len]
        # dif_sign = len(set(label_dur))
        # if dif_sign == 1 and {cur_label} == set(label_dur):
        X.append(c1[i:i + data_len, :].astype(float))
        y.append(cur_label)
        i += 1
    return X, y


# Training and test data, label
def GetXy(train_paths, test_paths, data_len):
    train_df_pos, test_df_pos = Read_data(train_paths[0][0], test_paths[0][0])
    train_df_neg, test_df_neg = Read_data(train_paths[0][1], test_paths[0][1])

    X_pos, y_pos = Dataframe_to_Xy(train_df_pos, data_len)
    X_neg, y_neg = Dataframe_to_Xy(train_df_neg, data_len)

    X = np.append(X_pos,X_neg, axis=0)
    # print(np.shape(X))
    y = np.append(y_pos,y_neg, axis=0)

    Xtest_pos, ytest_pos = Dataframe_to_Xy(test_df_pos, data_len)
    Xtest_neg, ytest_neg = Dataframe_to_Xy(test_df_neg, data_len)
    Xtest = np.append(Xtest_pos, Xtest_neg, axis=0)
    ytest = np.append(ytest_pos, ytest_neg, axis=0)
    X = np.expand_dims(X, axis=2)
    Xtest = np.expand_dims(Xtest, axis=2)
    for i in range(1, len(train_paths)):
        train_df_pos, test_df_pos = Read_data(train_paths[i][0], test_paths[i][0])
        train_df_neg, test_df_neg = Read_data(train_paths[i][1], test_paths[i][1])
        X_pos, y_pos = Dataframe_to_Xy(train_df_pos, data_len)
        X_neg, y_neg = Dataframe_to_Xy(train_df_neg, data_len)
        X_cmb = np.expand_dims(np.append(X_pos, X_neg, axis=0), axis=2)

        Xtest_pos, ytest_pos = Dataframe_to_Xy(test_df_pos, data_len)
        Xtest_neg, ytest_neg = Dataframe_to_Xy(test_df_neg, data_len)
        Xtest_cmb = np.expand_dims(np.append(Xtest_pos, Xtest_neg, axis=0), axis=2)

        X = np.append(X, X_cmb, axis=2)
        Xtest = np.append(Xtest, Xtest_cmb, axis=2)
    # print(np.shape(X))

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
        newx = X[y1_pos[idx]] + np.random.randn(30, 5)
        X = np.concatenate((X, newx.reshape((1, 30, 5))))
        y.append(1)
    y = np.array(y)
    ytest = np.array(ytest)
    return X, y, ytest


# Split data into training and test
def Splitdata(train_paths, test_paths, data_len):
    n = len(train_paths) - 1  # number of neighbors
    X, y, Xtest, ytest = GetXy(train_paths, test_paths, data_len)
    # X, y, Xtest, ytest = SlideXy(X, y, Xtest, ytest)
    X, y, ytest = Changelabel(ytest, y, X)
    (N, T, _, M) = np.shape(X)
    num = list(range(N))
    random.shuffle(num)
    X_perm, y_perm = X[np.array(num), :, :], y[np.array(num)]
    X, y = X_perm, y_perm
    X_train, y_train = X, y
    X_total = np.concatenate((X_train, Xtest))
    y_total = np.concatenate((y_train, np.array(ytest)))
    N_new = len(X_total)

    (Ntest, _, _, _) = np.shape(Xtest)
    num2 = list(range(Ntest))
    random.shuffle(num2)
    Xtest_perm, ytest_perm = Xtest[np.array(num2), :, :], ytest[np.array(num2)]
    Xtest, ytest = Xtest_perm, ytest_perm
    X_total = np.concatenate((X_train, Xtest))
    y_total = np.concatenate((y_train, np.array(ytest)))

    N_train = N
    X_train, y_train = X_total[:N_train, :, :], y_total[:N_train]
    Xtest, ytest = X_total[N_train:, :, :], y_total[N_train:]

    return X_train, y_train, Xtest, ytest, T, M, n


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
    train_paths = [["generated_data/abruzzo(positive)train.txt", "generated_data/abruzzo(negative)train.txt"],
                   ["generated_data/lazio(positive)train.txt", "generated_data/lazio(negative)train.txt"],
                   ["generated_data/marche(positive)train.txt", "generated_data/marche(negative)train.txt"],
                   ["generated_data/molise(positive)train.txt", "generated_data/molise(negative)train.txt"]]
    test_paths = [["generated_data/abruzzo(positive)test.txt", "generated_data/abruzzo(negative)test.txt"],
                  ["generated_data/lazio(positive)test.txt", "generated_data/lazio(negative)test.txt"],
                  ["generated_data/marche(positive)test.txt", "generated_data/marche(negative)test.txt"],
                  ["generated_data/molise(positive)test.txt", "generated_data/molise(negative)test.txt"]]
    data_len = 30
    learning_rate = 0.001
    batch_size = 8
    Epoch = 50

    X_train, y_train, Xtest, ytest, T, M, n = Splitdata(train_paths, test_paths, data_len)
    train_size = len(X_train)
    # Plotdata(X_train, y_train)
    gtl_nn_operators = GTL_NN_Operators(T, M, n)
    optimizer_operators = torch.optim.RMSprop(gtl_nn_operators.parameters(), lr=learning_rate)

    loss_iter = []

    for epoch in range(Epoch):
        for d_i in range(train_size // batch_size + 1):
            rand_idx = np.random.randint(0, train_size, batch_size)
            X_bt = Variable(torch.Tensor(X_train[rand_idx, :, :, :]))
            y_bt = Variable(torch.LongTensor(y_train[rand_idx,]))
            X_btpred = gtl_nn_operators(X_bt)
            k = gtl_nn_operators.operators()
            print(k)
            # print(X_btpred)
            # print(y_bt)
            loss_tlnn = torch.sum(torch.exp(-y_bt * X_btpred))
            # print(loss_tlnn)
            optimizer_operators.zero_grad()
            loss_tlnn.backward()
            optimizer_operators.step()
            loss_iter.append(loss_tlnn.detach().numpy())

    learning_rate = 0.001
    Epoch = 80
    gtl_nn = GTL_NN(T, M, n, k)
    optimizer = torch.optim.RMSprop(gtl_nn.parameters(), lr=learning_rate)
    loss_iter = []
    accu_iter = []
    Perfor_iter = []


    for epoch in range(Epoch):
        for d_i in range(train_size // batch_size + 1):
            rand_idx = np.random.randint(0, train_size, batch_size)
            X_bt = Variable(torch.Tensor(X_train[rand_idx, :, :, :]))
            y_bt = Variable(torch.LongTensor(y_train[rand_idx,]))
            X_btpred = gtl_nn(X_bt)
            # print(X_btpred)
            # print(y_bt)
            loss_tlnn = torch.sum(torch.exp(-y_bt * X_btpred))
            # print(loss_tlnn)
            optimizer.zero_grad()
            loss_tlnn.backward()
            optimizer.step()
            loss_iter.append(loss_tlnn.detach().numpy())

            if d_i % 2 == 0:
                pred_sign, test_accu = Test_accu(gtl_nn, Xtest, ytest)
                # print(pred_sign)
                # print(ytest)
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
        gtl_nn.print_properties()
    sensitivity, specificity, positive_pred_value, negative_pred_value, auc \
        = Evaluate_measure(tp, fp, tn, fn)
    print('Accuracy is {}%'.format(test_accu*100))
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()


