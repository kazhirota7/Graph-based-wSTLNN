import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from torch.autograd import Variable
from weatherModel import GTL_NN
from DistanceCalc import *
from weatherOpModel import GTL_NN_Operators
import time
start_time = time.time()

torch.manual_seed(223626)
np.random.seed(223626)

def FindNeighbors(coordData):
    df = pd.read_excel(coordData, dtype={'Location':str, 'Latitude':float, 'Longitude':float})

    loc = df['Location'].values.tolist()
    lat = df['Latitude'].values.tolist()
    long = df['Longitude'].values.tolist()

    neighbors = []
    for i in range(len(loc)):
        group = [loc[i]]
        for j in range(len(loc)):
            if i == j:
                continue

            elif latLongDist(lat[i], long[i], lat[j], long[j]) <= 300:
                group.append(loc[j])
        neighbors.append(group)
    return neighbors

def TrainingFiles(neighbors):
    paths = {"Albury": "weather_data/albury_train.xlsx",
            "BadgerysCreek": "weather_data/badgeryscreek_train.xlsx",
            "Cobar": "weather_data/cobar_train.xlsx",
            "CoffsHarbour": "weather_data/coffsharbour_train.xlsx",
            "Moree": "weather_data/moree_train.xlsx",
            "Newcastle": "weather_data/newcastle_train.xlsx",
            "NorahHead": "weather_data/norahhead_train.xlsx",
            "NorfolkIsland": "weather_data/norfolkisland_train.xlsx",
            "Penrith": "weather_data/penrith_train.xlsx",
            "Richmond": "weather_data/richmond_train.xlsx",
            "Sydney": "weather_data/sydney_train.xlsx",
            "SydneyAirport": "weather_data/sydneyairport_train.xlsx",
            "WaggaWagga": "weather_data/waggawagga_train.xlsx",
            "Williamtown": "weather_data/williamtown_train.xlsx",
            "Wollongong": "weather_data/wollongong_train.xlsx",
            "Canberra": "weather_data/canberra_train.xlsx",
            "Tuggeranong": "weather_data/tuggeranong_train.xlsx",
            "MountGinini": "weather_data/mountginini_train.xlsx",
            "Ballarat": "weather_data/ballarat_train.xlsx",
            "Bendigo": "weather_data/bendigo_train.xlsx",
            "Sale": "weather_data/sale_train.xlsx",
            "MelbourneAirport": "weather_data/melbourneairport_train.xlsx",
            "Melbourne": "weather_data/melbourne_train.xlsx",
            "Mildura": "weather_data/mildura_train.xlsx",
            "Nhil": "weather_data/nhil_train.xlsx",
            "Portland": "weather_data/portland_train.xlsx",
            "Watsonia": "weather_data/watsonia_train.xlsx",
            "Dartmoor": "weather_data/dartmoor_train.xlsx",
            "Brisbane": "weather_data/brisbane_train.xlsx",
            "Cairns": "weather_data/cairns_train.xlsx",
            "GoldCoast": "weather_data/goldcoast_train.xlsx",
            "Townsville": "weather_data/townsville_train.xlsx",
            "Adelaide": "weather_data/adelaide_train.xlsx",
            "MountGambier": "weather_data/mountgambier_train.xlsx",
            "Nuriootpa": "weather_data/nuriootpa_train.xlsx",
            "Woomera": "weather_data/woomera_train.xlsx",
            "Albany": "weather_data/albany_train.xlsx",
            "Witchcliffe": "weather_data/witchcliffe_train.xlsx",
            "PearceRAAF": "weather_data/pearceraaf_train.xlsx",
            "PerthAirport": "weather_data/perthairport_train.xlsx",
            "Perth": "weather_data/perth_train.xlsx",
            "SalmonGums": "weather_data/salmongums_train.xlsx",
            "Walpole": "weather_data/walpole_train.xlsx",
            "Hobart": "weather_data/hobart_train.xlsx",
            "Launceston": "weather_data/launceston_train.xlsx",
            "AliceSprings": "weather_data/alicesprings_train.xlsx",
            "Darwin": "weather_data/darwin_train.xlsx",
            "Katherine": "weather_data/katherine_train.xlsx",
            "Uluru": "weather_data/uluru_train.xlsx"
    }
    trainingPaths = []
    for group in neighbors:
        groupPaths = []
        for region in group:
            groupPaths.append(paths[region])
        trainingPaths.append(groupPaths)
    return trainingPaths

def TestFiles(neighbors):
    paths = {"Albury": "weather_data/albury_test.xlsx",
            "BadgerysCreek": "weather_data/badgeryscreek_test.xlsx",
            "Cobar": "weather_data/cobar_test.xlsx",
            "CoffsHarbour": "weather_data/coffsharbour_test.xlsx",
            "Moree": "weather_data/moree_test.xlsx",
            "Newcastle": "weather_data/newcastle_test.xlsx",
            "NorahHead": "weather_data/norahhead_test.xlsx",
            "NorfolkIsland": "weather_data/norfolkisland_test.xlsx",
            "Penrith": "weather_data/penrith_test.xlsx",
            "Richmond": "weather_data/richmond_test.xlsx",
            "Sydney": "weather_data/sydney_test.xlsx",
            "SydneyAirport": "weather_data/sydneyairport_test.xlsx",
            "WaggaWagga": "weather_data/waggawagga_test.xlsx",
            "Williamtown": "weather_data/williamtown_test.xlsx",
            "Wollongong": "weather_data/wollongong_test.xlsx",
            "Canberra": "weather_data/canberra_test.xlsx",
            "Tuggeranong": "weather_data/tuggeranong_test.xlsx",
            "MountGinini": "weather_data/mountginini_test.xlsx",
            "Ballarat": "weather_data/ballarat_test.xlsx",
            "Bendigo": "weather_data/bendigo_test.xlsx",
            "Sale": "weather_data/sale_test.xlsx",
            "MelbourneAirport": "weather_data/melbourneairport_test.xlsx",
            "Melbourne": "weather_data/melbourne_test.xlsx",
            "Mildura": "weather_data/mildura_test.xlsx",
            "Nhil": "weather_data/nhil_test.xlsx",
            "Portland": "weather_data/portland_test.xlsx",
            "Watsonia": "weather_data/watsonia_test.xlsx",
            "Dartmoor": "weather_data/dartmoor_test.xlsx",
            "Brisbane": "weather_data/brisbane_test.xlsx",
            "Cairns": "weather_data/cairns_test.xlsx",
            "GoldCoast": "weather_data/goldcoast_test.xlsx",
            "Townsville": "weather_data/townsville_test.xlsx",
            "Adelaide": "weather_data/adelaide_test.xlsx",
            "MountGambier": "weather_data/mountgambier_test.xlsx",
            "Nuriootpa": "weather_data/nuriootpa_test.xlsx",
            "Woomera": "weather_data/woomera_test.xlsx",
            "Albany": "weather_data/albany_test.xlsx",
            "Witchcliffe": "weather_data/witchcliffe_test.xlsx",
            "PearceRAAF": "weather_data/pearceraaf_test.xlsx",
            "PerthAirport": "weather_data/perthairport_test.xlsx",
            "Perth": "weather_data/perth_test.xlsx",
            "SalmonGums": "weather_data/salmongums_test.xlsx",
            "Walpole": "weather_data/walpole_test.xlsx",
            "Hobart": "weather_data/hobart_test.xlsx",
            "Launceston": "weather_data/launceston_test.xlsx",
            "AliceSprings": "weather_data/alicesprings_test.xlsx",
            "Darwin": "weather_data/darwin_test.xlsx",
            "Katherine": "weather_data/katherine_test.xlsx",
            "Uluru": "weather_data/uluru_test.xlsx"
    }
    testPaths = []
    for group in neighbors:
        groupPaths = []
        for region in group:
            groupPaths.append(paths[region])
        testPaths.append(groupPaths)
    return testPaths




# test accuracy
def Test_accu(tl_nn, X_test, y_test):
    Xtest_pred = tl_nn(Variable(torch.Tensor(X_test)))
    Xpred_sign = torch.sign(Xtest_pred).detach().numpy()
    accu = np.sum(Xpred_sign == y_test) / len(y_test)
    return Xpred_sign, accu


# load data
def Read_data(train_path, test_path):
    df = pd.read_excel(train_path)
    data_test = pd.read_excel(test_path)
    return df, data_test


# dataframe to data and label
def Dataframe_to_Xy(df, data_len):
    labelcol = df['RainTomorrow'].to_list()
    c1 = df.iloc[:, 3:20].to_numpy()
    X = []
    y = []
    missing = []

    i = 14
    while i <= len(labelcol) - 1:
        if labelcol[i] == 1.0 or labelcol[i] == 0.0:
            cur_label = int(labelcol[i])
            # label_dur = labelcol[i:i + data_len]
            # dif_sign = len(set(label_dur))
            # if dif_sign == 1 and {cur_label} == set(label_dur):
            X.append(c1[i-data_len+1:i+1, :].astype(float))
            y.append(cur_label)
        else:
            missing.append(i)
        i += 1
    return X, y, missing

def Df_to_Xy_Neighbor(df, data_len, missing_train):
    labelcol = df['RainTomorrow'].to_list()
    c1 = df.iloc[:, 3:20].to_numpy()
    X = []
    y = []

    i = 14
    while i <= len(labelcol) - 1:
        if i not in missing_train:
            X.append(c1[i-data_len+1:i+1, :].astype(float))
        i += 1
    return X


# Training and test data, label
def GetXy(train_paths, test_paths, data_len):
    train_df, test_df = Read_data(train_paths[0], test_paths[0])
    # print(np.shape(train_df))

    X, y, missing_train = Dataframe_to_Xy(train_df, data_len)

    Xtest, ytest, missing_test = Dataframe_to_Xy(test_df, data_len)
    # print(np.shape(X))
    X = np.expand_dims(X, axis=2)
    Xtest = np.expand_dims(Xtest, axis=2)

    for i in range(1, len(train_paths)):
        train_df, test_df = Read_data(train_paths[i], test_paths[i])

        X_nei = Df_to_Xy_Neighbor(train_df, data_len, missing_train)

        X_cmb = np.expand_dims(X_nei, axis=2)

        Xtest_nei = Df_to_Xy_Neighbor(test_df, data_len, missing_test)
        Xtest_cmb = np.expand_dims(Xtest_nei, axis=2)

        X = np.append(X, X_cmb, axis=2)
        # print(np.shape(Xtest), np.shape(Xtest_cmb))
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


def Changelabel(ytest, y, X, n):
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
    for i in range(y10dif_len//2):
        idx = np.random.randint(y1_len)
        newx = X[y1_pos[idx]] + np.random.randn(15, n+1, 17)
        X = np.concatenate((X, newx.reshape((1, 15, n+1, 17))))
        y.append(1)
    y = np.array(y)
    ytest = np.array(ytest)
    return X, y, ytest


# Split data into training and test
def Splitdata(train_paths, test_paths, data_len):
    n = len(train_paths) - 1  # number of neighbors
    X, y, Xtest, ytest = GetXy(train_paths, test_paths, data_len)
    # X, y, Xtest, ytest = SlideXy(X, y, Xtest, ytest)
    X, y, ytest = Changelabel(ytest, y, X, n)
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
    if tp + fp == 0:
        positive_pred_value = -1
    else:
        positive_pred_value = tp / (tp + fp)
    if tn + fn == 0:
        negative_pred_value = -1
    else:
        negative_pred_value = tn / (tn + fn)
    auc = (sensitivity + specificity) / 2
    return sensitivity, specificity, positive_pred_value, negative_pred_value, \
           auc


def main():
    coordData = "LatLong.xlsx"
    neighbors = FindNeighbors(coordData)
    train_paths_tot = TrainingFiles(neighbors)
    test_paths_tot = TestFiles(neighbors)
    accuracy = []

    for i in range(len(train_paths_tot)):
        train_paths = train_paths_tot[i]
        test_paths = test_paths_tot[i]

        data_len = 15
        learning_rate = 0.001
        batch_size = 16
        Epoch = 40

        X_train, y_train, Xtest, ytest, T, M, n = Splitdata(train_paths, test_paths, data_len)
        # print(np.isnan(np.sum(X_train)))
        train_size = len(X_train)
        # Plotdata(X_train, y_train)
        gtl_nn_operators = GTL_NN_Operators(T, M, n)
        optimizer_operators = torch.optim.Adam(gtl_nn_operators.parameters(), lr=learning_rate)

        loss_iter = []

        for epoch in range(Epoch):
            for d_i in range(train_size // batch_size + 1):
                rand_idx = np.random.randint(0, train_size, batch_size)
                X_bt = Variable(torch.Tensor(X_train[rand_idx, :, :, :]))
                y_bt = Variable(torch.LongTensor(y_train[rand_idx,]))
                X_btpred = gtl_nn_operators(X_bt)
                k = gtl_nn_operators.operators()
                # print(k)
                # print(X_btpred)
                # print(y_bt)
                loss_tlnn = torch.sum(torch.exp(-y_bt * X_btpred))
                # print(loss_tlnn)
                optimizer_operators.zero_grad()
                loss_tlnn.backward(retain_graph=True)
                optimizer_operators.step()
                loss_iter.append(loss_tlnn.detach().numpy())

        learning_rate = 0.001
        Epoch = 70
        gtl_nn = GTL_NN(T, M, n, k)
        optimizer = torch.optim.Adam(gtl_nn.parameters(), lr=learning_rate)
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
                    # print(np.shape(pred_sign))
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
        accuracy.append(test_accu*100)
        print("--- %s seconds ---" % (time.time() - start_time))

    print('Total Accuracy is {}%'.format(np.sum(accuracy)/len(accuracy)))


if __name__ == "__main__":
    main()
