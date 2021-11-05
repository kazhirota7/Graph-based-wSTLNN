# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from CaseStudy2 import *
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from CaseStudy1 import *

train_paths = {"Albury": "weather_data/albury_train.xlsx",
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

test_paths = {"Albury": "weather_data/albury_test.xlsx",
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
X_train = []
X_test = []
y_train = []
y_test = []
for key in train_paths:
    if len(X_train) == 0:
        df = pd.read_excel(train_paths[key])
        df2 = pd.read_excel(test_paths[key])
        X_train = df.iloc[:,3:20].to_numpy()
        X_test = df2.iloc[:,3:20].to_numpy()
        y_train = df['RainTomorrow'].to_list()
        y_test = df2['RainTomorrow'].to_list()
    else:
        df = pd.read_excel(train_paths[key])
        df2 = pd.read_excel(test_paths[key])
        X_train = np.concatenate((X_train, df.iloc[:, 3:20].to_numpy()), axis=0)
        X_test = np.concatenate((X_test, df2.iloc[:, 3:20].to_numpy()), axis=0)
        y_train = np.concatenate((y_train, df['RainTomorrow'].to_list()), axis=0)
        y_test = np.concatenate((y_test, df2['RainTomorrow'].to_list()), axis=0)



# train = "weather_data/albury_train.xlsx"
# test = "weather_data/albury_test.xlsx"
# total = "AustWeather_zeros.csv"
#
#
# df1 = pd.read_excel(train)
# df2 = pd.read_excel(test)
# df3 = pd.read_csv(total)
# X_train = df3.iloc[:100000, 2:19].to_numpy()
# X_test = df3.iloc[100000:, 2:19].to_numpy()
# y_train = df3.iloc[:100000, 19].to_list()
# y_test = df3.iloc[100000:, 19].to_list()
# X_train = df1.iloc[:,3:20].to_numpy()
# X_test = df2.iloc[:,3:20].to_numpy()
# y_train = df1['RainTomorrow'].to_list()
# y_test = df2['RainTomorrow'].to_list()

def classify(classifier, x_train, y_train, x_test, y_test):
    true = 0
    false = 0
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    for i in range(len(y_pred)):
        if y_pred[i] == y_test[i]:
            true += 1
        else:
            false += 1
    return true/(true+false)

# coordData = "LatLong.xlsx"
# neighbors = FindNeighbors(coordData)
# train_paths_tot = TrainingFiles(neighbors)
# test_paths_tot = TestFiles(neighbors)
# data_len = 15
# X_train, y_train, Xtest, ytest, T, M, n = Splitdata(train_paths_tot[0], test_paths_tot[0], data_len)

print(np.shape(X_train))
# Classifiers

# KNN
knn = KNeighborsClassifier(n_neighbors=3)
print("KNN:", str(classify(knn, X_train,y_train,X_test,y_test) * 100) + "%")

# Decision Tree Algorithm

clf = DecisionTreeClassifier()
print("Decision Tree:", str(classify(clf, X_train,y_train,X_test,y_test) * 100) + "%")

# SVM
svm = make_pipeline(StandardScaler(), SVC(gamma='auto'))
print("SVM:", str(classify(svm, X_train,y_train,X_test,y_test) * 100) + "%")