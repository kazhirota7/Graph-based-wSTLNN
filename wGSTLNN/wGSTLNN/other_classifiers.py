# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from CaseStudy2 import *
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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

train_paths = [["generated_data/abruzzo(positive)train.txt", "generated_data/abruzzo(negative)train.txt"],
                   ["generated_data/lazio(positive)train.txt", "generated_data/lazio(negative)train.txt"],
                   ["generated_data/marche(positive)train.txt", "generated_data/marche(negative)train.txt"],
                   ["generated_data/molise(positive)train.txt", "generated_data/molise(negative)train.txt"]]
test_paths = [["generated_data/abruzzo(positive)test.txt", "generated_data/abruzzo(negative)test.txt"],
                  ["generated_data/lazio(positive)test.txt", "generated_data/lazio(negative)test.txt"],
                  ["generated_data/marche(positive)test.txt", "generated_data/marche(negative)test.txt"],
                  ["generated_data/molise(positive)test.txt", "generated_data/molise(negative)test.txt"]]

# Load training data

df1 = pd.read_csv("generated_data/abruzzo(positive)train.txt")
df2 = pd.read_csv("generated_data/abruzzo(negative)train.txt")
y_pos_train = df1['Output'].to_list()
X_pos_train = df1.iloc[:, 1:5].to_numpy()
y_neg_train = df2['Output'].to_list()
X_neg_train = df2.iloc[:, 1:5].to_numpy()

X_train = np.concatenate((X_pos_train, X_neg_train), axis=0)
y_train = np.concatenate((y_pos_train, y_neg_train), axis=0)

(N, inputs) = np.shape(X_train)
num = list(range(N))
random.shuffle(num)
X_perm, y_perm = X_train[np.array(num),:], y_train[np.array(num)]
X_train, y_train = X_perm, y_perm

# Load testing data

df3 = pd.read_csv("generated_data/abruzzo(positive)test.txt")
df4 = pd.read_csv("generated_data/abruzzo(negative)test.txt")
y_pos_test = df3['Output'].to_list()
X_pos_test = df3.iloc[:, 1:5].to_numpy()
y_neg_test = df4['Output'].to_list()
X_neg_test = df4.iloc[:, 1:5].to_numpy()

X_test = np.concatenate((X_pos_test, X_neg_test), axis=0)
y_test = np.concatenate((y_pos_test, y_neg_test), axis=0)

(N, inputs) = np.shape(X_test)
num = list(range(N))
random.shuffle(num)
X_perm, y_perm = X_test[np.array(num),:], y_test[np.array(num)]
X_test, y_test = X_perm, y_perm

print(np.shape(y_test))
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