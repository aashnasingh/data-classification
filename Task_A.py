

import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.model_selection import KFold  # import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid

"""            Import Datasets              """
data1 = np.genfromtxt("ATNTFaceImages400.txt", delimiter=",")

knn_accuracy_list = []
centroid_accuracy_list = []
svm_accuracy_list = []


def KNN(i,X,y):
    kf = KFold(n_splits=i, random_state=None, shuffle=True)
    print("printing kf", kf)
    kf.get_n_splits(X)

    clf = KNeighborsClassifier(n_neighbors=5)

    accuracy_knn = 0
    for train_index, test_index in kf.split(X):
        # print('TRAIN:', train_index, 'TEST:', test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        # print("Test set predictions:\n{}".format(clf.predict(X_test)))
        # print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))
        accuracy_knn += clf.score(X_test, y_test)
        print("KNN Accuracy with ", i, " Fold: ",(clf.score(X_test, y_test)))
    print("Average accuracy of KNN with all folds: ",accuracy_knn/i)
    knn_accuracy_list.append(accuracy_knn/i)


def Centroid(i,X,y):
    kf = KFold(n_splits=i, random_state=None, shuffle=True)
    print("printing kf", kf)
    kf.get_n_splits(X)

    clf = NearestCentroid()

    accuracy_centroid = 0

    for train_index, test_index in kf.split(X):
        # print('TRAIN:', train_index, 'TEST:', test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        # print("Test set predictions:\n{}".format(clf.predict(X_test)))
        # print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))
        accuracy_centroid += clf.score(X_test, y_test)
        print("Centroid Accuracy with ", i, " Fold: ", (clf.score(X_test, y_test)))
    print("Average accuracy of centroid with all folds: ",accuracy_centroid/i)
    centroid_accuracy_list.append(accuracy_centroid/i)



def SVM(i,X,y):
    kf = KFold(n_splits=i, random_state=None, shuffle=True)
    print("printing kf", kf)
    kf.get_n_splits(X)

    clf = svm.SVC()

    accuracy_svm = 0
    accuracy_knn_all_folds = []
    for train_index, test_index in kf.split(X):
        # print('TRAIN:', train_index, 'TEST:', test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        # print("Test set predictions:\n{}".format(clf.predict(X_test)))
        # print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))
        accuracy_svm += clf.score(X_test, y_test)
        print("SVM Accuracy with ",i," Fold: ", (clf.score(X_test, y_test)))
    print("Average accuracy of SVM with all folds : ",accuracy_svm/i)
    svm_accuracy_list.append(accuracy_svm/i)




# Main Function
data_of_all_rows = data1                 ######  Transpose the file
array = np.array(data1)                  ######  Transpose the file
data = array.transpose()                 ######  Transpose the file

total_rows = len(data)
total_columns = len(data[0,0:])
train_label= data[0:,0]
train_data = data[0:,1:]
X=train_data
y=train_label

split_list = [2,3,5,10]

for i in split_list:
     KNN(i,X,y)
for i in split_list:
     Centroid(i,X,y)
for i in split_list:
     SVM(i,X,y)


print("Knn Accuracy List")
print(knn_accuracy_list)
print("Centroid Accuracy List")
print(centroid_accuracy_list)
print("Svm Accuracy List")
print(svm_accuracy_list)


plt.plot(knn_accuracy_list, 'r--',centroid_accuracy_list,'b--',svm_accuracy_list,'y--')
plt.show()



