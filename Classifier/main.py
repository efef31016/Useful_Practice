# linear SVM
# https://www.pycodemates.com/2022/10/implementing-SVM-from-scratch-in-python.html

from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from SVM.dosvm import linearSVM

# Creating dataset
X, y = datasets.make_blobs(

        n_samples = 100, # Number of samples
        n_features = 3, # Features
        centers = 2,
        cluster_std = 1,
        random_state=40
    )

# Classes 1 and -1
y = np.where(y == 0, -1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

svm = linearSVM()
w, b, loss_save = svm.fit(X_train, y_train)

prediction = svm.predict(X_test)

print("Loss:", loss_save.pop())
print("Prediction:", prediction)
print("Accuracy:", accuracy_score(prediction, y_test))
print("w, b:", [w, b])

see = SVMvisualize(X, y, X_test, y_test, w, b, 0, 1)
see.visualize_dataset()
see.visualize_svm()


# Below two algorithm share same dataset(Theorem — Naive Bayes classifiers on binary features are subsumed by logistic regression classifiers.)
# Guassian Naive Bayes classifier
#https://kknews.cc/zh-tw/code/9gvamp8.html
#https://roger010620.medium.com/%E8%B2%9D%E6%B0%8F%E5%88%86%E9%A1%9E%E5%99%A8-naive-bayes-classifier-%E5%90%ABpython%E5%AF%A6%E4%BD%9C-66701688db02

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from GBC.doGBC import Gaussian_NB

X, y = load_iris(return_X_y=True)

# two classes
X = np.delete(X, np.where(y==2), axis=0)
y = np.delete(y, np.where(y==2))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=0)
data = np.concatenate([X_train,y_train.reshape(-1,1)],axis = 1)

nb = Gaussian_NB(data)
nb.fit()
print(sum(nb.predict(X_test)==y_test)/len(y_test))


# Logistic regression
#https://www.baeldung.com/cs/gradient-descent-logistic-regression  
#https://ithelp.ithome.com.tw/articles/10271080

from sklearn.preprocessing import StandardScaler
from LR.doLR import LogisticRegression
sc = StandardScaler()

train_feature = sc.fit_transform(X_train)
test_feature = sc.fit_transform(X_test)
train_target = np.array(y_train)
test_target = np.array(y_test)

w = np.zeros(train_feature.shape[1])
b = np.zeros(1)
model = LogisticRegression(train_feature, w, b, train_target)

learning_rate = 0.01
num_epochs = 100
loss = model.fit(num_epochs, learning_rate)

y_pred = model.forward(test_feature).reshape(-1)
y_pred_cls = y_pred.round()
acc = np.sum(y_pred_cls == test_target) / test_target.shape[0]
print(f'accuracy = {acc: .4f}')
