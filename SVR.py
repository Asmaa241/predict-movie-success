import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import preProcessing as preProc
import sklearn.svm as ml
import matplotlib.pyplot as plt


dataset = preProc.preProcessing()
dataset = np.array(dataset)
X = dataset[:,:-1]
Y = dataset[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.3, random_state=0,shuffle=True)
svr = ml.SVR(kernel='poly' , C=0.1 , degree=15).fit(X_train , y_train)
y_predict = svr.predict(X_test)
print("SVR - poly kernal with degree 15")
print("Data:", dataset.shape)
print("Train set size: ", len(X_train))
print("Test set size: ", len(X_test))
print("Mean absolute error: %.5f" % np.mean(np.absolute(y_predict - y_test)))
print("Residual sum of squares (MSE): %.5f" % np.mean((y_predict - y_test) ** 2))
print("R2-score: %.5f" % r2_score(y_test , y_predict) )