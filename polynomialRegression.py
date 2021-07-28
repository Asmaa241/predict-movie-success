import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import preProcessing as preProc


# dataset = pd.read_csv('USA_Students.csv')
# dataset = np.array(dataset)
# X = dataset[:,:2]
# Y = dataset[:,2]
# print(X)
dataset = preProc.preProcessing()
dataset = np.array(dataset)
X = dataset[:,:-1]
Y = dataset[:,-1]
X = np.c_[np.ones((X.shape[0], 1)), X]
X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.1, random_state=0,shuffle=True)

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_train)
poly.fit(X_poly, y_train)
model = linear_model.LinearRegression()
model.fit(X_poly, y_train)
y_predict = model.predict(poly.fit_transform(X_test))
print("Polynomial with degree 2")
print("Data:", dataset.shape)
print("Train set size: ", len(X_train))
print("Test set size: ", len(X_test))
print("Mean absolute error: %.5f" % np.mean(np.absolute(y_predict - y_test)))
print("Residual sum of squares (MSE): %.5f" % np.mean((y_predict - y_test) ** 2))
print("R2-score: %.5f" % r2_score(y_test , y_predict) )