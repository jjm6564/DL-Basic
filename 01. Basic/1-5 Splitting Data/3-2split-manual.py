import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

X, y = np.arange(0,24).reshape((12,2)), range(12)

# print('X의 전체 데이터 :')
# print(X)
# print('y의 전체 데이터 :')
# print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
print('------------')
print('X 전체 데이터 :')
print(X)
print('y 데이터 :')
print(list(y))
print('------------')
num_of_train = int(len(X)*0.8)
num_of_test = int(len(X)-num_of_train)
print('train 데이터 :',num_of_train)
print('test 데이터 :',num_of_test)
print('------------')
X_test = X[num_of_train:]
y_test = y[num_of_train:]
X_train = X[:num_of_train]
y_train = y[:num_of_train]
print('X train 데이터 :',X_test)
print('y test :',list(y_test))
print('------------')
