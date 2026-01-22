import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

X, y = np.arange(10).reshape((5,2)), range(5)

# print('X의 전체 데이터 :')
# print(X)
# print('y의 전체 데이터 :')
# print(y)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=1234)

print('X 트레인 데이터 :')
print(X_train)
print('X 테스트 데이터 :')
print(X_test)
print('------------')
print('y train data :')
print(y_train)
print('y test data :')
print(y_test)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
print('------------')
print('X 트레인 데이터 :')
print(X_train)
print('X 테스트 데이터 :')
print(X_test)
print('------------')
print('y train data :')
print(y_train)
print('y test data :')
print(y_test)