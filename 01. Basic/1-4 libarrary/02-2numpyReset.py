import numpy as np

zero_mat = np.zeros((2,3)) #2행 3열의 0으로 채워진 행렬
print(zero_mat)

print('--'*5)
one_mat = np.ones((2,3)) #2행 3열의 1로 채워진 행렬
print(one_mat)

print('--'*5)
same_value_mat = np.full((2,2),7) #2행 2열의 7로 채워진 행렬
print(same_value_mat)

print('--'*5)
np_mat = np.eye(3) #3*3의 단위 행렬
print(np_mat)

print('--'*5)
rand_mat = np.random.random((2,2)) #22난수행렬
print(rand_mat)