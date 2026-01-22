import numpy as np

vec = np.array([1, 2, 3,4,5])
print(vec)
print('--'*15)
mat = np.array([[10,20,30],[60,70,80]])
print(mat)

print('vec type : {}'.format(type(vec)))
print('mat type : {}'.format(type(mat)))

print('vec dim : {}'.format(vec.ndim))#축의 개수 춮력
print('vec shape : {}'.format(vec.shape))#각 축의 크기 출력

print('mat dim : {}'.format(mat.ndim))#축의 개수 춮력
print('mat shape : {}'.format(mat.shape))#각 축의 크기 출력