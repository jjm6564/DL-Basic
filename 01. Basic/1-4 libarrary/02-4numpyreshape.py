import numpy as np

reshape_mat = np.array(np.arange(30)).reshape(5,6)
print(reshape_mat)
print('--'*5)

mat = np.array([[1,2,3],[4,5,6]])
print(mat)
print('--'*5)

slice_mat = mat[0,:] #index 1번 행 전체
print(slice_mat)
print('--'*5)

slice_mat2 = mat[1,:]#index 2 행
print(slice_mat2)
print('--'*5)

slice_mat3 = mat[:,0] #index 1번 열 전체
print(slice_mat3)
print('--'*5)

slice_mat4 = mat[:,1] #index 2번 열 전체
print(slice_mat4)
print('--'*5)

mat = np.array([[1,2],[4,5],[7,8]])
print(mat)
print(mat[1,0]) #2행 1열 ,특정 원소만

# mat[[2행, 1행],[0열, 1열]]
# 각 행과 열의 쌍을 매칭하면 2행 0열, 1행 1열의 두 개의 원소.
indexing_mat = mat[[2, 1],[0, 1]]
print(indexing_mat)

print('--'*20)

# Numpy 연산
x = np.array([1,2,3])
y = np.array([4,5,6])
plusMat = x + y
print(plusMat)
print('--'*5)

minusMat = x - y
print(minusMat)
print('--'*5)

# multiplication = mp.multiply(Mat, x)
multiMat1 = plusMat * x
multiMat2 = minusMat * x
print(multiMat1)
print(multiMat2)
print('--'*5)

divMat = multiMat2 / x
print(divMat)
print('--'*5)

#벡터와 행렬곱 -> .dot() 사용
mat1 = np.array([[1,2],[3,4]])
mat2 = np.array([[5,6],[7,8]])
dotMat = np.dot(mat1, mat2)
print(dotMat)