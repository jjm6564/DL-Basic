import torch
import numpy as np
t = torch.FloatTensor([0.,1.,2.,3.,4.,5.,6.])
print(t)

print(t.dim())
print(t.shape)
print(t.size())

print(t[0],t[1],t[-1])
print(t[2:5],t[4:-1])
print(t[:2], t[3:])

print('-'*20)

t = torch.FloatTensor([[1.,2.,3.,],
                       [4.,5.,6.],
                       [7.,8.,9.],
                       [10.,11.,12.]
                       ])
print(t)
print(t.dim())
print(t.size())
print(t[:,1])
print(t[:,1].size())
print(t[:,:-1])

#broadcasting~

print('--'*10)
m1 = torch.FloatTensor([[3,3]])
m2 = torch.FloatTensor([[2,2]])
print(m1+m2)

print('--'*10)

m1 = torch.FloatTensor([1,2])
m2 = torch.FloatTensor([[3],[4]])
print(m1+m2)

print('--'*10)

# usually used function
m1 = torch.FloatTensor([[1,2],[3,4]])
m2 = torch.FloatTensor([[1],[2]])
print('Shape of Matrix 1 : ',m1.shape)
print('Shape of Matrix 2 : ',m2.shape)
print(m1.matmul(m2))

print(m1*m2)
print(m1.mul(m2)) #multiple , *

print('--'*10)

t = torch.FloatTensor([1,2])
print(t.mean()) # average
t = torch.FloatTensor([[1,2],[3,4]])
print(t)
print(t.mean()) # all average
print(t.mean(dim=0)) # average 행 제거? 얘는 1,3 2,4로 묶임 세로로
print(t.mean(dim=1)) # average 열 제거? 얘는 1,2 3,4로 묶임. 가로로
print(t.sum()) # sum all
print(t.sum(dim=0))
print(t.sum(dim=1))

print('--'*10)

# max and Argmax
# max 는 value argmax index return

print(t.max())
print(t.max(dim=0))

t = np.array([[[0,1,2],
               [3,4,5]],
               [[6,7,8],
                [9,10,11]]])
ft = torch.FloatTensor(t)

print(ft.view([-1,3]))
print(ft.view([-1,3]).shape) # 4,3

print(ft.view(-1,1,3))
print(ft.view([-1,1,3]).shape) # 4,1,3

#squeez 1인 차원 제거
ft = torch.FloatTensor([[0],[1],[2]])
print(ft)
print(ft.shape) #3,1
print(ft.squeeze())
print(ft.squeeze().shape) #3
#Unsqueeze 특정위치에 1인 차원추가
ft = torch.FloatTensor([0,1,2])
print(ft.shape)
print(ft.unsqueeze(0))
print(ft.unsqueeze(0).shape)
#type casting
