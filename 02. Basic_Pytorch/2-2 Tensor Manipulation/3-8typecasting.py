import torch
import numpy as np

lt = torch.LongTensor([1,2,3,4])
print(lt)
print(lt.float())

bt = torch.ByteTensor([True,False,False,True])
print(bt)

print(bt.long())
print(bt.float())

#concatenate

x = torch.FloatTensor([[1,2],[3,4]])
y = torch.FloatTensor([[5,6],[7,8]])

print(torch.cat([x,y],dim=0))
print('-'*20)
print(torch.cat([x,y],dim=1))
#Stacking
print('-'*20)
x = torch.FloatTensor([1,4])
y = torch.FloatTensor([2,5])
z = torch.FloatTensor([3,6])

print(torch.stack([x,y,z]))
print('-'*20)
# 11 ones_like zeros_like - 0,1로 채워진 텐서   
x = torch.FloatTensor([[0,1,2],[2,1,0]])
print(x)
print(torch.ones_like(x))
print(torch.zeros_like(x))
# 12 in-place Operation
x = torch.FloatTensor([[1,2],[3,4]])
print(x.mul(2.))
print(x)
print(x.mul_(3.))
print(x)