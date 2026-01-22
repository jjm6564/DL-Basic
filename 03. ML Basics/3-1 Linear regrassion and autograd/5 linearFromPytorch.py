import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[2],[4],[6]])

# print(x_train)
# print(x_train.shape)

# print(y_train)
# print(y_train.shape)

W = torch.zeros(1,requires_grad=True) #기울기
b = torch.zeros(1,requires_grad=True) #편향 
#print(W,b)

#hypothesis = x_train*W +b # 가설
#print(hypothesis)
#cost = torch.mean((hypothesis - y_train) **2) #예측값과 실제값의 평균제곱 오차 -> 비용
#print(cost)

optimizer = optim.SGD([W,b],lr=0.01) # Stochastic Gradient Descent
#optimizer.zero_grad() # gradient 0으로 초기화
#cost.backward() # cost 미분으로 gradient 계산
#optimizer.step() # W,b 업데이트

nb_epochs = 1999
for epoch in range(nb_epochs+1):
    hypothesis = x_train*W +b
    cost = torch.mean((hypothesis - y_train) **2)
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, W.item(), b.item(), cost.item()
        ))