import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

x_train = torch.FloatTensor([[73,80,75],
                             [93,88,93],
                             [89,91,80],
                             [96,98,100],
                             [73,66,70]])

y_train = torch.FloatTensor([[152],[185],[180],[196],[142]])

print(x_train.shape)
print(y_train.shape)

W = torch.zeros((3,1),requires_grad=True)
b = torch.zeros(1,requires_grad=True)



optimizer = optim.SGD([W,b],lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs +1):
    hypothesis = x_train.matmul(W)+b

    cost = torch.mean((hypothesis - y_train)**2)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    print('Epoch {:4d}/{} hypothesis: {} Cost: {:.6f}'.format(
        epoch,nb_epochs,hypothesis.squeeze().detach(),cost.item()
    ))

    # 임의의 입력 값에 대한 예측
with torch.no_grad():
    new_input = torch.FloatTensor([[75, 85, 72]])  # 예측하고 싶은 임의의 입력
    prediction = new_input.matmul(W) + b
    print('Predicted value for input {}: {}'.format(new_input.squeeze().tolist(), prediction.item()))

