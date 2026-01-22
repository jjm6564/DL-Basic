import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import random

#device = torch.cuda.is_available()
USE_CUDA = torch.backends.mps.is_available()
device = torch.device("mps" if USE_CUDA else 'cpu')
print('device : ',device)

random.seed(777)
torch.manual_seed(777)
if device == 'mps':
    torch.cuda.manual_seed_all(777)

training_epochs = 15
batch_size = 100

mnist_train = dsets.MNIST(root='MNIST_data/',#다운경로
                          train=True,# true 훈련데이터 리턴 , false는 테스트 데이터를 리턴
                          transform=transforms.ToTensor(), # 현재 데이터를 토치텐서로 변환
                          download=True #경로에 MNIST가 없으면 다운받음
                          )
mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True
                         )

data_loader = DataLoader(dataset=mnist_train,
                         batch_size=batch_size,
                         shuffle=True,
                         drop_last=True
                         )

linear = nn.Linear(784,10,bias=True).to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(linear.parameters(),lr=0.1)

for epoch in range(training_epochs): 
    avg_cost = 0
    total_batch = len(data_loader)

    for X, Y in data_loader:
        
        X = X.to(device).view(-1, 28 * 28)
        
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = linear(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning finished')

# 테스트 데이터를 사용하여 모델을 테스트한다.
with torch.no_grad(): # torch.no_grad()를 하면 gradient 계산을 수행하지 않는다.
    X_test = mnist_test.test_data.view(-1, 28 * 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = linear(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())

    # MNIST 테스트 데이터에서 무작위로 하나를 뽑아서 예측을 해본다
    r = random.randint(0, len(mnist_test) - 1)
    X_single_data = mnist_test.test_data[r:r + 1].view(-1, 28 * 28).float().to(device)
    Y_single_data = mnist_test.test_labels[r:r + 1].to(device)

    print('Label: ', Y_single_data.item())
    single_prediction = linear(X_single_data)
    print('Prediction: ', torch.argmax(single_prediction, 1).item())

    plt.imshow(mnist_test.test_data[r:r + 1].view(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()
