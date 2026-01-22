import torch

w = torch.tensor(2.0,requires_grad=True)

nb_epochs = 20
for epoch in range(nb_epochs+1):
    z = 2*w
    z.backward()
    print('수식 w 미분값 : {}'.format(w.grad))

'''
수식 w 미분값 : 8.0
수식 w 미분값 : 10.0
수식 w 미분값 : 12.0
수식 w 미분값 : 14.0
수식 w 미분값 : 16.0
수식 w 미분값 : 18.0
수식 w 미분값 : 20.0
수식 w 미분값 : 22.0
수식 w 미분값 : 24.0
수식 w 미분값 : 26.0
수식 w 미분값 : 28.0
수식 w 미분값 : 30.0
수식 w 미분값 : 32.0
수식 w 미분값 : 34.0
수식 w 미분값 : 36.0
수식 w 미분값 : 38.0
수식 w 미분값 : 40.0
수식 w 미분값 : 42.0
'''

# 미분값 2가 계속 누적
# optimizer.zero_grad()로 초기화의 필요성.