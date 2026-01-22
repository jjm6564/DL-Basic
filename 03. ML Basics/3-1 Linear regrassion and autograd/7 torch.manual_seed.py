import torch

torch.manual_seed(3)
print("seed 3")
for i in range(1,3):
    print(torch.rand(1))

torch.manual_seed(5)
print("seed 5")
for i in range(1,3):
    print(torch.rand(1))

torch.manual_seed(3)
print("seed 3")
for i in range(1,3):
    print(torch.rand(1))
