import torch
from torch import nn

a = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
c = torch.tensor(0.0, requires_grad=True)
x = torch.tensor([1,2,3,4,5])
y = torch.tensor([3,5,7,9,11])


# lr = 1e-3
# opt = torch.optim.SGD([a,b], lr=lr)
# for _ in range(10000):
#   loss = torch.sum((a*x + b - y) ** 2)
#   print(loss)
#   loss.backward()
#   opt.step()
#   opt.zero_grad()
# print(a,b)


# model = torch.nn.Sequential(
#   # ds, v_ego, v_pre -> a
#   torch.nn.Linear(5, 32),
#   torch.nn.ReLU(),
#   torch.nn.Linear(32, 1)
# )

model = torch.nn.Sequential(
  torch.nn.Linear(1, 32),
  torch.nn.ReLU(),
  torch.nn.Linear(32, 1)
)

x = x[:, None].type(torch.FloatTensor)
y = y[:, None].type(torch.FloatTensor)
# print(model(x)); exit()

# print(model.parameters()); exit()

lr = 1e-3
opt = torch.optim.SGD(model.parameters(), lr=lr)
for _ in range(100):
  pred = model(x)
  loss = torch.sum((y-pred) ** 2)
  print(loss)
  loss.backward()
  opt.step()
  opt.zero_grad()

print(model(torch.tensor([[2.0]])))