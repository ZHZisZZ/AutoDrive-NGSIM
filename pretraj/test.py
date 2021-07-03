import torch

a = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
c = torch.tensor(0.0, requires_grad=True)
x = torch.tensor([1,2,3,4,5])
# y = torch.tensor([3,5,7,9,11])
y = torch.tensor([3,11,7,9,0])

# lr = 1e-3
# for _ in range(10000):
#   loss = torch.sum((a*x + b - y) ** 2)
#   print(loss)
#   loss.backward()
#   with torch.no_grad():
#       a -= a.grad * lr
#       b -= b.grad * lr
#       a.grad.zero_()
#       b.grad.zero_()
# print(a,b)

lr = 1e-3
opt = torch.optim.SGD([a,b], lr=lr)
for _ in range(10000):
  loss = torch.sum((a*x + b - y) ** 2)
  print(loss)
  loss.backward()
  opt.step()
  opt.zero_grad()
print(a,b)