import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

n_data = torch.ones(100, 2)
x0 = torch.normal(2*n_data, 1)
y0 = torch.zeros(100)
x1 = torch.normal(-2*n_data, 1)
y1 = torch.ones(100)

# print(x0)
# print(x1)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1), ).type(torch.LongTensor)
# print(x)
# print(y)

x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1])
# plt.show()
class Net(nn.Module):
    def __init__(self, n_feather, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = nn.Linear(n_feather, n_hidden)
        self.predict = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x
net1 = Net(2, 10, 2)
net2 = nn.Sequential(
    nn.Linear(2, 10),
    nn.ReLU(),
    nn.Linear(10, 2)
)

print(net1)
'''
Net (
  (hidden): Linear (2 -> 10)
  (predict): Linear (10 -> 2)
)
'''
print(net2)
'''
Sequential (
  (0): Linear (2 -> 10)
  (1): ReLU ()
  (2): Linear (10 -> 2)
)
'''


