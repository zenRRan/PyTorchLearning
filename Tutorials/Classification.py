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

net = Net(n_feather=2, n_hidden=10, n_output=2)

# print(net)
optimizer = optim.SGD(net.parameters(), lr=0.02)
loss_func = nn.CrossEntropyLoss()

plt.ion()
# plt.show()

for i in range(100):
    out = net(x)

    loss = loss_func(out, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    plt.cla()
    prediction = torch.max(F.softmax(out), 1)[1]
    pred_y = prediction.data.numpy().squeeze()
    plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, cmap="RdYlGn")
    accuracy = sum(pred_y == y.data.numpy())/200.
    plt.text(1.5, -4, 'Accuracy=%.4f' % accuracy)
    plt.pause(0.1)
    plt.show()








