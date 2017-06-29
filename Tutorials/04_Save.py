import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1,1,100), dim=1)
y = x.pow(2) + 0.2 * torch.rand(x.size())

x, y = Variable(x), Variable(y)

def save_all():
    net = nn.Sequential(
        nn.Linear(1, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    )
    optimizer = optim.SGD(net.parameters(), lr=0.5)
    loss_func = nn.MSELoss()
    plt.ion()
    for i in range(25):
        predict = net(x)
        loss = loss_func(predict, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        plt.clf()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), predict.data.numpy(), 'k-', lw=5)
        plt.pause(0.1)
        plt.show()
    torch.save(net, "net.pkl")
    torch.save(net.state_dict(), "net_params.pkl")
save_all()

def restore_all():
    net1 = torch.load("net.pkl")
    prediction = net1(x)

    plt.clf()
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    plt.pause(2)
    plt.show()
restore_all()

def restore_params():
    net2 = nn.Sequential(
        nn.Linear(1, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    )
    net2.load_state_dict(torch.load("net_params.pkl"))
    prediction = net2(x)

    plt.clf()
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'g-', lw=5)
    plt.pause(2)
    plt.show()
restore_params()




