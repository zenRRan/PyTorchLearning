import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + torch.rand(x.size())

torch_dataset = Data.TensorDataset(data_tensor=x, target_tensor=y)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2
)

class net(torch.nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.hidden = torch.nn.Linear(1, 20)   # hidden layer
        self.predict = torch.nn.Linear(20, 1)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x
# net = nn.Sequential(
#     nn.Linear(1, 20),
#     nn.ReLU(),
#     nn.Linear(20, 1)
# )

net_SGD =       net()
net_Momentum =  net()
net_RMSProp =   net()
net_Adam =      net()
nets = [net_SGD, net_Momentum, net_RMSProp, net_Adam]

opt_SGD = optim.SGD(net_SGD.parameters(), lr=0.01)
opt_Momentum = optim.SGD(net_Momentum.parameters(), lr=0.01, momentum=0.8)
opt_RMSProp = optim.RMSprop(net_RMSProp.parameters(), lr=0.01, alpha=0.9)
opt_Adam = optim.Adam(net_Adam.parameters(), lr=0.01, betas=(0.9, 0.99))
optmizers = [opt_SGD, opt_Momentum, opt_RMSProp, opt_Adam]

loss_func = nn.MSELoss()
losses_his = [[], [], [], []]

for epoch in range(12):
    print("Epoch:",epoch)
    for step, (batch_x, batch_y) in enumerate(loader):
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)

        for net, opt, l_his in zip(nets, optmizers, losses_his):
            output = net(b_x)
            loss = loss_func(output, b_y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            l_his.append(loss.data[0])

label = ["SGD", "Momentum", "RMSProp", "Adam"]
for i, l_his in enumerate(losses_his):
    plt.plot(l_his, label=label[i])
plt.legend(loc='best')
plt.xlabel("Steps")
plt.ylabel("Loss")
# plt.ylim(0,0.4)
plt.show()







