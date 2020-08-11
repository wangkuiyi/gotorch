from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import time

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view((-1, 28 * 28,))
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def main():
    device = torch.device("cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    dataset = datasets.MNIST('/root/.cache/mnist', train=True, download=True,
                       transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=64)

    model = Net().to(device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    epochs = 10
    total_throughput = 0
    for epoch in range(1, epochs + 1):
        start = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 200 == 0:
                print("Train Epoch: {}, BatchIdx: {}, Loss: {:.6f}".format(
                    epoch, batch_idx, loss.item()))
        duration = time.time() - start
        throughput = len(dataset) * 1.0 / duration
        total_throughput += throughput
        print("End Epoch: {}, Throughput: {} samples/sec".format(
            epoch, throughput))
    print("The Average Throughput: {} samples.sec".format(
        total_throughput / epochs))


if __name__ == '__main__':
    main()
