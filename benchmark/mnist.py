from __future__ import print_function
from torchvision import datasets, transforms
import os
import pathlib
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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

    cache_dir = os.path.join(str(pathlib.Path.home()), ".cache/mnist")
    dataset = datasets.MNIST(cache_dir, train=True, download=True,
                             transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=64)

    model = Net().to(device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    start = time.time()
    epochs = 5
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
    throughput = len(dataset) * epochs * 1.0 / (time.time() - start)
    print("The throughput: {} samples/sec".format(throughput))


if __name__ == '__main__':
    main()
