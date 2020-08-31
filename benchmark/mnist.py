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


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    device = torch.device("cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    cache_dir = os.path.join(str(pathlib.Path.home()), ".cache/mnist")
    dataset1 = datasets.MNIST(cache_dir, train=True, download=True,
                             transform=transform)
    dataset2 = datasets.MNIST(cache_dir, train=False,
                       transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=64)
    test_loader =torch.utils.data.DataLoader(dataset2, batch_size=64)

    model = Net().to(device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    epochs = 5
    for epoch in range(epochs):
        start = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
        throughput = len(dataset1) * 1.0 / (time.time() - start)
        print('Train Epoch: {}, Loss: {:.6f}, throughput: {} samples/sec'.format(
                epoch, loss.item(), throughput))
        test(model, device, test_loader)


if __name__ == '__main__':
    main()
