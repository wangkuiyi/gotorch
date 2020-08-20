import torch
import torch.optim
import torch.nn.functional as F

import torchvision.models as models


def adjust_learning_rate(optimizer, epoch, lr):
    new_lr = lr * (0.1**(epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


if __name__ == "__main__":
    batch_size = 16
    epochs = 90
    lr = 0.1
    mementum = 0.9
    weight_decay = 1e-4

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = models.resnet50().to(device)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr,
                                momentum=mementum,
                                weight_decay=weight_decay)

    for epoch in range(epochs):
        adjust_learning_rate(optimizer, epoch, lr)

        model.train()

        image = torch.randn((batch_size, 3, 224, 224)).to(device)
        target = torch.empty(batch_size,
                             dtype=torch.long).random_(1000).to(device)

        output = model(image)
        loss = F.cross_entropy(output, target)

        print("loss: ", loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
