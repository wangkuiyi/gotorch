import time
import torch
import torch.optim
import torch.nn.functional as F

import torchvision.models as models


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    new_lr = lr * (0.1**(epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train(model, optimizer, batch_size, device):
    model.train()

    for i in range(10):
        images = torch.randn((batch_size, 3, 224, 224)).to(device)
        target = torch.empty(batch_size,
                             dtype=torch.long).random_(1000).to(device)

        output = model(images)
        loss = F.cross_entropy(output, target)

        acc1, acc5 = accuracy(output, target, (1, 5))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 5 == 0:
            print("loss: %f, acc1: %f, acc5: %f" % (loss, acc1, acc5))


if __name__ == "__main__":
    batch_size = 16
    epochs = 10
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

    start = time.time()
    for epoch in range(epochs):
        adjust_learning_rate(optimizer, epoch, lr)
        train(model, optimizer, batch_size, device)
    print(time.time() - start)
