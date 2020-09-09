import argparse
import time
import torch
import torch.optim
import torch.nn.functional as F

import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--train-dir', type=str,
                    help='url used to set up distributed training')


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


def train(epoch, train_loader, model, optimizer, batch_size, device):
    model.train()
    start = time.time()
    for i, (image, target) in enumerate(train_loader):
        image = image.to(device)
        target = target.to(device)

        output = model(image)
        loss = F.cross_entropy(output, target)

        acc1, acc5 = accuracy(output, target, (1, 5))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i and i % 10 == 0:
            throughput = (10 * batch_size * 1.0) / (time.time() - start)
            print('epoch: %d, batch: %d, loss: %f, acc1: %f, acc5: %f, throughput: %f' % (
                epoch, i, loss, acc1, acc5, throughput))
            start = time.time()


if __name__ == "__main__":
    args = parser.parse_args()
    batch_size = 32
    epochs = 100
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

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(
        args.train_dir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        adjust_learning_rate(optimizer, epoch, lr)
        train(epoch, train_loader, model, optimizer, batch_size, device)
