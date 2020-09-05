import numpy as np
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--load_pytorch", type=bool, default=False)
parser.add_argument("--load_gotorch", type=bool, default=False)


def get_loss(prefix):
    step = []
    loss_d = []
    loss_g = []
    with open(prefix.lower() + '-dcgan.log') as f:
        lines = f.readlines()
        for line in lines:
            if 'Step' in line:
                fields = line.split('\t')
                step.append(int(fields[2].split(':')[-1]))
                loss_d.append(float(fields[3].split(':')[-1]))
                loss_g.append(float(fields[4].split(':')[-1]))
    return step, loss_d, loss_g


def plot_loss(prefix):
    step, loss_d, loss_g = get_loss(prefix)
    plt.figure(figsize=(8, 4))
    plt.plot(loss_g, label="G")
    plt.plot(loss_d, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.title("Generator and Discriminator Loss During Training of " + prefix)
    plt.legend()
    plt.savefig(prefix + "-dcgan-loss.png")


if __name__ == "__main__":
    args = parser.parse_args()

    if args.load_pytorch:
        plot_loss("PyTorch")

    if args.load_gotorch:
        plot_loss("GoTorch")
