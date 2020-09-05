#!/usr/bin/env python3
import torch
import torchvision.utils as vutils
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--load_pytorch", type=bool, default=False)
parser.add_argument("--load_gotorch", type=bool, default=False)
parser.add_argument("--save_image", type=bool, default=False)
parser.add_argument("--save_video", type=bool, default=False)


def get_ckpt_files(prefix):
    files = glob.glob(prefix + "*.pt")
    files = [(file, int(file.split('.')[0].split('-')[-1])) for file in files]
    files.sort(key=lambda f: f[1])
    files = [f[0] for f in files]
    return files


def load_ckpt_files(prefix, save_image):
    images_per_ckpt = 64
    files = get_ckpt_files(prefix)
    img_list = []
    for file in files:
        if prefix == "gotorch":
            module = torch.jit.load(file, map_location=torch.device("cpu"))
            images = list(module.parameters())[0].detach().cpu()
        else:
            images = torch.load(file, map_location=torch.device("cpu"))
        img_list.append(vutils.make_grid(images, padding=2, normalize=True))
        if save_image:
            # save fake images to directory
            directory = os.path.splitext(file)[0]
            if not os.path.exists(directory):
                os.mkdir(directory)
            images = [
                images[i].transpose(1, 2, 0) for i in range(images_per_ckpt)
            ]
            for i, image in enumerate(images):
                vutils.save_image(image,
                                  directory + '/' + str(i) + '.png',
                                  normalize=True)

    return img_list


if __name__ == "__main__":
    args = parser.parse_args()

    gotorch_img_list = load_ckpt_files(
        "gotorch", args.save_image) if args.load_gotorch else []
    pytorch_img_list = load_ckpt_files(
        "pytorch", args.save_image) if args.load_pytorch else []

    if args.save_video:
        img_list = []
        if len(gotorch_img_list) > 0 and len(pytorch_img_list) > 0:
            num = min(len(gotorch_img_list), len(pytorch_img_list))
            h = gotorch_img_list[0].shape[1]
            w = 128
            for i in range(num):
                img = torch.cat((gotorch_img_list[i], torch.ones(
                    3, h, w), pytorch_img_list[i]),
                                dim=2)
                img_list.append(img)

            fig = plt.figure(figsize=(16, 8))
        elif len(gotorch_img_list) > 0:
            img_list = gotorch_img_list
            fig = plt.figure(figsize=(8, 8))
        elif len(pytorch_img_list) > 0:
            img_list = pytorch_img_list
            fig = plt.figure(figsize=(8, 8))
        else:
            exit(0)
        plt.axis("off")
        ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)]
               for i in img_list]
        ani = animation.ArtistAnimation(fig,
                                        ims,
                                        interval=1000,
                                        repeat_delay=1000,
                                        blit=True)

        writer = animation.writers['ffmpeg']
        writer = writer(fps=4, metadata=dict(artist='Me'), bitrate=1800)
        ani.save("dcgan.mp4", writer)
        # save the last image
        plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
        plt.savefig("dcgan.png")
