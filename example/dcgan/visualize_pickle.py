import torch
import torchvision.utils as vutils
import glob, os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--save_image", type=bool, default=False)
parser.add_argument("--save_video", type=bool, default=False)


def get_files():
    files = glob.glob("*.pt")
    files = [(file, int(file.split('.')[0].split('-')[-1])) for file in files]
    files.sort(key=lambda f: f[1])
    files = [f[0] for f in files]
    return files


if __name__ == "__main__":
    args = parser.parse_args()
    img_list = []

    files = get_files()
    for file in files:
        module = torch.jit.load(file, map_location=torch.device("cpu"))
        images = list(module.parameters())[0].detach().cpu().transpose(2, 3)
        img_list.append(vutils.make_grid(images, padding=2, normalize=True))

        if args.save_image:
            # save fake images to directory
            directory = os.path.splitext(file)[0]
            if not os.path.exists(directory):
                os.mkdir(directory)
            images = [images[i].reshape(3, 64, 64) for i in range(64)]
            for i, image in enumerate(images):
                vutils.save_image(image,
                                  directory + '/' + str(i) + '.png',
                                  normalize=True)

    if args.save_video:
        fig = plt.figure(figsize=(8, 8))
        plt.axis("off")
        ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)]
               for i in img_list]
        ani = animation.ArtistAnimation(fig,
                                        ims,
                                        interval=1000,
                                        repeat_delay=1000,
                                        blit=True)

        writer = animation.writers['ffmpeg']
        writer = writer(fps=1, metadata=dict(artist='Me'), bitrate=1800)
        ani.save("dcgan.mp4", writer)
