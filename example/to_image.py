import numpy as np
import torchvision.utils as vutils
import torch
import glob, os

NUM = 10
SHAPE = (NUM, 28, 28)

files = glob.glob("*.txt")
for file in files:
    directory = os.path.splitext(file)[0]
    if not os.path.exists(directory):
        os.mkdir(directory)
    a = np.loadtxt(file)
    a = a.reshape(SHAPE)
    for i in range(NUM):
        t = torch.from_numpy(a[i])
        vutils.save_image(t, directory + '/' + str(i) + '.png', normalize=True) 
