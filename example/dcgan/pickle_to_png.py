import torch
import torchvision.utils as vutils
import glob, os

files = glob.glob("*.pt")
for file in files:
    directory = os.path.splitext(file)[0]
    if not os.path.exists(directory):
        os.mkdir(directory)

    module = torch.jit.load(file, map_location=torch.device("cpu"))
    images = list(module.parameters())[0]
    for i in range(64):
        image = images[i].detach().cpu().reshape(3, 64, 64)
        image = image.transpose(1, 2)
        vutils.save_image(image, directory + '/' + str(i) + '.png', normalize=True) 
