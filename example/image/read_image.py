from PIL import Image
from torchvision import transforms

file_path = "188242.jpg"
im = Image.open(file_path)
out = transforms.ToTensor()(im)
print(out.shape)
print(out[2][217][177])
