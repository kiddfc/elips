import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.utils import make_grid

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image
from IPython.display import display

import warnings
warnings.filterwarnings('ignore')

train_transform = transforms.Compose([
    # transforms.RandomRotation(10),
    # transforms.RandomHorizontalFlip(),
    # transforms.Resize(224),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder('shapes_dataset_LR',transform=train_transform)
torch.manual_seed(42)
train_data, test_data = torch.utils.data.random_split(dataset, [9000, 1000])

class_names = dataset.classes

train_loader = DataLoader(train_data, batch_size = 10, shuffle = True)
test_loader = DataLoader(test_data, batch_size = 10)

for images, labels in train_loader:
    break

im = make_grid(images, nrow=5)

inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.486/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

im_inv = inv_normalize(im)
print(labels)
plt.figure(figsize=(12,4))
plt.imshow(np.transpose(im_inv.numpy(), (1,2,0)))
plt.show()


class ConvolutionalNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(54 * 54 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 54 * 54 * 16)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)

        return F.log_softmax(X, dim=1)

torch.manual_seed(101)
CNNmodel = ConvolutionalNetwork()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(CNNmodel.parameters(), lr=0.001)



# # to count each class in validation set
# arr = np.array(np.array(dataset.imgs)[test_data.indices, 1], dtype=int)
# cnt = np.zeros((6,1), dtype = int)
# for i in range(1000):
#     for j in range(6):
#         if arr[i] == j:
#             cnt[j] += 1
#             break
# print(cnt)


# for reproducable results
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


#The compose function allows for multiple transforms
#transforms.ToTensor() converts our PILImage to a tensor of shape (C x H x W) in the range [0,1]
#transforms.Normalize(mean,std) normalizes a tensor to a (mean, std) for (R, G, B)
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#
# train_set = torchvision.datasets.CIFAR10(root='./cifardata', train=True, download=True, transform=transform)
#
# test_set = torchvision.datasets.CIFAR10(root='./cifardata', train=False, download=True, transform=transform)
#
# classes = ('0', '1', '2', '3', '4', '5')

# x = torch.rand(5, 3)
# print(x)