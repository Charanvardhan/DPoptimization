import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from opacus import PrivacyEngine
from opacus.utils.uniform_sampler import UniformWithReplacementSampler


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#dataloader
transform = transforms.compose([    #wraper
    transforms.RandomHorizontalFlip(), # randomly flips  the image horizontally with prob 0.5
    transforms.RandomCrop(32, padding=4), # crops image size to 32x32 
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform)

#Data Loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=256, shuffle=False)

