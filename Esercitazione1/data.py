import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.datasets import MNIST
from torch.utils.data import Subset
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torchvision.datasets import STL10

def get_data(dataset, valid_size):
    if dataset== "MNIST":
        train_set, test_set, validation_set = get_MNIST(valid_size)

    elif dataset == "CIFAR10":
        train_set, test_set, validation_set = get_CIFAR10(valid_size)
    
    elif dataset == "STL10":
        train_set, test_set, validation_set = get_STL10(valid_size)
    
    return train_set, test_set, validation_set

def get_MNIST(validation_size = None):
    transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])

    train_set = MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = MNIST(root='./data', train=False, download=True, transform=transform)
    validation_set = None

    #if validation_size != None:
    I = np.random.permutation(len(train_set))
    validation_set = Subset(train_set, I[:validation_size])
    train_set = Subset(train_set, I[validation_size:])

    return train_set, test_set, validation_set

def get_CIFAR10(validation_size = None):
    transform_train = transforms.Compose([
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.RandomCrop(32, padding=4),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])
    
    transform_test = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])

    train_set = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_set = CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    validation_set = None

    #if validation_size != None:
    I = np.random.permutation(len(train_set))
    validation_set = Subset(train_set, I[:validation_size])
    train_set = Subset(train_set, I[validation_size:])

    return train_set, test_set, validation_set

def get_STL10(validation_size = None):
    transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])

    train_set = STL10(root='./data', split='train', download=True, transform=transform)
    test_set = STL10(root='./data', split='test', download=True, transform=transform)
    validation_set = None

    #if validation_size != None:
    I = np.random.permutation(len(train_set))
    validation_set = Subset(train_set, I[:validation_size])
    train_set = Subset(train_set, I[validation_size:])

    return train_set, test_set, validation_set

def make_loader(set, batch_size, shuffle = False, drop_last = False):
    loader = torch.utils.data.DataLoader(dataset=set, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return loader