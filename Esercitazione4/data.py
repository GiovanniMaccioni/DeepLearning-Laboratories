import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.datasets import MNIST
from torch.utils.data import Subset
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, FakeData
from torchvision.datasets import STL10

def get_data(dataset, valid_size):
    if dataset== "MNIST":
        train_set, test_set, validation_set = get_MNIST(valid_size)

    elif dataset == "CIFAR10":
        train_set, test_set, validation_set = get_CIFAR10(valid_size)
    
    elif dataset == "STL10":
        train_set, test_set, validation_set = get_STL10(valid_size)
    
    elif dataset == "CIFAR100":
        train_set, test_set, validation_set = get_CIFAR100(valid_size)
    
    elif dataset == "Fake":
        train_set, test_set, validation_set = get_Fake(valid_size)
    
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
    print(f"Trainset len: {len(train_set)}")
    print(f"Validation len: {len(validation_set)}")
    print(f"Test len: {len(test_set)}")

    return train_set, test_set, validation_set

def get_CIFAR10(validation_size = None):
    transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])

    train_set = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = CIFAR10(root='./data', train=False, download=True, transform=transform)
    validation_set = None

    #if validation_size != None:
    I = np.random.permutation(len(train_set))
    validation_set = Subset(train_set, I[:validation_size])
    train_set = Subset(train_set, I[validation_size:])

    return train_set, test_set, validation_set

def get_CIFAR100(validation_size = None):
    transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                        ])

    train_set = CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_set = CIFAR100(root='./data', train=False, download=True, transform=transform)
    validation_set = None

    #if validation_size != None:
    I = np.random.permutation(len(train_set))
    validation_set = Subset(train_set, I[:validation_size])
    train_set = Subset(train_set, I[validation_size:])

    return train_set, test_set, validation_set

def get_STL10(validation_size = None):
    transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                        ])

    train_set = STL10(root='./data', split='train', download=True, transform=transform)
    test_set = STL10(root='./data', split='test', download=True, transform=transform)
    validation_set = None

    #if validation_size != None:
    I = np.random.permutation(len(train_set))
    validation_set = Subset(train_set, I[:validation_size])
    train_set = Subset(train_set, I[validation_size:])

    return train_set, test_set, validation_set

def get_Fake(validation_size = None):
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        
    ds_fake = FakeData(validation_size, image_size=(3, 32, 32), transform=transform)
    #dl_fake = torch.utils.data.DataLoader(ds_fake, batch_size=config["batch_size"], shuffle=False, num_workers=2)
    return ds_fake, None, None

def make_loader(set, batch_size, shuffle = False, drop_last = False):
    loader = torch.utils.data.DataLoader(dataset=set, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return loader

