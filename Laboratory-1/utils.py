import numpy as np
import torch
import matplotlib.pyplot as plt

import models

def load_model(path):
    m = torch.load(path)
    config = m['config']
    #print(config)
    if(config["model"] == "MLP"):
        model = models.MLP(config['input_size'], config['num_hidden_layers'], config['width'], config['num_classes'], config['residual_step'])
    elif(config["model"] == "ResNet"):
        model = models.ResNet(config['in_channels'], config['out_channels'], config['num_classes'], config['num_conv_per_level'], config['num_levels'], config['residual_step'])
    model.load_state_dict(m['model'])
    return model, config

def save_image(image, path, title, colormap=None):
    image = image.permute(1, 2, 0)
    height = image.shape[0]
    width = image.shape[1]
    fig = plt.figure(figsize=(height/10, width/10), frameon=False)#, layout='tight')
    if colormap != None:
        plt.imshow(image, cmap=colormap)
    else:
        plt.imshow(image)
    plt.axis('off')
    plt.title(title)
    fig.savefig(path)#, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def save_mix(image, heat, path, title, colormap):
    image = image.permute(1, 2, 0)
    heat = heat.permute(1, 2 ,0)
    height = image.shape[0]
    width = image.shape[1]

    fig = plt.figure(figsize=(height/10, width/10), frameon=False)#, layout='tight')
    plt.imshow(image)
    plt.imshow(heat, cmap=colormap, alpha=0.33)
    plt.axis('off')
    plt.title(title)
    fig.savefig(path)#, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def get_labels(dataset):
    if dataset == 'MNIST':
        labels_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    elif dataset == 'CIFAR10':
        labels_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    elif dataset == 'CIFAR100':
        labels_list = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',\
                        'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', \
                        'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', \
                        'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',\
                        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', \
                        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',\
                        'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter',\
                        'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain',\
                        'plate', 'poppy', 'porcupine', 'possum', 'rabbit', \
                        'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal',\
                        'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',\
                        'spider', 'squirrel', 'streetcar', 'sunflower', \
                        'sweet_pepper', 'table', 'tank', 'telephone', \
                        'television', 'tiger', 'tractor', 'train', \
                        'trout', 'tulip', 'turtle', 'wardrobe', \
                        'whale', 'willow_tree', 'wolf', 'woman','worm']

    return labels_list

def denormalize(data, dataset):
    if dataset == 'MNIST':
        data[:,0,:, :] = data[:,0,:, :]*0.1307 + 0.3081
    elif dataset == 'CIFAR10':
        data[:,0,:, :] = data[:,0,:, :]*0.2023 + 0.4914
        data[:,1,:, :] = data[:,1,:, :]*0.1994 + 0.4822
        data[:,2,:, :] = data[:,2,:, :]*0.2010 + 0.4465
    elif dataset == 'CIFAR100':
        data[:,0,:, :] = data[:,0,:, :]*0.2023 + 0.4914
        data[:,1,:, :] = data[:,1,:, :]*0.1994 + 0.4822
        data[:,2,:, :] = data[:,2,:, :]*0.2010 + 0.4465

    return data
