import torch
import torch.nn as nn
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
import PIL

import matplotlib.pyplot as plt

import models
import data as dt
from utils import *


def cam(linear_weights, feature_maps, images, preds):#FIXME non utilizzo l'informazione della classe!!!
    b, c, h, w = feature_maps.shape

    feature_maps_reshaped = torch.reshape(feature_maps, (b, c, h*w))
    activation_maps = torch.matmul(torch.unsqueeze(linear_weights, 1), feature_maps_reshaped)

    activation_maps = torch.reshape(activation_maps, (b, 1, h, w))
    activation_maps = torchvision.transforms.functional.resize(activation_maps, 32)

    return activation_maps


if __name__=="__main__":
 
    model, config = load_model("./model_weights/res_nl_2_nc_4_res_2.pt")

    model_trunc = nn.Sequential(*list(model.children())[:-2])#-2

    linear_parameters = list(model.linear.parameters())
    linear_weights = linear_parameters[0]

    with torch.no_grad():
        #im = torchvision.io.read_image("./Frog_tree.jpg")
        im = PIL.Image.open("./cat2.jpg")

        transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])
        
        """width = im.shape[1]
        height = im.shape[2]
        channels = im.shape[0]"""

        #im = im.unsqueeze(0).to(torch.float32)
        im = transform(im)
        im = im.unsqueeze(0)

        original = im.clone()
        original = denormalize(original, config['dataset'])

        #save_image(original[0].detach().cpu(), f"./cam_results/original.png")

        width = im.shape[2]
        height = im.shape[3]
        channels = im.shape[1]

        im = torchvision.transforms.functional.resize(im, (32, 32))

        logits = model(im)
        preds = torch.argmax(logits, dim=1)
        preds = preds.detach()#.cpu()

        print(preds.shape)

        feature_maps = model_trunc(im)

        linear_weights = linear_weights[preds]

        activation_map = cam(linear_weights, feature_maps, im, preds)

        im = torchvision.transforms.functional.resize(im, (width, height))
        activation_map = torchvision.transforms.functional.resize(activation_map, (width, height))

        im = denormalize(im, config['dataset'])

        label_list = get_labels(config['dataset'])
        save_image(im[0].detach().cpu(), f"./cam_results/single_orig_cat_p_{label_list[preds[0]]}.png", "Base")
        save_image(activation_map[0].detach().cpu(), f"./cam_results/single_actv_map_cat_p_{label_list[preds[0]]}.png","Activation Map", colormap='jet')
        #mix = 0.7*data_cpu[i] + 0.3*activation_maps[i]
        #save_image(activation_maps[i].detach().cpu(), f"./cam_results/actv_map_{label_list[labels[i]]}_p_{label_list[preds[i]]}_{count}.png", colormap='jet')
        save_mix(original[0].detach().cpu(), activation_map[0].detach().cpu(), f"./cam_results/single_mix_cat_p_{label_list[preds[0]]}.png", f"GT:cat P:{label_list[preds[0]]}", colormap='jet')

