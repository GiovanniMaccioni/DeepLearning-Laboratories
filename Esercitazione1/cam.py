import torch
import torch.nn as nn
from tqdm import tqdm
import torchvision

import matplotlib.pyplot as plt

import models
import data as dt

from utils import *

import argparse

num_images_to_save = None 

def loop_(model, model_trunc, linear_weights, dataset, test_loader):#,device
    model.eval()
    model_trunc.eval()
    num_batches = len(test_loader)
    count=0

    with torch.no_grad():
        for (data, labels) in tqdm(test_loader, desc=f'Activation Maps', leave=True):
            data = data#.to(device)
            logits = model(data)
            preds = torch.argmax(logits, dim=1)
            preds = preds.detach()#.cpu()

            #get the feature maps from the last convolutional layer
            feature_maps = model_trunc(data)

            #get the linear weights relative to the predicted class for all the samples in the batch
            linear_weights_stack = []
            for i in range(data.shape[0]):
                linear_weights_stack.append(linear_weights[preds[i]])
            
            linear_weights_stack = torch.stack((linear_weights_stack))

            #Compute the activation maps
            activation_maps = cam(linear_weights_stack, feature_maps, data.shape)

            data_cpu = data.detach().cpu().clone()
            
            #Revert the Normalization
            data_cpu = denormalize(data_cpu, dataset)

            for i in range(data.shape[0]):
                #save the original image, the activation map, and a composition of these two; images are saved with the true label and the predicted label in the title
                #TODO add denormalization
                label_list = get_labels(dataset)
                save_image(data_cpu[i], f"./cam_results/imf_{label_list[labels[i]]}_p_{label_list[preds[i]]}_{count}.png", "Original")
                save_image(activation_maps[i].detach().cpu(), f"./cam_results/actv_map_{label_list[labels[i]]}_p_{label_list[preds[i]]}_{count}.png","Activation Map", colormap='jet')
                #mix = 0.7*data_cpu[i] + 0.3*activation_maps[i]
                #save_image(activation_maps[i].detach().cpu(), f"./cam_results/actv_map_{label_list[labels[i]]}_p_{label_list[preds[i]]}_{count}.png", colormap='jet')
                save_mix(data_cpu[i], activation_maps[i].detach().cpu(), f"./cam_results/mix_{label_list[labels[i]]}_p_{label_list[preds[i]]}_{count}.png", f"GT:{label_list[labels[i]]}  P:{label_list[preds[i]]}", colormap='jet')
                count +=1
                if count == num_images_to_save:
                    break

            if count == num_images_to_save:#TOCHECK Maybe i don't need that
                break    
    return

def cam(linear_weights, feature_maps, image_shape):
    """
    linear weights: batch of linear weights. Each sample correspond to the linear weights for a specific class
    feature_maps: output of the last convolutional layer
    image_shape: shape of the image; needed to have resize reference
    """

    #Take the shape of the feature maps
    b, c, h, w = feature_maps.shape

    #First reshape the feature maps to perform matmul after
    feature_maps_reshaped = torch.reshape(feature_maps, (b, c, h*w))
    #Then compute the activation maps
    activation_maps = torch.matmul(torch.unsqueeze(linear_weights, 1), feature_maps_reshaped)

    #In conclusion we need to recover the [B, C, H, W] structure and resize to the original image shape
    activation_maps = torch.reshape(activation_maps, (b, 1, h, w))
    activation_maps = torchvision.transforms.functional.resize(activation_maps, (image_shape[2], image_shape[3]))

    return activation_maps


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='CAM')

    #General Arguments
    #parser.add_argument("--load_model_path", type=str, default="./model_weights/", help='Convolutional model to compute the activation maps')
    parser.add_argument("--saved_model", type=str, default="def.pt", help='Specify a <name>.pt to load the model')
    parser.add_argument("--num_images_to_save", type=int, default=1, help='Number of triplets of images to save')

    args = parser.parse_args()

    #setting the seed to have the same split of validation and train
    np.random.seed(seed=102)
    
    model, config = load_model("./model_weights/" + args.saved_model)

    #Truncate the model before the global average pooling, in correspondence to the last convolutionl layer
    #The index to truncate at, may vary with the architecture; -2 for the ResNet implemented
    model_trunc = nn.Sequential(*list(model.children())[:-2])

    #Take the Linear Parameters realtive to the layer just after the Global Average Pooling
    linear_parameters = list(model.linear.parameters())
    linear_weights = linear_parameters[0]

    _, test_set, _ = dt.get_data(config['dataset'], 5000)

    test_loader = dt.make_loader(test_set, 50)

    num_images_to_save = args.num_images_to_save

    loop_(model, model_trunc, linear_weights, config['dataset'], test_loader)







