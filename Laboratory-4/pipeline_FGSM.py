import numpy as np
import torch
import torch.nn as nn

import data as dt
import train as tr
import models
from utils import *

import wandb

import argparse

config_pipe = {
    'model': "ResNet",
    'validation_size': 5000,
    'epochs': 1,
    'learning_rate': 3e-4,
    'batch_size': 128,
    'epsilon': 0.5,
    'class_target': None,
    'iterations': 0,
    'num_images_to_save': 0
}

config_MLP = {
    'model': "MLP",
    'dataset': "MNIST",
    'input_size': 28*28,
    'num_hidden_layers': 4,
    'width': 16,
    'num_classes': 10,
    'residual_step': 0
}

config_ResNet = {
    'model': "ResNet",
    'dataset': "CIFAR10",
    'in_channels': 3,
    'out_channels': 64,
    'num_classes': 10, 
    'num_conv_per_level': 2,
    'num_levels': 2,
    'residual_step': 1
}

"""config_ResNet = {
    'model': "ResNet",
    'dataset': "CIFAR10",
    'in_channels': 3,
    'out_channels': 64,
    'num_classes': 10, 
    'num_conv_per_level': [4,4,4,4],
    'num_levels': 4,
    'residual_step': 2
}"""

config_model = {}
load_model_as = ""
#save_model_path = ""
wandb_mode = ""
project_name = ""

def load_config(model_name):
    if(model_name== "MLP"):
        config = config_MLP
    elif(model_name == "ResNet"):
        config = config_ResNet
    return config

def model_pipeline(config, device):
    global config_model
    if (train == False) & (train_adv == False):
        model, config_model = load_model("./model_weights/"+load_model_as)
        model = model.to(device)
        config = {**config_model, **config}
    else:
        if(config["model"] == "MLP"):
            model = models.MLP(config['input_size'], config['num_hidden_layers'], config['width'], config['num_classes'], config['residual_step']).to(device)
        elif(config["model"] == "ResNet"):
            model = models.ResNet(config['in_channels'], config['out_channels'], config['num_classes'], config['num_conv_per_level'], config['num_levels'], config['residual_step']).to(device)
        
    with wandb.init(project=project_name, config = config, mode=wandb_mode):#"disabled"
        config = wandb.config
        train_loader, test_loader, validation_loader, criterion, optimizer = make(model, config)

        nparameters = sum(p.numel() for p in model.parameters())
        print("Number of Parameters: ", nparameters)

        wandb.watch(model, criterion, log="all", log_freq=1)
        if train: 
            tr.train(model, train_loader, validation_loader, criterion, optimizer, config, device)
            test_acc, test_loss = tr.evaluate_batch(model, test_loader, criterion, device)
            print(f"Test Accuracy: {test_acc}   Test Loss: {test_loss}")

        elif train_adv:
            tr.train_adv(model, train_loader, validation_loader, criterion, optimizer, config, device)
            acc1, acc2, acc3 = tr.evaluate_batch_adv(model, test_loader, criterion, config['class_target'], config['iterations'], config['epsilon'],\
                                                      config['dataset'], device, config['num_images_to_save'])
            
            print(f"accuracy_preds_lab: {acc1}   accuracy_adv_preds: {acc2}    accuracy_adv_lab: {acc3}")

        else:
            acc1, acc2, acc3 = tr.evaluate_batch_adv(model, test_loader, criterion, config['class_target'], config['iterations'], config['epsilon'],\
                                                      config['dataset'], device, config['num_images_to_save'])
            
            print(f"accuracy_preds_lab: {acc1}   accuracy_adv_preds: {acc2}    accuracy_adv_lab: {acc3}")

 
    wandb.finish()
    return model

def make(model, config):
    #make data
    train_set, test_set, validation_set = dt.get_data(config["dataset"], config["validation_size"])
    
    train_loader = dt.make_loader(train_set, config['batch_size'], shuffle=True)
    test_loader = dt.make_loader(test_set, config['batch_size'], drop_last=True)
    validation_loader = dt.make_loader(validation_set, config['batch_size'])
        
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    return train_loader, test_loader, validation_loader, criterion, optimizer

if __name__=="__main__":

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    parser = argparse.ArgumentParser(description='Laboratory 1')

    #Arguments for config_pipe
    parser.add_argument("--model_name", type=str, default="MLP", help='Name of the Model')
    parser.add_argument("--val_size", type=int, default=5000, help='Validation Size')
    parser.add_argument("--num_epochs", type=int, default=1, help='Number of Epochs')
    parser.add_argument("--lr", type=float, default=3e-4, help='Learning rate')
    parser.add_argument("--batch_size", type=int, default=128, help='Batch Size')
    parser.add_argument("--epsilon", type=float, default=0.001, help='Epsilon for adversarial attack')
    parser.add_argument("--class_target", type=int, default=None, help='if mode is same, you need to specify the class target')
    parser.add_argument("--iterations", type=int, default=0, help='')
    parser.add_argument("--num_images_to_save", type=int, default=0, help='')

    #Arguments for config_model
    #For all models
    #parser.add_argument("--model_name", type=str, default="test", help='Name of the Model')
    parser.add_argument("--dataset", type=str, default="MNIST", help='Dataset')
    parser.add_argument("--num_classes", type=int, default=10, help='Number of Classes')
    parser.add_argument("--residual_step", type=int, default=0, help='Residual Step')

    #MLP
    parser.add_argument("--input_size", type=int, default=28*28*1, help='Flattened Size of Input Image (H*W*C)')
    parser.add_argument("--width", type=int, default=16, help='Number of Neurons in the hidden layers')
    parser.add_argument("--num_hidden_layers", type=int, default=4, help='Number of Hidden Layers')

    #ResNet
    parser.add_argument("--in_channels", type=int, default=3, help='Number of Input Image Channels')
    parser.add_argument("--out_channels", type=int, default=64, help='Number of Output Channels for the first block')
    parser.add_argument("--num_levels", type=int, default=4, help='Number of ResNet Block')
    parser.add_argument("--num_conv_per_level", type=int, default=4, help='Number of Convolutions per Block')

    #General Arguments
    parser.add_argument("--wandb", type=str, default="disabled", help='Logging with wandb; default: disabled; Pass online to log')
    parser.add_argument("--wandb_proj_name", type=str, default="Bin", help='Project name on wandb platform')
    parser.add_argument("--train", type=bool, default=False, help='If True train the model; If train==False & train_adv==False Evaluate the model')
    parser.add_argument("--train_adv", type=bool, default=False, help='If True  and train== False, train adversarially the model; If train==False & train_adv==False Evaluate the model')
    parser.add_argument("--load_model_as", type=str, default="def.pt", help='Specify a <name>.pt to load the model; used if train==False')
    #parser.add_argument("--save_model_path", type=str, default="./", help='Path to save the model to')
    parser.add_argument("--save_model_as", type=str, default="def.pt", help='Specify a <name>.pt to save the model')

    args = parser.parse_args()

    #Populate config_pipe
    config_pipe['model'] = args.model_name
    config_pipe['validation_size'] = args.val_size
    config_pipe['epochs'] = args.num_epochs
    config_pipe['learning_rate'] = args.lr
    config_pipe['batch_size'] = args.batch_size
    config_pipe['epsilon'] = args.epsilon
    config_pipe['class_target']= args.class_target
    config_pipe['iterations']= args.iterations
    config_pipe['num_images_to_save']= args.num_images_to_save
    
    #Populate config_MLP. It will be used only if train is True and if model_name is MLP
    config_MLP['model']=args.model_name
    config_MLP['dataset']=args.dataset
    config_MLP['input_size']=args.input_size
    config_MLP['num_hidden_layers']=args.num_hidden_layers
    config_MLP['width']=args.width
    config_MLP['num_classes']=args.num_classes
    config_MLP['residual_step']=args.residual_step

    #Populate config_ResNet. It will be used only if train is True and if model_name is ResNet
    config_ResNet['model']=args.model_name
    config_ResNet['dataset']=args.dataset
    config_ResNet['in_channels']=args.in_channels
    config_ResNet['out_channels']=args.out_channels
    config_ResNet['num_classes']=args.num_classes
    config_ResNet['num_conv_per_level']=args.num_conv_per_level
    config_ResNet['num_levels']=args.num_levels
    config_ResNet['residual_step']=args.residual_step

    #General Arguments
    load_model_as = args.load_model_as
    wandb_mode = args.wandb
    project_name = args.wandb_proj_name
    train = args.train
    train_adv = args.train_adv

    #setting the seed to have the same split of validation and train
    np.random.seed(seed=102)

    if (train == True) | (train_adv == True):
        config_model = load_config(args.model_name)
        config = {**config_model, **config_pipe}
        print(config)
    else:
        config = config_pipe

    model = model_pipeline(config, device)

    torch.save({
        'model': model.state_dict(),
        'config': config_model
        }, "./model_weights/"+args.save_model_as)