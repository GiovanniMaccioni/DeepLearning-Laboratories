import numpy as np
import torch
import torch.nn as nn

import data as dt
from utils import *

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
    'num_images_to_save': 0,
    'OOD_dataset': "CIFAR100"
}

#config_model = {}
load_model_as = ""


def model_pipeline(config, device):
    
    model, config_model = load_model("./model_weights/"+load_model_as)
    model = model.to(device)
    config = {**config_model, **config}
        
    _, test_loader, _, ood_loader, _, _ = make(model, config)

    nparameters = sum(p.numel() for p in model.parameters())
    print("Number of Parameters: ", nparameters)
    
    logits_ID, logits_OOD = OOD_(model, test_loader, ood_loader, device)

    return model, logits_ID, logits_OOD

def make(model, config):
    #make data
    _, test_set, _ = dt.get_data(config["dataset"], config["validation_size"])
    
    ood_set, _, _ = dt.get_data(config["OOD_dataset"], config["validation_size"])
    if(config["OOD_dataset"] == "CIFAR100"):
        #select subset
        label_list = [3, 8, 11, 12, 35, 41, 43, 48, 76, 90, 95, 97]
        ood_set = select_subset(ood_set, label_list)

    #train_loader = dt.make_loader(train_set, config['batch_size'], shuffle=True)
    test_loader = dt.make_loader(test_set, config['batch_size'], drop_last=True)
    #validation_loader = dt.make_loader(validation_set, config['batch_size'])

    ood_loader = dt.make_loader(ood_set, config['batch_size'])
        
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    return None, test_loader, None, ood_loader, criterion, optimizer

def OOD_(model, id_loader, ood_loader, device):
    logits_ID = collect_logits(model, id_loader, device)
    logits_OOD = collect_logits(model, ood_loader, device)

    plot_histograms(logits_ID.max(1), logits_OOD.max(1))
    plot_histograms(logits_ID.var(1), logits_OOD.var(1))
    plot_histograms(logits_ID.mean(1), logits_OOD.mean(1))
    plot_ROC(logits_ID, logits_OOD, "max")
    plot_PrecisionRecall(logits_ID, logits_OOD, "max")
    plot_ROC(logits_ID, logits_OOD, "var")
    plot_PrecisionRecall(logits_ID, logits_OOD, "var")
    plot_ROC(logits_ID, logits_OOD, "mean")
    plot_PrecisionRecall(logits_ID, logits_OOD, "mean")

    return logits_ID, logits_OOD

if __name__=="__main__":

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    parser = argparse.ArgumentParser(description='Laboratory 4')

    #Arguments for config_pipe
    #parser.add_argument("--model_name", type=str, default="MLP", help='Name of the Model')
    parser.add_argument("--val_size", type=int, default=5000, help='Validation Size')
    #parser.add_argument("--num_epochs", type=int, default=1, help='Number of Epochs')
    #parser.add_argument("--lr", type=float, default=3e-4, help='Learning rate')
    parser.add_argument("--batch_size", type=int, default=128, help='Batch Size')
    parser.add_argument("--epsilon", type=float, default=0.001, help='Epsilon for adversarial attack')
    parser.add_argument("--class_target", type=int, default=None, help='if mode is same, you need to specify the class target')
    parser.add_argument("--iterations", type=int, default=0, help='')
    parser.add_argument("--num_images_to_save", type=int, default=0, help='')
    parser.add_argument("--OOD_dataset", type=str, default="CIFAR100", help='Chose an out of distribution dataset: Fake or CIFAR100')

    #General Arguments
    parser.add_argument("--load_model_as", type=str, default="def.pt", help='Specify a <name>.pt to load the model; used if train==False')

    args = parser.parse_args()

    #Populate config_pipe
    #config_pipe['model'] = args.model_name
    config_pipe['validation_size'] = args.val_size
    #config_pipe['epochs'] = args.num_epochs
    #config_pipe['learning_rate'] = args.lr
    config_pipe['batch_size'] = args.batch_size
    config_pipe['epsilon'] = args.epsilon
    config_pipe['class_target']= args.class_target
    config_pipe['iterations']= args.iterations
    config_pipe['num_images_to_save']= args.num_images_to_save

    #General Arguments
    load_model_as = args.load_model_as

    #setting the seed to have the same split of validation and train
    np.random.seed(seed=102)

    config = config_pipe

    model, logits_ID, logits_OOD = model_pipeline(config, device)