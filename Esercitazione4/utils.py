import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, confusion_matrix, ConfusionMatrixDisplay

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

def normalize(data, dataset):
    if dataset == 'MNIST':
        data[:,0,:, :] = (data[:,0,:, :]-0.3081)/0.1307
    elif dataset == 'CIFAR10':
        data[:,0,:, :] = (data[:,0,:, :]- 0.4914)/0.2023 
        data[:,1,:, :] = (data[:,1,:, :]- 0.4822)/0.1994
        data[:,2,:, :] = (data[:,2,:, :]- 0.4465)/0.2010
    elif dataset == 'CIFAR100':
        data[:,0,:, :] = (data[:,0,:, :]- 0.4914)/0.2023 
        data[:,1,:, :] = (data[:,1,:, :]- 0.4822)/0.1994
        data[:,2,:, :] = (data[:,2,:, :]- 0.4465)/0.2010

    return data

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

def collect_logits(model, dl, device):
    logits = []
    with torch.no_grad():
        for (Xs, _) in dl:
            logits.append(model(Xs.to(device)).cpu().numpy())
    return np.vstack(logits)

def select_subset(dataset, label_list):
    samples = []
    labels = []
    for data in dataset:
        if data[1] in label_list:
            samples.append(data[0])
            labels.append(torch.tensor(data[1]))

    samples = torch.stack(samples)
    labels = torch.stack(labels)
    newset = torch.utils.data.TensorDataset(samples, labels)
    return newset

def plot_ROC(logits_ID, logits_OOD, metric):

    #Labels of the data. In this case 1 for in distribution, 0 for out_of_distribution
    y = np.concatenate((np.ones(len(logits_ID)), np.zeros(len(logits_OOD))))

    if metric == "mean":
        list_ID_ = [ logits_ID[i].mean()   for i in range(len(logits_ID))]
        list_OOD_ = [ logits_OOD[i].mean() for i in range(len(logits_OOD))]
    if metric == "max":
        list_ID_ = [ logits_ID[i].max()   for i in range(len(logits_ID))]
        list_OOD_ = [ logits_OOD[i].max() for i in range(len(logits_OOD))]
    if metric == "var":
        list_ID_ = [ logits_ID[i].var()   for i in range(len(logits_ID))]
        list_OOD_ = [ logits_OOD[i].var() for i in range(len(logits_OOD))]

    scores = list_ID_ + list_OOD_

    RocCurveDisplay.from_predictions( y, scores, name="", color="darkorange",)

    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("OOD Detetection on CIFAR 10")
    plt.legend()
    plt.show()
    return

def plot_PrecisionRecall(logits_ID, logits_OOD, metric):
    #Labels of the data. In this case 1 for in distribution, 0 for out_of_distribution
    y = np.concatenate((np.ones(len(logits_ID)), np.zeros(len(logits_OOD))))

    if metric == "mean":
        list_ID_ = [ logits_ID[i].mean()   for i in range(len(logits_ID))]
        list_OOD_ = [ logits_OOD[i].mean() for i in range(len(logits_OOD))]
    if metric == "max":
        list_ID_ = [ logits_ID[i].max()   for i in range(len(logits_ID))]
        list_OOD_ = [ logits_OOD[i].max() for i in range(len(logits_OOD))]
    if metric == "var":
        list_ID_ = [ logits_ID[i].var()   for i in range(len(logits_ID))]
        list_OOD_ = [ logits_OOD[i].var() for i in range(len(logits_OOD))]

    scores = list_ID_ + list_OOD_

    no_skill = len(logits_ID) / (len(logits_ID) + len(logits_OOD))
    PrecisionRecallDisplay.from_predictions(y, scores, name="")
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.title("OOD Detetection on CIFAR 10")
    plt.show()     
    return

def plot_histograms(x, y):
    _ = plt.hist(x, 50, density=True, alpha=0.5, label='ID')
    _ = plt.hist(y, 50, density=True, alpha=0.5, label='OOD')
    plt.legend()
    plt.title("OOD Detetection on CIFAR 10")
    plt.show()

def select_subset(dataset, label_list):
    samples = []
    labels = []
    for data in dataset:
        if data[1] in label_list:
            samples.append(data[0])
            labels.append(torch.tensor(data[1]))

    samples = torch.stack(samples)
    labels = torch.stack(labels)
    newset = torch.utils.data.TensorDataset(samples, labels)
    return newset

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=labels)
    disp.plot()
    plt.show()
    return


