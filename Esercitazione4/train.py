import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import utils as ut
import wandb

def train(model, train_loader, validation_loader, criterion, optimizer, config, device):
    epochs = config['epochs']
    for epoch in range(epochs):
        loss_epoch = train_batch(model, train_loader, criterion, optimizer, epoch, device)
        val_accuracy, val_loss = evaluate_batch(model, validation_loader, criterion, device)

        wandb.log({"Train Loss": loss_epoch, "Validation Accuracy": val_accuracy, "Validation Loss": val_loss})

    return 

def train_batch(model, train_loader, criterion, optimizer, epoch, device):
    model.train()
    running_loss = 0
    num_batches = len(train_loader)
    for (data, labels) in tqdm(train_loader, desc=f'Training epoch {epoch}', leave=True):
        data = data.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()/num_batches #TOCHECK
        
    
    return running_loss

def evaluate_batch(model, loader, criterion, device):
    model.eval()
    accuracy = 0
    running_loss = 0
    #num_batches = len(validation_loader)
    len_data = len(loader)*(loader.batch_size)
    with torch.no_grad():
        for (data, labels) in tqdm(loader, desc=f'Evaluating', leave=True):
            data = data.to(device)
            labels = labels.to(device)
            logits = model(data)
            loss = criterion(logits, labels)

            preds = torch.argmax(logits, dim=1)
            preds = preds.detach().cpu()
            labels = labels.cpu()
            
            accuracy += (preds==labels).float().sum()
            running_loss += loss.item()

    return accuracy/len_data, running_loss/len_data


def fgsm(data, labels, model, criterion, epsilon, dataset, class_target, device, norm=False):
    data_copy = data.clone()

    if norm:
        data = ut.normalize(data, dataset)

    data = data.requires_grad_()

    model.zero_grad()
    logits = model(data)

    preds = torch.argmax(logits, dim=1)
    #preds = preds.detach().cpu()

    if class_target != None:
        targets = torch.tensor(class_target).repeat(len(labels)).to(device)
    else:
        targets = labels.clone().to(device)

    loss = criterion(logits, targets)
    loss.backward()

    if not norm:
        data_copy = ut.denormalize(data_copy, dataset)

    sign = data.grad.data.sign()
    if class_target != None:
        sign = -sign

    adv_ex = data_copy + epsilon*sign
    adv_ex = torch.clamp(adv_ex, 0, 1)

    model.zero_grad()

    return adv_ex, preds


def evaluate_batch_adv(model, loader, criterion, class_target, iterations, epsilon, dataset, device, num_images):
    #model.eval()#TOCHECK does it have to be deleted?
    accuracy_preds_lab = 0
    accuracy_adv_preds = 0
    accuracy_adv_lab = 0

    preds_adv_list = []
    labels_list = []
    len_data = len(loader)*(loader.batch_size)
    sample_counter = 0
    for (data, labels) in tqdm(loader, desc=f'Evaluating', leave=True):

        data = data.to(device)
        labels.to(device)

        if class_target == None:
            adv_samples, preds = fgsm(data, labels, model, criterion, epsilon, dataset, class_target, device)
        else:
            for i in range(iterations):
                if i == 0:
                    adv_samples, preds = fgsm(data, labels, model, criterion, epsilon, dataset, class_target, device)
                else:
                    adv_samples, _ = fgsm(adv_samples, labels, model, criterion, epsilon, dataset, class_target, device, norm=True)

        with torch.no_grad():
            logits_adv = model(adv_samples)

        preds_adv = torch.argmax(logits_adv, dim=1)
        preds_adv = preds_adv.detach().cpu()

        preds_adv_list.append(preds_adv.clone().detach().cpu().numpy())
        labels_list.append(labels.clone().detach().cpu().numpy())
    
        preds = preds.detach().cpu()

        if num_images > 0:
            for i, adv in enumerate(adv_samples):
                t = adv.detach().clone().cpu()
                labels_names = ut.get_labels(dataset)
                if class_target == None:
                    ut.save_image(t, "./adv_samples/"+f"/{i}_non_targ_eps{epsilon}_.png", f"EPS: {epsilon} GT: {labels_names[labels[i]]} ADV:{labels_names[preds_adv[i]]}")
                else:
                    ut.save_image(t, "./adv_samples/"+f"/{i}_targ_ct_eps{epsilon}_{class_target}_it{iterations}.png", f"EPS: {epsilon} GT: {labels_names[labels[i]]} ADV:{labels_names[preds_adv[i]]}")
                num_images = num_images - 1
                if num_images == 0:#Added to save only the first 20 samples from the first batch of generated adversarial examples
                    break

        accuracy_preds_lab+= (preds==labels).float().sum()
        accuracy_adv_preds += (preds_adv==preds).float().sum()
        accuracy_adv_lab += (preds_adv==labels).float().sum()

    #preds_adv_list = np.concatenate(preds_adv_list, axis = None)
    #labels_list = np.concatenate(labels_list, axis = None)
    #ut.plot_confusion_matrix(labels_list, preds_adv_list, ut.get_labels(dataset))
    
    return accuracy_preds_lab/len_data, accuracy_adv_preds/len_data, accuracy_adv_lab/len_data

def train_adv(model, train_loader, validation_loader, criterion, optimizer, config, device):
    epochs = config['epochs']
    for epoch in range(epochs):
        loss_epoch = train_adv_batch(model, train_loader, criterion, config['epsilon'],  optimizer, epoch, config['dataset'], device)
        accuracy_preds_lab, accuracy_adv_preds, accuracy_adv_lab = evaluate_batch_adv(model, validation_loader, criterion, config['class_target'], config['iterations'], config['epsilon'],\
                                                      config['dataset'], device, config['num_images_to_save'])

        wandb.log({"Train Loss": loss_epoch, "accuracy_preds_lab": accuracy_preds_lab, "accuracy_adv_preds": accuracy_adv_preds, "accuracy_adv_lab": accuracy_adv_lab})

    return

def train_adv_batch(model, train_loader, criterion, epsilon, optimizer, epoch, dataset, device):
    model.train()
    running_loss = 0
    
    len_data = len(train_loader)*(train_loader.batch_size)
    for (data, labels) in tqdm(train_loader, desc=f'Training epoch {epoch}', leave=True):
        data = data.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        adv_samples, _ = fgsm(data, labels, model, criterion, epsilon, dataset, None, device)

        new_batch = torch.cat((data, adv_samples))
        new_labels = torch.cat((labels, labels))

        #TOCHECK
        logits = model(new_batch)

        loss = criterion(logits, new_labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() #TOCHECK
    
    return running_loss/(2*len_data)