import torch
from tqdm import tqdm

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



