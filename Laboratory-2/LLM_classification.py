from transformers import GPT2Model, AutoTokenizer, DistilBertModel
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from tqdm import tqdm

from datasets import load_dataset

import argparse

import wandb


def preprocess_function(data):
    return tokenizer(data["text"], truncation=True, padding=True, return_tensors="pt", add_special_tokens=True)##

class tweet_eval(Dataset):
    def __init__(self, split):
        
        data = load_dataset("tweet_eval", "sentiment")
        self.split = split
        self.tokenized_data = data.map(preprocess_function, batched=True, batch_size=(len(data[split])))
        
        """for i in self.tokenized_data[self.split][:]["input_ids"]:
            print(self.split,":", len(i))"""

    def __len__(self):
        return len(self.tokenized_data[self.split])

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.tokenized_data[self.split][idx]["input_ids"])
        label = torch.tensor(self.tokenized_data[self.split][idx]["label"])
        attention_mask = torch.tensor(self.tokenized_data[self.split][idx]["attention_mask"])
        return input_ids, label, attention_mask

def make_loader(set, batch_size, collator=None, shuffle = False):
    loader = torch.utils.data.DataLoader(dataset=set, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)
    return loader

def train(llm, classifier, train_loader, validation_loader, criterion, optimizer, epochs, device):
    for epoch in range(epochs):
        loss_epoch = train_batch(llm, classifier, train_loader, criterion, optimizer, epoch, device)
        val_accuracy, val_loss = evaluate_batch(llm, classifier, validation_loader, criterion, device)
        classifier.train()

        wandb.log({"Train Loss": loss_epoch, "Validation Accuracy": val_accuracy, "Validation Loss": val_loss})

    return llm, classifier

def train_batch(llm, classifier, train_loader, criterion, optimizer, epoch, device):
    llm.eval()
    classifier.train()
    running_loss = 0
    num_batches = len(train_loader)
    for (ids,  labels, attention) in tqdm(train_loader, desc=f'Training epoch {epoch}', leave=True):
        ids = ids.to(device)
        labels = labels.to(device)
        attention = attention.to(device)

        optimizer.zero_grad()
        with torch.no_grad():
            out = llm(ids, attention)

        logits = classifier(out)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()/num_batches #TOCHECK

    return running_loss

def evaluate_batch(llm, classifier, loader, criterion, device):
    llm.eval()
    classifier.eval()
    accuracy = 0
    running_loss = 0
    #num_batches = len(validation_loader)
    len_data = len(loader)*(loader.batch_size)
    with torch.no_grad():
        for (ids,  labels, attention) in tqdm(loader, desc=f'Evaluating', leave=True):
            ids = ids.to(device)
            labels = labels.to(device)
            attention = attention.to(device)
           
            out = llm(ids, attention)
            logits = classifier(out)

            loss = criterion(logits, labels)

            preds = torch.argmax(logits, dim=1)
            preds = preds.detach().cpu()
            labels = labels.cpu()
            
            accuracy += (preds==labels).float().sum()
            running_loss += loss.item()

    return accuracy/len_data, running_loss/len_data

class LLM_(nn.Module):
    def __init__(self, llm_name):
        super().__init__()
        self.llm_name = llm_name
        if(llm_name == "GPT2"):
            self.llm = GPT2Model.from_pretrained('gpt2')
        elif(llm_name == "BERT"):
            self.llm = DistilBertModel.from_pretrained('distilbert-base-uncased')

    def forward(self, x, a):
        if(self.llm_name == "GPT2"):
            #Indiced 
            indices = (a.sum(1) - 1).tolist()
            o = self.llm(x).last_hidden_state#[:,-1,:]
            for i, ind in enumerate(indices):
                if i == 0:
                    d = o[i, ind, :][None, :]
                else:
                    d = torch.cat((d, o[i, ind, :][None, :]))
            o = d
        elif(self.llm_name == "BERT"):
            o = self.llm(x, a).last_hidden_state[:,0,:]

        return o
    
class Linear_classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(768, 3)

    def forward(self, x):
        x = self.linear1(x)
        return x
    
class MLP_classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(768, 512)
        self.actv =  nn.ReLU()
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, 3)

    def forward(self, x):

        x = self.linear1(x)
        x = self.actv(x)
        x = self.linear2(x)
        x = self.actv(x)
        x = self.linear3(x)

        return x

if __name__=="__main__":
    global tokenizer

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    config={
        'batch_size': 64,
        'epochs': 10,
        'LLM': "GPT",
        'classifier': "Linear"
    }

    parser = argparse.ArgumentParser(description='Laboratory 2')

    #
    parser.add_argument("--batch_size", type=int, default=128, help='Batch Size')
    parser.add_argument("--epochs", type=int, default=50, help='Number of training epochs')
    parser.add_argument("--llm", type=str, default="GPT2", help='Choose LLM to use; GPT2 or BERT')
    parser.add_argument("--classifier", type=str, default="Linear", help='Choose the classifier to use; Linear or NonLinear')

    args = parser.parse_args()

    #
    config['batch_size'] = args.batch_size
    config['epochs'] = args.epochs
    config['LLM'] = args.llm
    config['classifier'] = args.classifier

    if config['LLM'] ==  "GPT2":
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side="right"
    elif config['LLM'] ==  "BERT":
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        tokenizer.padding_side="right"


    torch.manual_seed(0)

    trainset = tweet_eval("train")
    valset = tweet_eval("validation")
    testset = tweet_eval("test")

    trainloader = DataLoader(trainset, batch_size=config['batch_size'], shuffle=True)
    valloader = DataLoader(valset, batch_size=config['batch_size'])
    testloader = DataLoader(testset, batch_size=config['batch_size'])

    llm = LLM_(config['LLM']).to(device)
    if config['classifier'] == "Linear":
        classifier = Linear_classifier().to(device)
    elif config['classifier'] == "NonLinear":
        classifier = MLP_classifier().to(device)
        
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=3e-4)

    with wandb.init(project="LLM_classification", config = config, mode="online"):
        config = wandb.config
        wandb.watch(classifier, criterion, log="all", log_freq=1)
        llm, classifier = train(llm, classifier, trainloader, valloader, criterion, optimizer, config['epochs'], device)
        wandb.finish()

    test_accuracy, test_loss = evaluate_batch(llm, classifier, testloader, criterion, device)
    print(f"Test Accuracy: {test_accuracy}   Test Loss: {test_loss}")












