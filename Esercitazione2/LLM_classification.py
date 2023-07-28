from transformers import GPT2Model, GPT2Tokenizer, AutoTokenizer, DataCollatorWithPadding, DistilBertModel
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from tqdm import tqdm

from datasets import load_dataset

import argparse

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side="right"

class tweet_eval(Dataset):
    def __init__(self, split):
        
        data = load_dataset("tweet_eval", "sentiment")
        self.split = split
        self.tokenized_data = data.map(preprocess_function, batched=True, batch_size=(len(data["train"])+len(data["validation"])+len(data["test"])))
        
        """for i in self.tokenized_data[self.split][:]["input_ids"]:
            print(self.split,":", len(i))"""

    def __len__(self):
        return len(self.tokenized_data[self.split])

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.tokenized_data[self.split][idx]["input_ids"])
        label = torch.tensor(self.tokenized_data[self.split][idx]["label"])
        attention_mask = torch.tensor(self.tokenized_data[self.split][idx]["attention_mask"])
        return input_ids, label, attention_mask

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, return_tensors="pt")##

def make_loader(set, batch_size, collator=None, shuffle = False):
    loader = torch.utils.data.DataLoader(dataset=set, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)
    return loader

def train(llm, classifier, train_loader, validation_loader, criterion, optimizer, epochs, device):
    for epoch in range(epochs):
        loss_epoch = train_batch(llm, classifier, train_loader, criterion, optimizer, epoch, device)

        print(f"Loss Epoch: {loss_epoch}")

    return model

def train_batch(llm, classifier, train_loader, criterion, optimizer, epoch, device):
    llm.eval()
    classifier.train()
    running_loss = 0
    num_batches = len(train_loader)
    for (ids,  labels, attention) in tqdm(train_loader, desc=f'Training epoch {epoch}', leave=True):
        ids = ids.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            out = llm(ids, attention)

        logits = classifier(out)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()/num_batches #TOCHECK

    return running_loss

class LLM_(nn.Module):
    def __init__(self, llm):
        super().__init__()
        if(llm == "GPT2"):
            self.llm = GPT2Model.from_pretrained('gpt2')
        elif(llm == "BERT"):
            self.llm = DistilBertModel.from_pretrained('distilbert-base-uncased')

    def forward(self, x, a):
        if("GPT2"):
            indices = (a.sum(1) - 1).tolist()
            o = self.llm(x).last_hidden_state#[:,-1,:]
            for i, ind in enumerate(indices):
                if i == 0:
                    d = o[i, ind, :][None, :]
                else:
                    d = torch.cat((d, o[i, ind, :][None, :]))
            o = d
        elif("BERT"):
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
        self.linear1 = nn.Linear(768, 128)
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

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    config={
        'batch_size': 128,
        'epochs': 10,
        'LLM': "GPT",
        'classifier': "Linear"
    }

    parser = argparse.ArgumentParser(description='Laboratory 2')

    #Arguments for config_pipe
    parser.add_argument("--batch_size", type=int, default=128, help='Batch Size')
    parser.add_argument("--epochs", type=int, default=20, help='Number of training epochs')
    parser.add_argument("--LLM", type=str, default="GPT2", help='Choose LLM to use; GPT2 or BERT')
    parser.add_argument("--classifier", type=str, default="Linear", help='Choose the classifier to use; Linear or NonLinear')

    args = parser.parse_args()

    #Populate config_pipe
    config['batch_size'] = args.batch_size
    config['epochs'] = args.epochs
    config['LLM'] = args.LLM
    config['classifier'] = args.classifier


    torch.manual_seed(0)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    trainset = tweet_eval("train")
    valset = tweet_eval("validation")
    testset = tweet_eval("test")

    trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
    valloader = DataLoader(trainset, batch_size=128)
    testloader = DataLoader(trainset, batch_size=128)

    llm = LLM_(config['LLM']).to(device)
    if config['classifier'] == "Linear":
        classifier = Linear_classifier().to(device)
    elif config['classifier'] == "NonLinear":
        classifier = MLP_classifier().to(device)
        
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=3e-4)

    config={
        'batch_size': 128,
        'epochs': 10,
        'LLM': "GPT",
        'classifier': "Linear"
    }



    model = train(llm, classifier, trainloader, valloader, criterion, optimizer, config['epochs'], device)













