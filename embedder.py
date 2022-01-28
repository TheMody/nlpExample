from transformers import BertTokenizer, BertModel, ElectraTokenizer, ElectraModel
import torch
import torch.nn as nn
import torch.nn.functional as F

models =['bert-base-uncased',
         'google/electra-small-discriminator'
    ]

class NLP_embedder(nn.Module):

    def __init__(self, hyperparameters, num_classes, batch_size):
        super(NLP_embedder, self).__init__()
        self.batch_size = batch_size
        self.hyperparameters = hyperparameters
        if hyperparameters["model_name"] == models[0]:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertModel.from_pretrained('bert-base-uncased')
            output_length = 768
        elif hyperparameters["model_name"] == models[1]:
            self.tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
            self.model = ElectraModel.from_pretrained('google/electra-small-discriminator')
            output_length = 256
        else:
            print("model not supported", hyperparameters["model_name"])
        self.fc1 = nn.Linear(output_length, hyperparameters["hidden_fc"])
        self.fc2 = nn.Linear(hyperparameters["hidden_fc"], num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=hyperparameters["lr"]) # higher lrs than 1e-5 result in no change in loss
        self.softmax =  torch.nn.Softmax(dim = 1)

    def forward(self,x):
         x = self.model(**x)
         #x = torch.mean(x.last_hidden_state, dim = 1)
         x = x.last_hidden_state[:,0]
         x = F.relu(self.fc1(x))
         x = F.relu(self.fc2(x))
         x = self.softmax(x)
         return x
    
    def embed(self,x):
    
        resultx = None

        for i in range(int(len(x)/self.batch_size)):
            batch_x = x[i*self.batch_size: (i+1)*self.batch_size]
            batch_x = self.tokenizer(batch_x, return_tensors="pt", padding = True)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            batch_x = batch_x.to(device)
     
            batch_x = self.model(**batch_x)
            #x = torch.mean(x.last_hidden_state, dim = 1)
            batch_x = batch_x.last_hidden_state[:,0]
            
            if i % int((len(x)/self.batch_size)*0.1) == 0:
                print(i, "of", int(len(x)/self.batch_size))
            
            if resultx == None:
                resultx = batch_x.detach()
            else:
                resultx = torch.cat((resultx,batch_x.detach()))
    
        return resultx
     
    def classify(self,X):
         x = F.relu(self.fc1(x))
         x = F.relu(self.fc2(x))
         x = self.softmax(x)
         return x
     
    def fit(self,x,y, epochs = 1):

        for e in range(epochs):
            for i in range(int(len(x)/self.batch_size)):
                batch_x = x[i*self.batch_size: (i+1)*self.batch_size]
                batch_y = y[i*self.batch_size: (i+1)*self.batch_size]
                batch_x = self.tokenizer(batch_x, return_tensors="pt", padding = True)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                batch_y = batch_y.to(device)
                batch_x = batch_x.to(device)
                y_pred = self(batch_x)
                loss = self.criterion(y_pred,batch_y)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if i % int((len(x)/self.batch_size)*0.1) == 0:
                    print(i, loss.item())
                   # print(y_pred, batch_y)
            print(e, loss.item())
        
        return 
    
    def predict(self,x):
        resultx = None

        for i in range(int(len(x)/self.batch_size)):
            batch_x = x[i*self.batch_size: (i+1)*self.batch_size]
            batch_x = self.tokenizer(batch_x, return_tensors="pt", padding = True)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            batch_x = batch_x.to(device)
     
            batch_x = self(batch_x)
            
            if i % int((len(x)/self.batch_size)*0.1) == 0:
                print(i, "of", int(len(x)/self.batch_size))
            
            if resultx == None:
                resultx = batch_x.detach()
            else:
                resultx = torch.cat((resultx,batch_x.detach()))
    
        return resultx
