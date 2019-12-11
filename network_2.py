#Importing the Libraries
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

torch.set_num_threads(8)
torch.manual_seed(51)


#Path for saving the model
PATH = "model_2.pth"



#Bi-LSTM Network
class BiLSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, use_gpu, batch_size, dropout=0.5):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.dropout = dropout
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True)
        self.hidden2label = nn.Linear(hidden_dim*2, label_size)
        self.hidden = self.init_hidden()


    def init_hidden(self):
        if self.use_gpu:
            return (Variable(torch.randn(2, self.batch_size, self.hidden_dim).cuda()),
                    Variable(torch.randn(2, self.batch_size, self.hidden_dim).cuda()))
        else:
            return (Variable(torch.randn(2, self.batch_size, self.hidden_dim)),
                    Variable(torch.randn(2, self.batch_size, self.hidden_dim)))

    def forward(self, sentence):
        x = self.embeddings(sentence).view(len(sentence), self.batch_size, -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y = self.hidden2label(lstm_out[-1])
        log_probs = F.log_softmax(y,dim = -1)
        return log_probs




#Reading the data
data = pd.read_csv('dataset/train.tsv',delimiter='\t')
data_val = pd.read_csv('dataset/val.tsv',delimiter='\t') 



#Concatenating the data
data['statement'] = data["justification"].str.cat(data['statement'], sep =" ") 
data['statement'] = data["statement"].str.cat(data['job_title'], sep =" ") 
data['statement'] = data["statement"].str.cat(data['party'], sep =" ") 

data_val['statement'] = data_val["justification"].str.cat(data_val['statement'], sep =" ") 
data_val['statement'] = data_val["statement"].str.cat(data_val['job_title'], sep =" ") 
data_val['statement'] = data_val["statement"].str.cat(data_val['party'], sep =" ") 

#Assigning ids to each word
word_to_ix = {}
for sent in data['statement']:
    for word in sent.split():
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

    

label_to_ix = {}
for label in data['label']:
    if label not in label_to_ix:
        label_to_ix[label] = len(label_to_ix)

    	



def prepare_sequence(seq, to_ix):
    idxs = []
    for w in seq:
        try:
            idxs.append(to_ix[w])
        except KeyError:
            continue
    return torch.tensor(idxs, dtype=torch.long)


#Defining the hyperparameters of the model
EPOCHS = 20
USE_GPU = torch.cuda.is_available()
EMBEDDING_DIM = 300
HIDDEN_DIM = 150

BATCH_SIZE = 1
best_dev_acc = 0.0

model = BiLSTM(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, vocab_size=len(word_to_ix), label_size=len(label_to_ix),\
                          use_gpu=USE_GPU, batch_size=BATCH_SIZE)

if USE_GPU:
    model = model.cuda()



optimizer = optim.Adam(model.parameters(), lr=1e-5)
loss_function = nn.NLLLoss()


#Validating model
def val_model(trained_model, statement, label):
    model.eval()
    count_correct = 0
    for sent, label in zip(statement,label):
            sent = prepare_sequence(sent.split(), word_to_ix)
            label = prepare_sequence(label.split(), label_to_ix)
            label = label.cuda()
            sent = sent.cuda()
            pred = model(sent)
            indices = pred[0].argmax()
            if indices == label[0]:
                count_correct += 1
            accuracy = count_correct/len(statement)
    print("Accuracy: %2.2f"%(accuracy*100))
    return accuracy           



#Training Model
print('Training...')

def train_model(model, EPOCHS, statement, labels, loss_function, optimizer):
    model.train()
    highest_accuracy = 0
    best_model = model
    for epoch in range(EPOCHS):
        print("Epoch: %d/%d"%(epoch,EPOCHS-1))
        avg_loss = 0.0
        count = 0
        for sent, label in zip(statement,labels):
            sent = prepare_sequence(sent.split(), word_to_ix)
            label = prepare_sequence(label.split(), label_to_ix)
            label = label.cuda()
            sent = sent.cuda()
            model.hidden = model.init_hidden()
            pred = model(sent)
            model.zero_grad()
            loss = loss_function(pred, label)
            avg_loss += loss
            loss.backward()
            optimizer.step()
            count += 1
        print("Running Loss: %2.4f"%(avg_loss/count))
        accuracy = val_model(model, data_val['statement'], data_val['label'])
        model.train()
        if accuracy > highest_accuracy:
            highest_accuracy = accuracy
            best_model = model
            print("Saving Model...")
            torch.save(best_model.state_dict(), PATH)
    print("Highest Validation Accuracy %2.2f"%(highest_accuracy*100))
    return best_model


model = train_model(model, EPOCHS, data['statement'], data['label'], loss_function, optimizer)


model.load_state_dict(torch.load(PATH))


val_model(model, data_val['statement'], data_val['label'])


