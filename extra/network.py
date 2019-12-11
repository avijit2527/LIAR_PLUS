import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import time, random
import os
from tqdm import tqdm
from torchtext import data
import numpy as np
import argparse
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

torch.set_num_threads(8)
torch.manual_seed(100)
random.seed(100)


PATH = "model_sj.pth"


class BiLSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, use_gpu, batch_size, dropout=0.5):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.dropout = dropout
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm_1 = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True)
        self.hidden2label = nn.Linear((hidden_dim*4)+1, label_size)
        self.hidden_1 = self.init_hidden_1()

        self.lstm_2 = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True)
        self.hidden_2 = self.init_hidden_2()


    def init_hidden_1(self):
        if self.use_gpu:
            return (Variable(torch.randn(2, self.batch_size, self.hidden_dim).cuda()),
                    Variable(torch.randn(2, self.batch_size, self.hidden_dim).cuda()))
        else:
            return (Variable(torch.randn(2, self.batch_size, self.hidden_dim)),
                    Variable(torch.randn(2, self.batch_size, self.hidden_dim)))



    def init_hidden_2(self):
        if self.use_gpu:
            return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()),
                    Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()))
        else:
            return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim)),
                    Variable(torch.zeros(2, self.batch_size, self.hidden_dim)))

    def forward(self, sentence_1, sentence_2, party):
        x_1 = self.embeddings(sentence_1).view(len(sentence_1), self.batch_size, -1)
        lstm_out_1, self.hidden_1 = self.lstm_1(x_1, self.hidden_1)
        x_2 = self.embeddings(sentence_2).view(len(sentence_2), self.batch_size, -1)
        lstm_out_2, self.hidden_2 = self.lstm_2(x_2, self.hidden_2)
        temp = torch.cat((lstm_out_1[-1],lstm_out_2[-1]),1)
        party = torch.unsqueeze(party,0)
        temp = torch.cat((temp,party),1)
        y = self.hidden2label(temp)
        log_probs = F.log_softmax(y,dim = -1)
        return log_probs



data = pd.read_csv('dataset/train.tsv',delimiter='\t')
data_val = pd.read_csv('dataset/val.tsv',delimiter='\t')


word_to_ix = {}

for sent in data['statement']:
    for word in sent.split():
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)



for sent in data['justification']:
    for word in sent.split():
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print(len(word_to_ix))


label_to_ix = {}

for label in data['label']:
    if label not in label_to_ix:
        label_to_ix[label] = len(label_to_ix)
print(len(label_to_ix))
    	


party_to_ix = {}

for party in data['party']:
    if party not in party_to_ix:
        party_to_ix[party] = len(party_to_ix)
print("party",len(party_to_ix))
    	

job_title_to_ix = {}

for job_title in data['job_title']:
    if job_title not in job_title_to_ix:
        job_title_to_ix[job_title] = len(job_title_to_ix)
print("job_title",len(job_title_to_ix))
    	



def prepare_sequence(seq, to_ix):
    idxs = []
    for w in seq:
        try:
            idxs.append(to_ix[w])
        except KeyError:
            continue
    return torch.tensor(idxs, dtype=torch.long)



EPOCHS = 50
USE_GPU = torch.cuda.is_available()
EMBEDDING_DIM = 300
HIDDEN_DIM = 150

BATCH_SIZE = 1
timestamp = str(int(time.time()))
best_dev_acc = 0.0

model = BiLSTM(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, vocab_size=len(word_to_ix), label_size=len(label_to_ix),\
                          use_gpu=USE_GPU, batch_size=BATCH_SIZE)

if USE_GPU:
    model = model.cuda()



def get_accuracy(truth, pred):
    assert len(truth) == len(pred)
    right = 0
    for i in range(len(truth)):
        if truth[i] == pred[i]:
            right += 1.0
    return right / len(truth)



best_model = model
optimizer = optim.Adam(model.parameters(), lr=2e-5)
loss_function = nn.NLLLoss()


def val_model(trained_model, statement, label,justification,parties):
    trained_model.eval()
    count_correct = 0
    for sent, label, just, party in zip(statement,label,justification,parties):
            sent = prepare_sequence(sent.split(), word_to_ix)
            label = prepare_sequence(label.split(), label_to_ix)
            just = prepare_sequence(just.split(), word_to_ix)
            party = prepare_sequence(party.split('$'), party_to_ix)
            party = party.type(torch.FloatTensor)
            label = label.cuda()
            sent = sent.cuda()
            just = just.cuda()
            party = party.cuda()
            pred = model(sent,just,party)
            indices = pred[0].argmax()
            if indices == label[0]:
                count_correct += 1
            accuracy = count_correct/len(statement)
    print("Accuracy: ",accuracy) 
    return accuracy           




print('Training...')

def train_model(model, EPOCHS, statement, labels, justification, parties, loss_function, optimizer):
    model.train()
    highest_accuracy = 0
    best_model = model
    for epoch in range(EPOCHS):
        print(epoch)
        avg_loss = 0.0
        count = 0
        for sent, label, just, party in zip(statement,labels,justification, parties):
            #print(party)
            sent = prepare_sequence(sent.split(), word_to_ix)
            label = prepare_sequence(label.split(), label_to_ix)
            just = prepare_sequence(just.split(), word_to_ix)
            party = prepare_sequence(party.split('$'), party_to_ix)
            party = party.type(torch.FloatTensor)
            label = label.cuda()
            sent = sent.cuda()
            just = just.cuda()
            party = party.cuda()
            model.hidden_1 = model.init_hidden_1()
            model.hidden_2 = model.init_hidden_2()
            pred = model(sent,just,party)
            model.zero_grad()
            loss = loss_function(pred, label)
            avg_loss += loss
            loss.backward()
            optimizer.step()
            count += 1
        print(avg_loss/count)
        accuracy = val_model(model, data_val['statement'], data_val['label'],justification, parties)
        model.train() 
        if accuracy > highest_accuracy:
            highest_accuracy = accuracy
            best_model = model
    print("Highest_Accuracy:",highest_accuracy)
    return best_model


model = train_model(model, EPOCHS, data['statement'], data['label'],data['justification'],data['party'], loss_function, optimizer)
torch.save(model.state_dict(), PATH)

model.load_state_dict(torch.load(PATH))


val_model(model, data_val['statement'], data_val['label'],data_val['justification'],data['party'])


