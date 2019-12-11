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
        self.hidden2label = nn.Linear(hidden_dim*4, label_size)
        self.hidden_1 = self.init_hidden_1()

        self.lstm_2 = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True)
        self.hidden_2 = self.init_hidden_2()


    def init_hidden_1(self):
        # first is the hidden h
        # second is the cell c
        if self.use_gpu:
            return (Variable(torch.randn(2, self.batch_size, self.hidden_dim).cuda()),
                    Variable(torch.randn(2, self.batch_size, self.hidden_dim).cuda()))
        else:
            return (Variable(torch.randn(2, self.batch_size, self.hidden_dim)),
                    Variable(torch.randn(2, self.batch_size, self.hidden_dim)))



    def init_hidden_2(self):
        # first is the hidden h
        # second is the cell c
        if self.use_gpu:
            return (Variable(torch.randn(2, self.batch_size, self.hidden_dim).cuda()),
                    Variable(torch.randn(2, self.batch_size, self.hidden_dim).cuda()))
        else:
            return (Variable(torch.randn(2, self.batch_size, self.hidden_dim)),
                    Variable(torch.randn(2, self.batch_size, self.hidden_dim)))

    def forward(self, sentence_1, sentence_2):
        #x_1 = self.embeddings(sentence_1).view(len(sentence_1), self.batch_size, -1)
        lstm_out_1, self.hidden_1 = self.lstm_1(sentence_1, self.hidden_1)
        #x_2 = self.embeddings(sentence_2).view(len(sentence_2), self.batch_size, -1)
        lstm_out_2, self.hidden_2 = self.lstm_2(sentence_2, self.hidden_2)
        temp = torch.cat((lstm_out_1[-1],lstm_out_2[-1]),1)
        y = self.hidden2label(temp)
        log_probs = F.log_softmax(y,dim = -1)
        return log_probs



data = pd.read_csv('dataset/train.tsv',delimiter='\t')
data_val = pd.read_csv('dataset/val.tsv',delimiter='\t')


statement_array = np.load("statement.npy")
justification_array = np.load("justification.npy")
print("shape",len(statement_array[0]))


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
    	



def prepare_sequence(seq, to_ix):
    
    #idxs = [to_ix[w] for w in seq]
    idxs = []
    for w in seq:
        try:
            idxs.append(to_ix[w])
        except KeyError:
            continue
    return torch.tensor(idxs, dtype=torch.long)



EPOCHS = 10
USE_GPU = torch.cuda.is_available()
EMBEDDING_DIM = 100
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
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_function = nn.NLLLoss()


def val_model(trained_model, statement, label,justification):
    trained_model.eval()
    count_correct = 0
    for sent, label, just in zip(statement,label,justification):
            #sent = prepare_sequence(sent.split(), word_to_ix)
            sent = torch.unsqueeze(torch.FloatTensor(sent),1)
            label = prepare_sequence(label.split(), label_to_ix)
            #just = prepare_sequence(just.split(), word_to_ix)
            just = torch.unsqueeze(torch.FloatTensor(just),1)
            label = label.cuda()
            sent = sent.cuda()
            just = just.cuda()
            pred = model(sent,just)
            indices = pred[0].argmax()
            if indices == label[0]:
                count_correct += 1
            accuracy = count_correct/len(statement)
    print("Accuracy: ",accuracy) 
    return accuracy           




print('Training...')

def train_model(model, EPOCHS, statement, labels, justification, loss_function, optimizer):
    model.train()
    highest_accuracy = 0
    best_model = model
    for epoch in range(EPOCHS):
        print(epoch)
        avg_loss = 0.0
        count = 0
        for sent, label, just in zip(statement,labels,justification):
            #sent = prepare_sequence(sent.split(), word_to_ix)
            sent = torch.FloatTensor(sent)
            sent = torch.unsqueeze(torch.FloatTensor(sent),1)
            label = prepare_sequence(label.split(), label_to_ix)
            #just = prepare_sequence(just.split(), word_to_ix)
            just = torch.unsqueeze(torch.FloatTensor(just),1)
            label = label.cuda()
            sent = sent.cuda()
            just = just.cuda()
            model.hidden_1 = model.init_hidden_1()
            model.hidden_2 = model.init_hidden_2()
            pred = model(sent,just)
            model.zero_grad()
            loss = loss_function(pred, label)
            avg_loss += loss
            loss.backward()
            optimizer.step()
            count += 1
        print(avg_loss/count)
        accuracy = val_model(model, justification_array, data_val['label'],justification)
        model.train()
        if accuracy > highest_accuracy:
            highest_accuracy = accuracy
            best_model = model
    return best_model


model = train_model(model, EPOCHS, statement_array, data_val['label'], justification_array, loss_function, optimizer)
torch.save(model.state_dict(), PATH)

model.load_state_dict(torch.load(PATH))


val_model(model, statement_array, data_val['label'], justification_array)


