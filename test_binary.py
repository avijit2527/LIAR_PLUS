import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os
import numpy as np
import pandas as pd

torch.set_num_threads(8)
torch.manual_seed(1)



PATH_1 = "model_binary_1.pth"
PATH_2 = "model_binary_2.pth"
PATH_3 = "model_binary_3.pth"


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
        #y = F.relu(self.linear(y))
        log_probs = F.log_softmax(y,dim = -1)
        return log_probs




data = pd.read_csv('dataset/train_binary.tsv',delimiter='\t')
data_val = pd.read_csv('dataset/val_binary.tsv',delimiter='\t')
data_test = pd.read_csv('dataset/test_binary.tsv',delimiter='\t')


data['statement'] = data["justification"].str.cat(data['statement'], sep =" ") 
data['statement'] = data["statement"].str.cat(data['job_title'], sep =" ") 
data['statement'] = data["statement"].str.cat(data['party'], sep =" ") 

data_test['statement'] = data_test["justification"].str.cat(data_test['statement'], sep =" ") 
data_test['statement'] = data_test["statement"].str.cat(data_test['job_title'], sep =" ") 
data_test['statement'] = data_test["statement"].str.cat(data_test['party'], sep =" ") 

data_val['statement'] = data_val["justification"].str.cat(data_val['statement'], sep =" ") 
data_val['statement'] = data_val["statement"].str.cat(data_val['job_title'], sep =" ") 
data_val['statement'] = data_val["statement"].str.cat(data_val['party'], sep =" ") 


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




USE_GPU = torch.cuda.is_available()
EMBEDDING_DIM = 300
HIDDEN_DIM = 150

BATCH_SIZE = 1


model_1 = BiLSTM(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, vocab_size=len(word_to_ix), label_size=len(label_to_ix),\
                          use_gpu=USE_GPU, batch_size=BATCH_SIZE)
model_2 = BiLSTM(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, vocab_size=len(word_to_ix), label_size=len(label_to_ix),\
                          use_gpu=USE_GPU, batch_size=BATCH_SIZE)
model_3 = BiLSTM(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, vocab_size=len(word_to_ix), label_size=len(label_to_ix),\
                          use_gpu=USE_GPU, batch_size=BATCH_SIZE)



model_1.load_state_dict(torch.load(PATH_1))
model_2.load_state_dict(torch.load(PATH_2))
model_3.load_state_dict(torch.load(PATH_3))

if USE_GPU:
    model_1 = model_1.cuda()
    model_2 = model_2.cuda()
    model_3 = model_3.cuda()





def test_model(trained_model_1,trained_model_2,trained_model_3, statement, labels):
    trained_model_1.eval()
    trained_model_2.eval()
    trained_model_3.eval()
    count_correct = 0
    for sent , label in zip(statement,labels):
            sent = prepare_sequence(sent.split(), word_to_ix)
            label = prepare_sequence(label.split(), label_to_ix)
            sent = sent.cuda()
            label = label.cuda()
            pred_1 = trained_model_1(sent)
            pred_2 = trained_model_2(sent)
            pred_3 = trained_model_3(sent)
            pred = pred_1 + pred_2 + pred_3
            indices = pred[0].argmax()
            if indices == label[0]:
                count_correct += 1
            accuracy = count_correct/len(statement)
    return accuracy           







Test_Accuracy = test_model(model_1, model_2,model_3, data_test['statement'],data_test['label'])
Val_Accuracy = test_model(model_1, model_2,model_3, data_val['statement'],data_val['label'])

print("Test Accuracy:       %2.2f percent"%(Test_Accuracy*100))
print("Validation Accuracy: %2.2f percent"%( Val_Accuracy*100))
