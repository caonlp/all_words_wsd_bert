import os
import codecs
from keras.preprocessing.sequence import pad_sequences
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.functional import F
from torch.autograd import Variable
from torch.utils.data.sampler import RandomSampler, SequentialSampler
import pickle


embed = {}
f = open('nwjc2vec-norm1-bin2.pkl', 'rb')
embed = pickle.load(f)
f.close()

train_x = {}
train_y = {}

test_x = {}
test_y = {}


f = codecs.open('train.dat', 'r', encoding = "utf-8")
x_l = f.readline() 
# print(x_l)

y_l = f.readline() 
# print(y_l)

g = codecs.open('test.dat', 'r', encoding = "utf-8")
x_o = g.readline() 
# print(x_o)

y_o = g.readline()
# print(y_o)

tn = 0
rn = 0

while (x_l and y_l):
    x_l = x_l.rstrip()
    y_l = y_l.rstrip()
    train_x[tn] = x_l.split()
    train_y[tn] = y_l.split()
    tn += 1
    x_l = f.readline()
    y_l = f.readline()
f.close()

while (x_o and y_o):
    x_o = x_o.rstrip()
    y_o = y_o.rstrip()
    test_x[rn] = x_o.split()
    test_y[rn] = y_o.split()
    rn += 1
    x_o = g.readline()
    y_o = g.readline()
g.close()


print("train_x : ", train_x)
print("train_y : ", train_y)
print("test_x : ", test_x)
print("test_y : ", test_y)

def myembed(x_l):
    allemb = []
    ln = len(x_l)
    for i in range(len(x_l)):
        allemb = np.r_(allemb)

def myembedcore(x):
    if (x in embed):
        return  np.array(embed[x], dtype = np.float32)
    else:
        z = np.zeros(demb, dtype = np.float32) + 0.005 



class_size = 919
max_epoch = 10
batch_size = 10
train_size = train_x.__len__()

print("train_size is~", train_size)

n_batch = train_size // batch_size

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        self.embedding = nn.Embedding()
        self.layer = nn.Sequential(nn.Linear(input_dim, output_dim))

    def forward(self, x):
        x = self.layer(x)
        return x



class LSTMClassifier(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights):
        super(LSTMClassifier, self).__init__()
        """
        arguments~:
        
        batch_size: Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        output_size : class_size ~ 919
        hidden_size : Size of the hidden_state of the LSTM
        vocab_size: Size of the vocabulary containing unique words
        embedidng_length: Embedding dimension of GloVe word embeddings
        weights: Pre-training Glove word_embeddings which we will use to create our word_embedding look-up table
        """
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embeddding_lenth = embedding_length

        self.word_embeddings = nn.Embedding(vocab_size, embedding_length) # Initializing the look-up table
        self.word_embeddings.weight = nn.Parameter(weights, requires_grad = False) # Assiging the table to the pre-trained GloVe word embedding
        self.lstm = nn.LSTM(embedding_length, hidden_size)
        self.label = nn.Linear(hidden_size, output_size)

    def forward(self, input_sentence, batch_size = None):
        """
        :param input_sentence: input_sentence of shape = (batch_size, num_sequences)
        :param batch_size: default = None. Used only for prediction on a single sentence after training (batch_size = 1)
        :return:
        """



def train():
    model = Model(200, class_size)
    optmizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)
    cost = nn.CrossEntropyLoss()

    # train
    for epoch in range(max_epoch):
        running_loss = 0.0
        for step, inputs, labels in enumerate(train_loader, 0):
            inputs, labels = Variable(inputs), Variable(labels)
            inputs, labels = inputs.float(), labels.long()
            optmizer.zero_grad()
            outputs = model(inputs)
            loss = cost(outputs, labels)
            loss.backward()
            optmizer.step()
            running_loss += loss.item()

            if i % n_batch == n_batch - 1:
                print("[%d %5d] loss: % .3f" % (epoch + 1, step + 1, running_loss / n_batch))
                running_loss == 0.0
    print("Training　Finished~")
    torch.save(model, 'all_words_wsd_model.pkl')
    torch.save(model.state_dict(), 'all_words_wsd_model_params.pkl')

def reload_model():
    train_model = torch.load('all_words_wsd_model.pkl')
    return train_model

def test():
    test_loss = 0
    correct = 0
    model = reload_model()
    for test_data, test_target in test_loader:
        test_data, test_target = test_data.float(), test_target.long()
        test_data, test_target = Variable(test_data), Variable(test_target)
        outputs = model(test_data)

        test_loss += F.nll_loss(outputs, test_target, reduction = 'sum').item()
        pred = outputs.data.max(1, keepdim = True)[1]
        correct += pred.eq(test_target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
    print('Test Accuracy~'.format(100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    train()
    reload_model()
    test()



















