import os
import codecs
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from tqdm import trange

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn.functional as F
import random
import json
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from transformers import BertJapaneseTokenizer, BertConfig, AdamW, get_linear_schedule_with_warmup
from model import AllWSD

from seqeval.metrics import classification_report


train_x = []
train_y = []

test_x = []
test_y = []


# 定义句子的最大长度
MAX_LEN = 128

f = codecs.open('train.dat', 'r', encoding = "utf-8")
x_l = f.readline()
y_l = f.readline() 

g = codecs.open('test.dat', 'r', encoding = "utf-8")
x_o = g.readline() 

y_o = g.readline() 
tn = 0
rn = 0

while (x_l and y_l):
    x_l = x_l.rstrip()
    y_l = y_l.rstrip()
    x_l = "[CLS]" + " " + x_l + " " + "[SEP]"
    train_x.append(x_l.split())
    train_y.append(y_l.split())
    x_l = f.readline()
    y_l = f.readline()
f.close()

while (x_o and y_o):
    x_o = x_o.rstrip()
    y_o = y_o.rstrip()
    x_o = "[CLS]" + " " + x_o + " " + "[SEP]"
    test_x.append(x_o.split())
    test_y.append(y_o.split())
    x_o = g.readline()
    y_o = g.readline()
g.close()


tokenizer = BertJapaneseTokenizer.from_pretrained("bert-base-japanese-whole-word-masking")

train_input_ids = [tokenizer.convert_tokens_to_ids(sent) for sent in train_x]
test_input_ids = [tokenizer.convert_tokens_to_ids(sent) for sent in test_x]



train_input_ids = pad_sequences(train_input_ids, maxlen = MAX_LEN, dtype = "long", truncating = "post", padding = "post")
test_input_ids = pad_sequences(test_input_ids, maxlen = MAX_LEN, dtype = "long", truncating = "post", padding = "post")
train_input_labels = pad_sequences(train_y, maxlen = MAX_LEN, dtype = "long", truncating = "post", padding = "post")
test_input_labels = pad_sequences(test_y, maxlen = MAX_LEN, dtype = "long", truncating = "post", padding = "post")

train_segment_ids = []
test_segment_ids = []

""" mask """

train_attention_mask = []
test_attention_mask = []

for seq in train_input_ids:
    train_seq_mask = [float(i > 0) for i in seq]
    train_attention_mask.append(train_seq_mask)

for seq in test_input_ids:
    test_seq_mask = [float(i > 0) for i in seq]
    test_attention_mask.append(test_seq_mask)


train_segment_ids = np.zeros_like(train_attention_mask)
test_segment_ids = np.zeros_like(test_attention_mask)

MAX_SEQ_LENGTH = 128
MAX_EPOCH = 10
TRAIN_BATCH_SIZE = 10
TEST_BATCH_SIZE = 8
LR = 2e-5
num_labels = 919

warmup_proportion = 0.1
weight_decay = 0.01
adam_epsilon = 1e-8
max_grad_norm = 1.0
gradient_accumulation_steps = 1
local_rank = -1
seed = 42
loss_scale = 0.

if gradient_accumulation_steps < 1:
    raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(gradient_accumulation_steps))

TRAIN_BATCH_SIZE = TRAIN_BATCH_SIZE // gradient_accumulation_steps

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)



train_inputs = torch.tensor(train_input_ids, dtype=torch.long)
train_segment = torch.tensor(train_segment_ids, dtype=torch.long)
train_masks = torch.tensor(train_attention_mask, dtype=torch.long)
train_labels = torch.tensor(train_input_labels, dtype=torch.long)

test_inputs = torch.tensor(test_input_ids, dtype=torch.long)
test_segment = torch.tensor(test_segment_ids, dtype=torch.long)
test_masks = torch.tensor(test_attention_mask, dtype=torch.long)
test_labels = torch.tensor(test_input_labels, dtype=torch.long)

num_train_optimization_steps = int(len(train_inputs) / TRAIN_BATCH_SIZE / gradient_accumulation_steps) * MAX_EPOCH



train_data = TensorDataset(train_inputs, train_segment ,train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_loader = DataLoader(train_data, sampler = train_sampler, batch_size = TRAIN_BATCH_SIZE)

test_data = TensorDataset(test_inputs, test_segment, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_loader = DataLoader(test_data, sampler = test_sampler, batch_size = TEST_BATCH_SIZE)


model_config = BertConfig.from_pretrained("bert-base-japanese-whole-word-masking",num_labels = num_labels)
model = AllWSD.from_pretrained("bert-base-japanese-whole-word-masking", from_tf = False, config = model_config)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]


warmup_steps = int(warmup_proportion * num_train_optimization_steps)
optimizer = AdamW(optimizer_grouped_parameters, lr = LR, eps = adam_epsilon)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = warmup_steps, num_training_steps = num_train_optimization_steps)

global_step = 0
nb_tr_steps = 0
tr_loss = 0

def train():
    model.train()
    for epoch in trange(MAX_EPOCH):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, (train_inputs_ids, train_segment_ids, train_masks_ids, train_labels_ids) in enumerate(train_loader):
            optimizer.zero_grad()

            loss = model(train_inputs_ids, train_segment_ids, train_masks_ids, train_labels_ids)


            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            tr_loss += loss.item()
            nb_tr_examples += train_inputs.size(0)

            optimizer.step()
            scheduler.step()
            nb_tr_steps += 1

            print("[Epoch : %d batch : %d] loss: %.3f" % (epoch + 1, step + 1, tr_loss / nb_tr_steps))

    print("training completed~")
    # Save a trained model and the associated configuration
    model_to_save = model.module if hasattr(model, 'module') else model # Only save the model it-self
    model_to_save.save_pretrained("")
    tokenizer.save_pretrained("")
    model_config = {"bert_model": "bert-base-uncased", "do_lower": False,
                    "max_seq_length": MAX_SEQ_LENGTH}
    json.dump(model_config, open("model_config.json", "w"))
    torch.save(model, "all_wsd_model.pkl")


def reload_mdoel():
    train_model = torch.load('all_wsd_model.pkl')
    return train_model



"""
def test():

    correct = 0
    model = reload_mdoel()

    for test_inputs_ids, test_segment_ids, test_masks_ids, test_labels_ids in test_loader:
        with torch.no_grad():

            outputs = model(test_inputs_ids, test_segment_ids, test_masks_ids)

        predict = torch.argmax(F.log_softmax(outputs, dim = 2), dim = 2)

        print("predict = ", predict.detach().numpy())
        print("label = ", test_labels_ids.detach().numpy())

        correct += predict.eq(test_labels_ids.data.view_as(predict)).cpu().sum()

    print("Test Accuracy: [{:1f}%]\n".format(100. * correct / (len(test_loader.dataset)) * MAX_SEQ_LENGTH))
"""

def test():

    y_true = []
    y_pred = []


    label_key = np.array(range(0, 919))
    label_value = np.array(range(0, 919))
    label_map = dict(zip(label_key,label_value))

    model = reload_mdoel()

    for test_inputs_ids, test_segment_ids, test_masks_ids, test_labels_ids in test_loader:
        with torch.no_grad():
            outputs = model(test_inputs_ids, test_segment_ids, test_masks_ids)
            

        predict = torch.argmax(F.log_softmax(outputs, dim = 2), dim = 2)
        predict = predict.detach().numpy()

        test_labels_ids = test_labels_ids.detach().numpy()



        for i, label in enumerate(test_labels_ids):
            temp_1 = []
            temp_2 = []
            # print("label", label.shape)
            for j, m in enumerate(label):
                # print(test_labels_ids[i][j])
                if j == 0:
                    # print("j = 0 了")
                    continue
                elif test_labels_ids[i][j] == len(label_map):
                    print(len(test_labels_ids[i][j]))

                    y_true.append(temp_1)
                    y_pred.append(temp_2)
                    print("y_true", y_true)
                    print("y_pred", y_pred)
                    break
                else:
                    temp_1.append(label_map[test_labels_ids[i][j]])
                    temp_2.append(label_map[predict[i][j]])

    print(y_pred)
    print(y_true)

    report = classification_report(y_true, y_pred, digits = 4)
    print(report)
    output_test_file = os.path.join('', "test_result.txt")
    with open(output_test_file, "w") as writer:
        writer.write(report)



if __name__ == '__main__':
    # train()
    reload_mdoel()
    test()









