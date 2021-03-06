import torch
from transformers import BertForTokenClassification
import torch.nn as nn
import numpy as np


class AllWSD(BertForTokenClassification):

    def forward(self, input_ids, token_type_ids = None, attention_mask = None, labels = None, valid_ids = None, attention_mask_label = None):
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask, head_mask = None)[0]
        batch_size, max_len, feat_dim = sequence_output.shape
        tagger_input = sequence_output
        tagger_input = self.dropout(tagger_input)
        logits  = self.classifier(tagger_input)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()

            if attention_mask_label is not None:
                active_loss = attention_mask_label.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

        pass


































