# -*- coding: utf-8 -*-
"""

Author: Qitong Hu, Shanghai Jiao Tong University

"""

import os
import torch
import torch.nn as nn

from transformers import BertModel
from torchcrf import CRF

class ExtractionModel(nn.Module):
    """
    Information extraction AI model.
    
    Args:
        option: The parameters of main code.
    """
    def __init__(self, option):
        super(ExtractionModel, self).__init__()
        self.tag2id = option.tag2id
        self.tag_size = len(self.tag2id)
        
        self.bert = BertModel.from_pretrained(os.path.join(option.modelpath, option.bertmodel))
        self.lstm = nn.LSTM(input_size = option.bert_size, hidden_size=option.filter_size // 2,
                            bidirectional=True, batch_first = True)
        
        self.dropout = nn.Dropout(option.dropout)
        self.linear = nn.Linear(option.filter_size, self.tag_size)
        self.crf = CRF(self.tag_size, batch_first = True)
        
    def forward(self, sentence, tag, mask):
        with torch.no_grad():
            embeddings = self.bert(sentence)
        encoder_output, _ = self.lstm(embeddings[0])
        encoder_output = self.dropout(encoder_output)
        emissions = self.linear(encoder_output)
        if (tag != None):
            loss = -self.crf.forward(emissions, tag, mask).mean()
            return loss
        else:
            decoder_output = self.crf.decode(emissions, mask)
            return decoder_output
