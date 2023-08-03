# -*- coding: utf-8 -*-
"""

Author: Qitong Hu, Shanghai Jiao Tong University

"""

#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 3
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor,
#  Boston, MA  02110-1301, USA.

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
