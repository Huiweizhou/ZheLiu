#!/usr/bin/python
# -*- coding: UTF-8 -*-
from torch import nn
import torch
from torch import autograd
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

np.random.seed(1337)
torch.manual_seed(1337)
torch.cuda.manual_seed(1337)

class cnn_lstm_no_pad_model(nn.Module):
    def __init__(self,embedding_matrix,word_dim,char_len,char_dim,feature_maps,kernels,hidden_dim,tagset_size):
        super(cnn_lstm_no_pad_model, self).__init__()
        self.model_name = 'cnn_lstm_no_pad_liner'
        self.feature_maps=feature_maps
        self.kernels=kernels
        self.tagset_size = tagset_size
        self.char_dim=char_dim

        self.dropout_tain=nn.Dropout(p=0.5)
        self.dropout_test=nn.Dropout(p=0)

        self.word_emb=nn.Embedding(len(embedding_matrix),word_dim).cuda()
        self.word_emb.weight = nn.Parameter(torch.FloatTensor(embedding_matrix).cuda()) 

        self.char_emb=nn.Embedding(char_len,char_dim,padding_idx=0).cuda()

        self.conv1 = nn.Conv3d(in_channels=1, out_channels=self.feature_maps[0], kernel_size=(1, self.kernels[0], char_dim)).cuda()

        self.lstm = nn.LSTM(word_dim+feature_maps[0], hidden_dim, dropout=0.5, bidirectional=True,num_layers=1).cuda()
        
        self.liner1 = nn.Linear(hidden_dim * 2, hidden_dim).cuda()
        self.liner2 = nn.Linear(hidden_dim , self.tagset_size,bias=False).cuda()


    def forward(self, word,char,is_train):
        if is_train:
            dropout=self.dropout_tain
        else:
            dropout=self.dropout_test
        ###########emb#############    
        batch_size=1
        word=self.word_emb(word)
        char=self.char_emb(char)
        #############CNN############
        sent_len=char.size(0)
        word_len=char.size(1)
        char = char.unsqueeze(0).unsqueeze(1)
        char_1=self.conv1(char)
        split_char_1=[ele.squeeze(4).squeeze(2) for ele in char_1.split(1,dim=2)]#输出 sent_len*（batch_size,feature_map,word_len-kernel+1)
        char_1=torch.stack(split_char_1,dim=1)#输出（batch_size,sent_len,feature_map,word_len-kernel+1)
        char_1=F.relu(char_1)
        chars, _=char_1.max(dim=-1)
        ##########LSTM#############
        inputs=torch.cat((word,chars.squeeze(0)),dim=-1)
        inputs=dropout(inputs)
        inputs = inputs.unsqueeze(1)
        hidden, _ = self.lstm(inputs)
        hidden = torch.transpose(hidden, 1, 0)
        output = self.liner1(hidden)
        # output=F.relu(output)
        output=F.tanh(output)        
        output=dropout(output)
        output=self.liner2(output)
        return output

class sequence_label_model(nn.Module):
    def __init__(self,hidden_dim,tagset_size):
        super(sequence_label_model, self).__init__()
        self.model_name = 'sequence_label_model'
        self.liner = nn.Linear(hidden_dim , tagset_size).cuda()
        self.dropout_tain=nn.Dropout(p=0.5)
        self.dropout_test=nn.Dropout(p=0)

    def forward(self, hidden,is_train):
        if is_train:
            dropout=self.dropout_tain
        else:
            dropout=self.dropout_test
        output=dropout(hidden)
        output=self.liner(output)
        return output
