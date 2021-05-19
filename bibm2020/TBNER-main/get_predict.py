#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pickle as pkl
from model_ner import wipe_view_single_cnn_lstm_no_pad_model
import numpy as np
import torch
from torchcrf import CRF
from torch.autograd import Variable
import torch.optim as optim
from utils import sent_prf_cal
from tqdm import tqdm
import os

class InputTrainFeatures(object):
    """A single set of features of data."""
    def __init__(self, token,char,lable):
        self.token = torch.tensor(np.array(token), dtype=torch.long)
        self.char=torch.tensor(np.array(char), dtype=torch.long)
        self.lable= torch.tensor(np.array(lable), dtype=torch.long)      
    def call(self):
        return self.token,self.char,self.lable
        
class InputTestFeatures(object):
    """A single set of features of data."""
    def __init__(self, token,char):
        self.token = torch.tensor(np.array(token), dtype=torch.long)
        self.char=torch.tensor(np.array(char), dtype=torch.long)
    def call(self):
        return self.token,self.char
        
if __name__ == '__main__':
    root = r'/media/administrator/程序卷/zheliu/NCBI-disease-my/my_data_new/'#my_data/'#luo_data/'
    train_pkl = root + r'distant_CDWA.pkl'
    # train_pkl=root+r'train.pkl'
    # train_pkl=root+r'dev.pkl'
    # train_pkl=root+r'test.pkl'
    
    word_pkl= root +r'word_emb.pkl'

    load_lstm_path='model_knowledge_acquisition_CDWA/model_lstm67.pth'
    load_crf_path='model_knowledge_acquisition_CDWA/model_crf67.pth'

    # load_lstm_path='model_knowledge_acquisition_CDWC/model_lstm136.pth'
    # load_crf_path='model_knowledge_acquisition_CDWC/model_crf136.pth'

    word_dim=100#50
    char_dim=50#40#60
    feature_maps = [50]#[40]#[25, 25]#[30,30]#[25, 25]
    kernels = [3]#[3, 3]#[3,4]#[3, 3]#[2,3]
    hidden_dim=150#140#200
    tagset_size=3

    predict_pkl='predict_knowledge_acquisition_CDWA/distant_predict.pkl'
    # predict_pkl='predict_knowledge_acquisition_CDWA/train_predict.pkl'
    # predict_pkl='predict_knowledge_acquisition_CDWA/dev_predict.pkl'
    # predict_pkl='predict_knowledge_acquisition_CDWA/test_predict.pkl'
    
    ###########获得预测标签##################
    if not os.path.exists(predict_pkl):
        ########读取训练集语料###########
        with open(train_pkl, "rb") as f:
            train_features,word_index,char_index=pkl.load(f)
        print('读取训练集完成')
        train_count=len(train_features)
        #########获取词向量初始矩阵###############
        with open(word_pkl,'rb')as f:
            word_matrix=pkl.load(f)
        print('初始化词向量完成')
        ########加载模型##########
        lstm=wipe_view_single_cnn_lstm_no_pad_model(word_matrix,word_dim,len(char_index),char_dim,feature_maps,kernels,hidden_dim,tagset_size)
        lstm.load_state_dict(torch.load(load_lstm_path))
        lstm.cuda(device=0)

        crf = CRF(tagset_size,batch_first=True)
        crf.load_state_dict(torch.load(load_crf_path))
        crf.cuda(device=0)
        ########预测标签############
        predict=[]
        for index in tqdm(range(train_count)):
            word,char,lable=train_features[index].call()
            out=lstm(word.cuda(),char.cuda(),False)
            decoded=crf.decode(out)
            predict.append(decoded[0])
        with open(predict_pkl,'wb')as f:
            pkl.dump(predict,f,-1)
