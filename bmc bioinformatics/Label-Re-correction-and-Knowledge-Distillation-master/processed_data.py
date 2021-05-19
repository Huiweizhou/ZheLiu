#!/usr/bin/python
# -*- coding: UTF-8 -*-
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import torch
import os
from utils import *
import pickle as pkl
import string
np.random.seed(1337)
torch.manual_seed(1337)
torch.cuda.manual_seed(1337)

def read_data(path):
    """
    读取pos、chunk、句子和实体类别
    """
    sents_lists=[]
    types=[]
    sents_list=[]
    typ=[]
    word_count=dict()
    with open (path,encoding='utf-8')as read:
        for line in tqdm(read.readlines()):
            if line =='\n':
                sents_lists.append(sents_list)
                types.append(typ)
                sents_list=[]
                typ=[]
            else:
                line=line.strip('\n').split('\t')
                word=line[0]
                sents_list.append(word)
                typ.append(line[5])
                word=wordNormalize(word)                
                if word not in word_count:
                    word_count[word]=1
                else:
                    word_count[word]+=1
    return sents_lists,types,word_count

def read_distant_data(path):
    """
    读取pos、chunk、句子和实体类别
    """
    sents_lists=[]
    types=[]
    sents_list=[]
    typ=[]
    word_count=dict()
    with open (path,encoding='utf-8')as read:
        for line in tqdm(read.readlines()):
            if line =='\n':
                sents_lists.append(sents_list)
                types.append(typ)
                sents_list=[]
                typ=[]
            else:
                line=line.strip('\n').split('\t')
                word=line[0]
                sents_list.append(word)
                typ.append(line[1])
                word=wordNormalize(word)                
                if word not in word_count:
                    word_count[word]=1
                else:
                    word_count[word]+=1
    return sents_lists,types,word_count

class InputTrainFeatures(object):
    """A single set of features of data."""
    def __init__(self, token,char,lable):
        self.token = torch.tensor(np.array(token), dtype=torch.long)
        self.char=torch.tensor(np.array(char), dtype=torch.long)
        self.lable= torch.tensor(np.array(lable), dtype=torch.long)      
    def call(self):
        return self.token,self.char,self.lable

def convert_train_to_features(sents,types,word_emb,word_index,char_index,word_count,max_char_length,is_distant):
    '''
    将txt文件转成模型需要的输入形式
    '''
    features = []
    lable2num={'O':0,'B-Chemical':1,'I-Chemical':2,'B-Disease':3,'I-Disease':4}
    if is_distant:
        threshold=10
    else:
        threshold=5
    for  sent,typ in tqdm(zip(sents,types)):
        count=0
        position=0
        token=[]
        entity_lable=[]
        char_list=[]
        for word in sent:
            #字符特征
            chars=[]
            for char in word:
                if not char_index.get(char):
                    chars.append(char_index['**'])
                else:
                    chars.append(char_index[char])
            chars=chars[:max_char_length]
            chars+=[0] * (max_char_length-len(chars))
            char_list.append(chars)        
            #词特征
            word=wordNormalize(word)
            if not word_index.get(word) :
                if word_count[word]>=threshold:
                    word_index[word]=len(word_index)+2
                    token.append(word_index[word])
                else:
                    if word in word_emb:
                        word_index[word]=len(word_index)+2
                        token.append(word_index[word])
                    else:
                        for punc in string.punctuation:
                            word = word.replace(punc, '')
                        if  word_index.get(word):
                            token.append(word_index[word])
                        elif word in word_emb:
                            word_index[word]=len(word_index)+2
                            token.append(word_index[word])
                        else:
                            token.append(1)
            else:
                token.append(word_index[word])
                    
            #lable序列
            entity_lable.append(lable2num[typ[position]])
            position+=1
        
        assert len(token)==len(char_list)==len(entity_lable)
        

        features.append(
            InputTrainFeatures(token=token,
                          char=char_list,
                          lable=entity_lable
                          ))
    return features,word_index

class InputTestFeatures(object):
    """A single set of features of data."""
    def __init__(self, token,char):
        self.token = torch.tensor(np.array(token), dtype=torch.long)
        self.char=torch.tensor(np.array(char), dtype=torch.long)
    def call(self):
        return self.token,self.char

def convert_test_to_features(sents, word_emb,word_index,char_index,max_char_length):
    features = []
    for  sent in tqdm(sents):
        count=0
        position=0
        token=[]
        char_list=[]
        for word in sent:
            #字符特征
            chars=[]
            for char in word:
                if not char_index.get(char):
                    chars.append(char_index['**'])
                else:
                    chars.append(char_index[char])
            chars=chars[:max_char_length]
            chars+=[0] * (max_char_length-len(chars))
            char_list.append(chars)        
            #词特征
            word=wordNormalize(word)
            if not word_index.get(word) :
                if word in word_emb:
                    word_index[word]=len(word_index)+2
                    token.append(word_index[word])
                else:
                    for punc in string.punctuation:
                        word = word.replace(punc, '')
                    if  word_index.get(word):
                        token.append(word_index[word])
                    elif word in word_emb:
                        word_index[word]=len(word_index)+2
                        token.append(word_index[word])
                    else:
                        token.append(1)
            else:
                    token.append(word_index[word])
            position+=1

        assert len(token)==len(char_list)
        
        features.append(
            InputTestFeatures(token=token,
                          char=char_list
                          ))
    return features,word_index


def produce_matrix(word_index, word_emb,word_size):
    num_words=len(word_index)+2
    embedding_matrix = np.zeros((num_words, word_size))
    # embedding_matrix[1]=word_emb["UNKNOWN_TOKEN"]
    embedding_matrix[1]=np.random.uniform(-0.1, 0.1, word_size)
    for word, i in word_index.items():
        if word in word_emb:
            vec = word_emb.get(word)
        else:
            vec = np.random.uniform(-0.1, 0.1, word_size)
        embedding_matrix[i] = vec
    return embedding_matrix

def read_txt_embedding(embFile,word_size):
    """
    读取预训练的词向量文件，引入外部知识
    """
    print("\nProcessing Embedding File...")
    embeddings = OrderedDict()
    embeddings["PADDING_TOKEN"] = np.zeros(word_size)
    embeddings["UNKNOWN_TOKEN"] = np.random.uniform(-0.1, 0.1, word_size)
    # embeddings["NUMBER"] = np.random.uniform(-0.1, 0.1, word_size)

    with open(embFile) as f:
        lines = f.readlines()
    for i in tqdm(range(len(lines))):
        line = lines[i]
        if len(line.split())<=2:
            continue
        values = line.strip().split()
        # word = values[0].lower()
        word = wordNormalize(values[0])
        vector = np.asarray(values[1:], dtype=np.float32)
        # for word, replace digits
        embeddings[word] = vector
    return embeddings

if __name__ == '__main__':
    ########参数设置############
    root = r'data/'
    train_path = root + r'train.final.txt'
    dev_path = root + r'dev.final.txt'
    test_path = root + r'test.final.txt'
    distant_CDWA_path = root + r'distant_CDWA.final.txt'
    distant_CDWC_path = root + r'distant_CDWC.final.txt'
    train_pkl = root + r'train.pkl'
    dev_pkl = root + r'dev.pkl'
    test_pkl = root + r'test.pkl'
    distant_CDWA_pkl = root + r'distant_CDWA.pkl'
    distant_CDWC_pkl = root + r'distant_CDWC.pkl'    
    emb_path='../pubmed_w2v.lck.d100'
    word_pkl= root +r'word_emb.pkl'
    max_char_length = 29
    word_dim=50
    word_index=OrderedDict()#0给pad，1给未登录词
    char_index=createCharDict()
    ########读取词向量###########
    if not os.path.exists(word_pkl):
        word_emb=read_txt_embedding(emb_path,word_dim)        
    ########读取远程监督语料###########
    if not os.path.exists(distant_CDWA_pkl):
        sents,types,word_count=read_distant_data(distant_CDWA_path)
        distant_features,word_index=convert_train_to_features(sents,types,word_emb,word_index,char_index,word_count,max_char_length,True)
        with open(distant_CDWA_pkl, "wb") as f:
            pkl.dump((distant_features,word_index,char_index), f, -1)
    else:
        with open(distant_CDWA_pkl, "rb") as f:
           distant_features,word_index,char_index=pkl.load(f)
    print('读取远程监督语料完成')
    ########读取远程监督语料###########
    if not os.path.exists(distant_CDWC_pkl):
        sents,types,word_count=read_distant_data(distant_CDWC_path)
        distant_features,word_index=convert_train_to_features(sents,types,word_emb,word_index,char_index,word_count,max_char_length,True)
        with open(distant_CDWC_pkl, "wb") as f:
            pkl.dump((distant_features,word_index,char_index), f, -1)
    else:
        with open(distant_CDWC_pkl, "rb") as f:
           distant_features,word_index,char_index=pkl.load(f)
    print('读取远程监督语料完成')
    ########读取训练集###########
    if not os.path.exists(train_pkl):
        sents,types,word_count=read_data(train_path)
        train_features,word_index=convert_train_to_features(sents,types,word_emb,word_index,char_index,word_count,max_char_length,False)
        with open(train_pkl, "wb") as f:
            pkl.dump((train_features,word_index,char_index), f, -1)
    else:
        with open(train_pkl, "rb") as f:
           train_features,word_index,char_index=pkl.load(f)
    print('读取训练集完成')    
    ########读取验证集###########
    if not os.path.exists(dev_pkl):
        sents,types,word_count=read_data(dev_path)
        dev_features,word_index=convert_train_to_features(sents,types,word_emb,word_index,char_index,word_count,max_char_length,False)
        with open(dev_pkl, "wb") as f:
            pkl.dump((dev_features,word_index,char_index), f, -1)
    else:
        with open(dev_pkl, "rb") as f:
            dev_features,word_index,char_index=pkl.load(f)
    print('读取验证集完成')        
   ########读取测试集###########
    if not os.path.exists(test_pkl):
        sents,types,word_count=read_data(test_path)
        test_features,word_index=convert_test_to_features(sents, word_emb,word_index,char_index,max_char_length)
        with open(test_pkl, "wb") as f:
            pkl.dump((test_features,word_index,char_index), f, -1)
    else:
        with open(test_pkl, "rb") as f:
           test_features,word_index,char_index=pkl.load(f)
    print('读取测试集完成')    
    #########获取词向量初始矩阵###############
    if not os.path.exists(word_pkl):
        word_matrix=produce_matrix(word_index, word_emb,word_dim)
        with open(word_pkl,'wb')as f:
            pkl.dump(word_matrix,f,-1)
    else:
        with open(word_pkl,'rb')as f:
            word_matrix=pkl.load(f)
    print(len(word_index))
