#!/usr/bin/python
# -*- coding: UTF-8 -*-
import tokenization
import torch
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
from pytorch_pretrained_bert.modeling import BertModel
from torch import nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import time
import os
import pickle as pkl



class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

def convert_single_example(words,label_list,tokenizer,label_map,max_seq_length):
    tokens=[]
    labels=[]
    features=[]
    count_list=[]
    for i, word in enumerate(words):
        token=tokenizer.tokenize(word)
        tokens.extend(token)
        label_1=label_map[label_list[i]]
        for i in range(len(token)):
            if i==0:
                labels.append(label_1)
            else:
                if label_1==1 or label_1==3:
                    labels.append(label_1+1)
                else:
                    labels.append(label_1)
    ori=tokens
    token_feas=[]
    label_feas=[]
    if len(tokens)>(max_seq_length-2):
        while len(tokens)>(max_seq_length-2):
            tmp_label=labels[:max_seq_length-2]
            for iidx in range(len(tmp_label)):
                if tmp_label.pop()==0:
                    break
            
            
            token_one = ["[CLS]"] + tokens[:len(tmp_label)] + ["[SEP]"]

            tokens = tokens[len(tmp_label):]
            labels = labels[len(tmp_label):]
            tmp_label=[0]+tmp_label+[0]
            
            token_feas.append(token_one)
            label_feas.append(tmp_label)
            
            count_list.append(0)
    count_list.append(1)
    token_one = ["[CLS]"] + tokens+ ["[SEP]"]
    tmp_label=[0]+labels+[0]
    token_feas.append(token_one)
    label_feas.append(tmp_label)
    for token_fea,label_fea in zip(token_feas,label_feas):
        segment_ids = [0] * len(token_fea)
        input_ids = tokenizer.convert_tokens_to_ids(token_fea)
        input_mask = [1] * len(input_ids)
        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        label_fea+=padding
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_fea)==max_seq_length
        
        features.append(
            InputFeatures(input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        label_id=label_fea))
    return features,count_list
        

def convert_examples_to_features(sents,lables,tokenizer, label_map, max_seq_length):
    features = []
    count_lists=[]
    tokens=[]
    for sent, label in tqdm(zip(sents,lables)):
        assert len(sent)==len(label)
        token=tokenizer.tokenize(' '.join(sent))
        feature,count_list=convert_single_example(sent,label,tokenizer,label_map,max_seq_length)
        features.extend(feature)
        count_lists.extend(count_list)
        tokens.append(token)
    return features,count_lists,tokens

def read_data(path):
    """
    读取句子和实体类别
    """
    sents_lists=[]
    types=[]
    sents_list=[]
    typ=[]
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
                typ.append(line[-1])
    return sents_lists,types

if __name__ == '__main__':
    lable2num={'O':0,'B-Chemical':1,'I-Chemical':2,'B-Disease':3,'I-Disease':4}
    max_seq_length=512
    vocab_file='biobert_v1.1_pubmed/vocab.txt'
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=False)
    root = r'data/'
    distant_path = root + r'distant_CDWA.final.txt'
    # distant_path = root + r'distant_CDWC.final.txt'    
    train_path = root + r'train.final.txt'
    dev_path = root + r'dev.final.txt'
    test_path = root + r'test.final.txt'
    write_root='my_data_new/'

    distant_pkl = write_root + r'distant_CDWA.pkl'
    # distant_pkl = write_root + r'distant_CDWC.pkl'    
    train_pkl = write_root + r'train.pkl'    
    dev_pkl = write_root + r'dev.pkl'
    test_pkl=write_root + r'test.pkl'
    #############distant#################
    if not os.path.exists(distant_pkl):
        sents,lables=read_data(distant_path)
        with open(distant_lable_refinery_pkl, "rb") as f:
            # lables,_=pkl.load(f)
            lables=pkl.load(f)            
        label_map=list(range(len(lable2num)) )
        distant_features,_,_=convert_examples_to_features(sents,lables,tokenizer, label_map, max_seq_length)
        all_input_ids = torch.tensor([f.input_ids for f in distant_features], dtype=torch.long).cuda()
        all_input_mask = torch.tensor([f.input_mask for f in distant_features], dtype=torch.long).cuda()
        all_segment_ids = torch.tensor([f.segment_ids for f in distant_features], dtype=torch.long).cuda()
        all_label_ids = torch.tensor([f.label_id for f in distant_features], dtype=torch.long).cuda()
        distant_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        with open(distant_pkl, "wb") as f:
            pkl.dump((distant_data), f, -1)
    else:
        with open(distant_pkl, "rb") as f:
           distant_data=pkl.load(f)
    print(f'distant data len {len(distant_data)}')
    ##############train##################
    if not os.path.exists(train_pkl):
        sents,lables=read_data(train_path)
        sents1,lables1=read_data(train_path)
        sents2,lables2=read_data(dev_path)   
        sents=sents1+sents2[:int(len(sents2)*0.8)]
        lables=lables1+lables2[:int(len(sents2)*0.8)]
        train_features,_,tokens=convert_examples_to_features(sents,lables,tokenizer, lable2num, max_seq_length)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long).cuda()
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long).cuda()
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long).cuda()
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long).cuda()
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        with open(train_pkl, "wb") as f:
            pkl.dump((train_data,tokens), f, -1)
    else:
        with open(train_pkl, "rb") as f:
           train_data,tokens=pkl.load(f)
    print(f'train data len {len(train_data)}')
    #############dev#################
    if not os.path.exists(dev_pkl):
        sents,lables=read_data(dev_path)
        sents=sents[int(len(sents)*0.8):]
        lables=lables[int(len(lables)*0.8):]
        dev_features,count_lists,tokens=convert_examples_to_features(sents,lables,tokenizer, lable2num, max_seq_length)
        all_input_ids = torch.tensor([f.input_ids for f in dev_features], dtype=torch.long).cuda()
        all_input_mask = torch.tensor([f.input_mask for f in dev_features], dtype=torch.long).cuda()
        all_segment_ids = torch.tensor([f.segment_ids for f in dev_features], dtype=torch.long).cuda()
        all_label_ids = torch.tensor([f.label_id for f in dev_features], dtype=torch.long).cuda()
        dev_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        with open(dev_pkl, "wb") as f:
            pkl.dump((dev_data,count_lists,tokens), f, -1)
    else:
        with open(dev_pkl, "rb") as f:
           dev_data,count_lists,tokens=pkl.load(f)
    print(f'dev data len {len(dev_data)}')
    print(f'dev token len {len(tokens)}')
    #############test#################
    if not os.path.exists(test_pkl):
        sents,lables=read_data(test_path)
        test_features,count_lists,tokens=convert_examples_to_features(sents,lables,tokenizer, lable2num, max_seq_length)
        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long).cuda()
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long).cuda()
        all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long).cuda()
        all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long).cuda()
        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        with open(test_pkl, "wb") as f:
            pkl.dump((test_data,count_lists,tokens), f, -1)
    else:
        with open(test_pkl, "rb") as f:
           test_data,count_lists,tokens=pkl.load(f)
    print(f'test data len {len(test_data)}')
    print(f'test token len {len(tokens)}')
    
