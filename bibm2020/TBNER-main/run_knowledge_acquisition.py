#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pickle as pkl
from model import wipe_view_single_cnn_lstm_no_pad_model
import numpy as np
import torch
from torchcrf import CRF
from torch.autograd import Variable
import torch.optim as optim
from utils import sent_prf_cal
from tqdm import tqdm
import os

np.random.seed(1337)
mySeed = np.random.RandomState (1337)
torch.manual_seed(1337)
torch.cuda.manual_seed(1337)

def read_data(path):
    """
    读取句子
    """
    sents_lists=[]
    sents_list=[]
    with open (path,encoding='utf-8')as read:
        for line in tqdm(read.readlines()):
            if line =='\n':
                sents_lists.append(sents_list)
                sents_list=[]
            else:
                line=line.strip('\n').split('\t')
                word=line[0]
                sents_list.append(word)
    return sents_lists

def read_split_data(path):
    """
    读取句子
    """
    sents_lists=[]
    sents_list=[]
    count=0
    with open (path,encoding='utf-8')as read:
        for line in tqdm(read.readlines()):
            if count<400:
                if line =='\n':
                    count+=1
            else:
                if line =='\n':
                    sents_lists.append(sents_list)
                    sents_list=[]
                else:
                    line=line.strip('\n').split('\t')
                    word=line[0]
                    sents_list.append(word)
    return sents_lists

class InputTrainFeatures(object):
    """A single set of features of data."""
    def __init__(self, token,char,lable):
        self.token = torch.tensor(np.array(token), dtype=torch.long)
        self.char=torch.tensor(np.array(char), dtype=torch.long)
        self.lable= torch.tensor(np.array(lable), dtype=torch.long)      
    def call(self):
        return self.token,self.char,self.lable
        
if __name__ == '__main__':
    ########参数设置############
    root = r'/my_data_new/'
    write_root = r'/NCBI-disease-my/'  
    train_pkl = root + r'distant_CDWA.pkl'
    dev_pkl = root + r'dev.pkl'
    word_pkl= root +r'word_emb.pkl'
    dev_path = root + r'dev.final.txt'
    write_path=write_root+'predict_knowledge_acquisition_CDWA/'
    if not os.path.exists(write_path):
        os.makedirs(write_path)
        print(write_path,'succeed')
    predict_path=write_path
    record_path=write_root+'prf_ner_all.txt'
    model_save_path=write_root+'model_knowledge_acquisition_CDWA'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    word_dim=100#50
    char_dim=50#40#60
    feature_maps = [50]#[40]#[25, 25]#[30,30]#[25, 25]
    kernels = [3]#[3, 3]#[3,4]#[3, 3]#[2,3]
    hidden_dim=150#140#200
    tagset_size=3
    learn_rate=1e-3#5e-4#0.005#1e-3
    epoch_num=3#60
    batch_size=32#8#4#8#16#32
    train_step=992

    judge_sims_pkl=predict_path+'judge_f.pkl'
    judge_lstm_path='model_judge/model_lstm15.pth'
    judge_crf_path='model_judge/model_crf15.pth'
    ########读取训练集语料###########
    with open(train_pkl, "rb") as f:
        train_features,word_index,char_index=pkl.load(f)
    print('读取训练集完成')
    train_count=len(train_features)
    ########读取验证集###########
    with open(dev_pkl, "rb") as f:
        dev_features,word_index,char_index=pkl.load(f)
    dev_sents=read_data(dev_path)
    print('读取验证集完成')
    dev_count=len(dev_features)
    print(f'dev_count:{dev_count}')
    #########获取词向量初始矩阵###############
    with open(word_pkl,'rb')as f:
        word_matrix=pkl.load(f)
    print('初始化词向量完成')
    #########加载模型###############
    lstm=wipe_view_single_cnn_lstm_no_pad_model(word_matrix,word_dim,len(char_index),char_dim,feature_maps,kernels,hidden_dim,tagset_size)
    lstm.cuda(device=0)
    crf = CRF(tagset_size,batch_first=True)
    crf.cuda(device=0)
    parameters=[]
    for param in lstm.parameters():
        parameters.append(param)
    for param in crf.parameters():
        parameters.append(param)
    optimizer=optim.RMSprop(parameters, lr=learn_rate)
    # optimizer=optim.Adam(parameters, lr=learn_rate)
    # optimizer=optim.Adagrad(parameters, lr=learn_rate)    
    # optimizer=optim.SGD(parameters, lr=learn_rate)
    ########加载judge模型##########
    if not os.path.exists(judge_sims_pkl):
        judge_lstm=wipe_view_single_cnn_lstm_no_pad_model(word_matrix,word_dim,len(char_index),char_dim,feature_maps,kernels,hidden_dim,tagset_size)
        judge_lstm.load_state_dict(torch.load(judge_lstm_path))
        judge_lstm.cuda(device=0)

        judge_crf = CRF(tagset_size,batch_first=True)
        judge_crf.load_state_dict(torch.load(judge_crf_path))
        judge_crf.cuda(device=0)
        ########获取相似度得分#########    
        f_scores=[]
        for index in tqdm(range(train_count)):
            word,char,lable=train_features[index].call()
            segment_ids=torch.zeros_like(lable)
            lable=torch.where(lable<3,segment_ids,lable-2).cuda() 
            out=judge_lstm(word.cuda(),char.cuda(),False)
            decoded=judge_crf.decode(out) 
            judge_label=np.array(decoded[0])
            f=sent_prf_cal(lable.data.cpu().numpy(),judge_label)
            f_scores.append(f)
        with open(judge_sims_pkl, "wb") as f:
            pkl.dump((f_scores),f,-1)
    else:
         with open(judge_sims_pkl, "rb") as f:
             f_scores=pkl.load(f)
    f_scores=[ele[2] for ele in f_scores]
    print('获取f_scores完成')
    train_index=np.argsort(np.array(f_scores))#从小到大   
    ########训练和测试##############
    dev_index=list(range(dev_count))    
    max_f=0.0
    max_f_dev=0.0    
    train_step_count=0
    for epoch in range(epoch_num):
        #############训练语料##############
        count=0
        sum_loss=0.0
        lstm.train()
        crf.train()
        total_loss = Variable(torch.FloatTensor([0]).cuda(device=0))
        for index in tqdm(train_index):
            word,char,lable=train_features[index].call()
            segment_ids=torch.zeros_like(lable)
            lable=torch.where(lable<3,segment_ids,lable-2).cuda()
            out=lstm(word.cuda(),char.cuda(),True)
            loss=crf(out,lable.unsqueeze(0).cuda(),reduction='sum')
            total_loss = torch.add(total_loss, -1*loss)
            count += 1
            if count % batch_size == 0:
                total_loss = total_loss / batch_size
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                total_loss = Variable(torch.FloatTensor([0]).cuda(device=0))
                # break
            if count% train_step==0:
                ################验证##############
                lstm.eval()
                crf.eval()
                predict=[]
                gold=[]
                for index in tqdm(dev_index):
                    word,char,lable=dev_features[index].call()
                    out=lstm(word.cuda(),char.cuda(),False)
                    decoded=crf.decode(out)
                    predict.extend(decoded[0])
                    gold.extend(lable.data.cpu().numpy())            
                ###########写入验证集结果################
                predict_file=predict_path+'predict_dev_'+str(train_step_count)+'.pkl'
                p_dev,r_dev,f1_dev=sent_prf_cal(predict,gold)
                dev_record=str(epoch)+'\t'+str(p_dev)+'\t'+str(r_dev)+'\t'+str(f1_dev)
                print('验证集:',dev_record)      
                with open(predict_file,'wb')as f:
                    pkl.dump(predict,f,-1)
                if float(f1_dev)>max_f_dev:
                    max_f_dev=float(f1_dev)
                    torch.save(lstm.state_dict(), model_save_path+'/model_lstm'+str(train_step_count)+'.pth')
                    torch.save(crf.state_dict(), model_save_path+'/model_crf'+str(train_step_count)+'.pth')
                with open(record_path,'a',encoding='utf-8')as w:
                    w.write(dev_record+'\n')
                train_step_count+=1
                lstm.train()
                crf.train()
    with open(record_path,'a',encoding='utf-8')as w:
        w.write(str(max_f_dev)+'\n')        
    print('max_f_dev:',max_f_dev)    
    