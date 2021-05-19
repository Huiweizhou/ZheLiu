#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pickle as pkl
import numpy as np
import torch
from torchcrf import CRF
from torch.autograd import Variable
import torch.optim as optim
from utils import write_predict_result
from model import cnn_lstm_no_pad_model
from tqdm import tqdm
import os

np.random.seed(1337)
torch.manual_seed(1337)
torch.cuda.manual_seed(1337)


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
    root = r'/media/administrator/程序卷/zheliu/bc5/data/'
    train_pkl = root + r'train.pkl'
    dev_pkl = root + r'dev.pkl'
    test_pkl=root + r'test.pkl'
    word_pkl= root +r'word_emb.pkl'
    # test_path = root + r'cdr_test.ner.doc.token4.BIO_allfea_drug'
    dev_path = root + r'dev.final.txt'
    split_ori_dev_path= root + r'/original-data/split_CDR_DevelopmentSet.PubTator.txt'
    write_path='/media/administrator/程序卷/zheliu/bc5/predict_base_split_CDWA_transfer/'
    # model_save_path='/media/administrator/程序卷/zheliu/bc5/model_base_split_CDWC_transfer'    
    if not os.path.exists(write_path):
        os.makedirs(write_path)
    predict_path=write_path
    record_path='/media/administrator/程序卷/zheliu/bc5/prf_ner_all.txt'
    model_save_path='/media/administrator/程序卷/zheliu/bc5/model_base_split_CDWA_transfer'
    # model_save_path='/media/administrator/程序卷/zheliu/bc5/model_base_split_CDWC_transfer'    
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    lstm_load_path='/media/administrator/程序卷/zheliu/bc5/model_distant_base_CDWA/model_lstm0.pth'
    crf_load_path='/media/administrator/程序卷/zheliu/bc5/model_distant_base_CDWA/model_crf0.pth'
    word_dim=100#50
    char_dim=50#40#60
    feature_maps = [50]#[40]#[25, 25]#[30,30]#[25, 25]
    kernels = [3]#[3, 3]#[3,4]#[3, 3]#[2,3]
    hidden_dim=150#140#200
    tagset_size=5
    learn_rate=1e-3#5e-4#0.005#1e-3
    epoch_num=30#60
    batch_size=4#8#4#8#16#32
    ########读取训练集语料###########
    with open(train_pkl, "rb") as f:
        train_features,word_index,char_index=pkl.load(f)
    print('读取训练集完成')
    train_count=len(train_features)
    ########读取验证集###########
    with open(dev_pkl, "rb") as f:
        dev_features,word_index,char_index=pkl.load(f)
    dev_sents=read_split_data(dev_path)
    print('读取验证集完成')
    dev_count=len(dev_features)
    train_features=train_features+dev_features
    train_count=train_count+dev_count
    train_count=int(train_count*0.9)
    dev_features=dev_features[400:]
    dev_count=len(dev_features)
    print(f'dev_count:{dev_count}')
    #########获取词向量初始矩阵###############
    with open(word_pkl,'rb')as f:
        word_matrix=pkl.load(f)
    print('初始化词向量完成')
    #########加载模型###############
    lstm=cnn_lstm_no_pad_model(word_matrix,word_dim,len(char_index),char_dim,feature_maps,kernels,hidden_dim,tagset_size)
    lstm.load_state_dict(torch.load(lstm_load_path))
    lstm.cuda(device=0)
    crf = CRF(tagset_size,batch_first=True)
    crf.load_state_dict(torch.load(crf_load_path))
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
    ########训练和测试##############
    train_index=list(range(train_count))
    dev_index=list(range(dev_count))    
    max_f_dev=0.0    
    for epoch in range(epoch_num):
        #############训练语料##############
        count=0
        sum_loss=0.0
        np.random.shuffle(train_index)
        lstm.train()
        crf.train()
        total_loss = Variable(torch.FloatTensor([0]).cuda(device=0))
        for index in tqdm(train_index):
            word,char,lable=train_features[index].call()
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
        ################验证##############
        lstm.eval()
        crf.eval()
        predict=[]
        for index in tqdm(dev_index):
            word,char,lable=dev_features[index].call()
            out=lstm(word.cuda(),char.cuda(),False)
            decoded=crf.decode(out)
            predict.append(decoded[0])
        ###########写入验证集结果################
        write_file=write_path+'write_dev_'+str(epoch)+'.PubTator.txt'
        predict_file=predict_path+'predict_dev_'+str(epoch)+'.pkl'
        write_predict_result(split_ori_dev_path,dev_sents,predict,write_file)
        # 对实体识别进行评估
        os.chdir("/media/administrator/程序卷/zheliu/bc5/BC5CDR_Evaluation-0.0.3")
        p = os.popen('./eval_mention.sh Pubtator ' + split_ori_dev_path + ' ' + write_file).read()
        p = p.split('\n')
        for ele in p:
            if 'Precision' in ele:
                ele=ele.strip('\n').split(': ')
                precision=ele[1]
            if 'Recall'in ele:
                ele = ele.strip('\n').split(': ')
                recall = ele[1]
            if'F-score'in ele:
                ele = ele.strip('\n').split(': ')
                f1_dev = ele[1]
        if f1_dev !='NaN':
            with open(predict_file,'wb')as f:
                pkl.dump(predict,f,-1)
            if float(f1_dev)>max_f_dev:
                max_f_dev=float(f1_dev)
            torch.save(lstm.state_dict(), model_save_path+'/model_lstm'+str(epoch)+'.pth')
            torch.save(crf.state_dict(), model_save_path+'/model_crf'+str(epoch)+'.pth')
        print('验证集:',str(epoch)+'\t'+'\t'.join(p))
        dev_record=str(epoch)+'\t'+'\t'.join(p)
        with open(record_path,'a',encoding='utf-8')as w:
            w.write(dev_record+'\n')
    print('max_f_dev:',max_f_dev)    
    
