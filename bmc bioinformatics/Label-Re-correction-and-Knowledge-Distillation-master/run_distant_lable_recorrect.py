#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pickle as pkl
from model import cnn_lstm_no_pad_model
import numpy as np
import torch
from torchcrf import CRF
from torch.autograd import Variable
import torch.optim as optim
from utils import write_predict_result
from tqdm import tqdm
import os

np.random.seed(1337)
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
    distant_pkl = root + r'distant.pkl'
    dev_pkl = root + r'dev.pkl'
    word_pkl= root +r'word_emb.pkl'
    dev_path = root + r'dev.final.txt'
    ori_dev_path= root + r'/original-data/CDR_DevelopmentSet.PubTator.txt'
    write_path='/media/administrator/程序卷/zheliu/bc5/predict_distant_base_CDWA_lable_recorrect3/'
    if not os.path.exists(write_path):
        os.makedirs(write_path)
    predict_path='/media/administrator/程序卷/zheliu/bc5/predict_distant_base_CDWA_lable_recorrect3/'
    record_dev_path='/media/administrator/程序卷/zheliu/bc5/prf_ner_distant_dev.txt'
    record_test_path='/media/administrator/程序卷/zheliu/bc5/prf_ner_distant_test.txt'
    model_save_path='/media/administrator/程序卷/zheliu/bc5/model_distant_base_CDWA_lable_recorrect3'

    distant_lable_refinery_pkl=predict_path+'distant_lable_refinery.pkl'
    teacher_lstm_load_path='/media/administrator/程序卷/zheliu/bc5/model_base_split_CDWA_transfer/model_lstm5.pth'
    teacher_crf_load_path='/media/administrator/程序卷/zheliu/bc5/model_base_split_CDWA_transfer/model_crf5.pth'

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    word_dim=100#50
    char_dim=50#40#60
    feature_maps = [50]#[40]#[25, 25]#[30,30]#[25, 25]
    kernels = [3]#[3, 3]#[3,4]#[3, 3]#[2,3]
    hidden_dim=150#140#200
    tagset_size=5
    learn_rate=1e-3#5e-4#0.005#1e-3
    epoch_num=5#60
    batch_size=32#4#8#16#32
    ########读取远程监督语料###########
    with open(distant_pkl, "rb") as f:
        distant_features,word_index,char_index=pkl.load(f)
    print('读取远程监督语料完成')
    distant_count=len(distant_features)
    ########读取验证集###########
    with open(dev_pkl, "rb") as f:
        dev_features,word_index,char_index=pkl.load(f)
    dev_sents=read_data(dev_path)
    print('读取验证集完成')
    dev_count=len(dev_features)
    ########读取测试集###########
    with open(test_pkl, "rb") as f:
        test_features,word_index,char_index=pkl.load(f)
    test_sents=read_data(test_path)
    print('读取测试集完成')
    test_count=len(test_features)    
    #########获取词向量初始矩阵###############
    with open(word_pkl,'rb')as f:
        word_matrix=pkl.load(f)
    print('初始化词向量完成')
    distant_index=list(range(distant_count))
    dev_index=list(range(dev_count))
    test_index=list(range(test_count))
    ########获取lable refinery的远程监督的标签###############
    if not os.path.exists(distant_lable_refinery_pkl):
        distant_refinered_lables=[]
        teacher_lstm=cnn_lstm_no_pad_model(word_matrix,word_dim,len(char_index),char_dim,feature_maps,kernels,hidden_dim,tagset_size)
        teacher_lstm.load_state_dict(torch.load(teacher_lstm_load_path))
        teacher_lstm.cuda(device=0)    
        teacher_crf = CRF(tagset_size,batch_first=True)
        teacher_crf.load_state_dict(torch.load(teacher_crf_load_path))
        teacher_crf.cuda(device=0)
        teacher_lstm.eval()
        teacher_crf.eval()
        for index in tqdm(distant_index):
            word,char,lable=distant_features[index].call()
            out=teacher_lstm(word.cuda(),char.cuda(),False)
            decoded=teacher_crf.decode(out)
            distant_refinered_lables.append(torch.tensor(np.array(decoded[0]), dtype=torch.long))
        with open(distant_lable_refinery_pkl, "wb") as f:
            pkl.dump(distant_refinered_lables,f,-1)
    else:
        with open(distant_lable_refinery_pkl, "rb") as f:
            distant_refinered_lables=pkl.load(f)
    print('获取lable refinery的远程监督的标签完成')
    #########加载模型###############
    lstm=cnn_lstm_no_pad_model(word_matrix,word_dim,len(char_index),char_dim,feature_maps,kernels,hidden_dim,tagset_size)
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
    ########训练和测试##############
    max_f_dev=0.0
    for epoch in range(epoch_num):
        #############训练远程监督语料##############
        count=0
        sum_loss=0.0
        np.random.shuffle(distant_index)
        lstm.train()
        crf.train()
        total_loss = Variable(torch.FloatTensor([0]).cuda(device=0))
        for index in tqdm(distant_index):
            word,char,_=distant_features[index].call()
            lable=distant_refinered_lables[index]
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
        #################验证##############
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
        write_predict_result(ori_dev_path,dev_sents,predict,write_file)
        # 对实体识别进行评估
        os.chdir("/media/administrator/程序卷/zheliu/bc5/BC5CDR_Evaluation-0.0.3")
        p = os.popen('./eval_mention.sh Pubtator ' + ori_dev_path + ' ' + write_file).read()
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
        print('验证集:',str(epoch)+'\t'+'\t'.join(p))
        with open(record_dev_path,'a',encoding='utf-8')as w:
            w.write(str(epoch)+'\t'+'\t'.join(p)+'\n')
        # 对实体识别模型进行保存  
        if f1_dev!='NaN':
            torch.save(lstm.state_dict(), model_save_path+'/model_lstm'+str(epoch)+'.pth')
            torch.save(crf.state_dict(), model_save_path+'/model_crf'+str(epoch)+'.pth')
    print('max_f_dev:',max_f_dev)
