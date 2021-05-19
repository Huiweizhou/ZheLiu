#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pickle as pkl
import numpy as np
import torch
from torchcrf import CRF
# from my_crf_simple import teacher_CRF,student_CRF
from torch.autograd import Variable
import torch.optim as optim
from utils import write_predict_result
from model import cnn_lstm_no_pad_model
from tqdm import tqdm
import os
import torch.nn.functional as F

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
        return self.token.cuda(),self.char.cuda(),self.lable.cuda()
        

def kl(student,teacher,T=3.0):
     return F.kl_div(F.log_softmax(student/T,dim=1),F.softmax(teacher/T,dim=-1))*(T*T*2.)

def L1_soft(outputs, targets):
    softmax_outputs = F.softmax(outputs, dim=1)
    softmax_targets = F.softmax(targets, dim=1)
    return F.l1_loss(softmax_outputs, softmax_targets)

def L2_soft(outputs, targets):
    softmax_outputs = F.softmax(outputs, dim=1)
    softmax_targets = F.softmax(targets, dim=1)
    return F.mse_loss(softmax_outputs, softmax_targets)

if __name__ == '__main__':
    ########参数设置############
    root = r'/media/administrator/程序卷/zheliu/bc5/my_data/'
    distant_pkl = root + r'distant.pkl'
    dev_pkl = root + r'dev.pkl'
    word_pkl= root +r'word_emb.pkl'
    dev_path = root + r'dev.final.txt'
    split_ori_dev_path= root + r'/original-data/split_CDR_DevelopmentSet.PubTator.txt'
    write_path='/media/administrator/程序卷/zheliu/bc5/predict_base_split_teacher_student_l2soft/'
    if not os.path.exists(write_path):
        os.makedirs(write_path)
    predict_path=write_path
    record_path='/media/administrator/程序卷/zheliu/bc5/prf_ner_all_large_l2soft_new.txt'
    model_save_path='/media/administrator/程序卷/zheliu/bc5/model_base_split_teacher_student_l2soft'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    predict_distant_label_sim_pkl='/media/administrator/程序卷/zheliu/bc5/predict_base_split_teacher_student_l2soft/predict_distant_label_sim.pkl'

    lstm_load_paths=[]
    crf_load_paths=[]
    lstm_load_path='/media/administrator/程序卷/zheliu/bc5/model_distant_base_CDWA_lable_recorrect2/model_lstm3.pth'
    crf_load_path='/media/administrator/程序卷/zheliu/bc5/model_distant_base_CDWA_lable_refinery2/model_crf3.pth'
    lstm_load_paths.append(lstm_load_path)
    crf_load_paths.append(crf_load_path)
    lstm_load_path='/media/administrator/程序卷/zheliu/bc5/model_base_distant_CDWC_lable_refinery3/model_lstm1.pth'
    crf_load_path='/media/administrator/程序卷/zheliu/bc5/model_base_distant_CDWC_lable_refinery3/model_crf1.pth'
    lstm_load_paths.append(lstm_load_path)
    crf_load_paths.append(crf_load_path)
    
    num_teacher=len(lstm_load_paths)

    word_dim=100#50
    char_dim=50#40#60
    feature_maps = [50]#[40]#[25, 25]#[30,30]#[25, 25]
    kernels = [3]#[3, 3]#[3,4]#[3, 3]#[2,3]
    hidden_dim=150#140#200
    tagset_size=5
    learn_rate=1e-3#5e-4#0.005#1e-3
    epoch_num=2#60
    batch_size=32#8#4#8#16#32
    train_step=992
     ########读取远程监督语料###########
    with open(distant_pkl, "rb") as f:
        distant_features,word_index,char_index=pkl.load(f)
    print('读取远程监督语料完成')
    distant_count=len(distant_features)
    print(f'distant_pubtator_count:{distant_count}')
    ########读取验证集###########
    with open(dev_pkl, "rb") as f:
        dev_features,word_index,char_index=pkl.load(f)
    dev_sents=read_split_data(dev_path)
    print('读取验证集完成')
    dev_features=dev_features[400:]
    dev_count=len(dev_features)
    print(f'dev_count:{dev_count}')
    #########获取词向量初始矩阵###############
    with open(word_pkl,'rb')as f:
        word_matrix=pkl.load(f)
    print('初始化词向量完成')
    word_matrix=torch.FloatTensor(word_matrix)
    #########加载模型###############
    student_lstm=cnn_lstm_no_pad_model(word_matrix,word_dim,len(char_index),char_dim,feature_maps,kernels,hidden_dim,tagset_size)
    student_lstm.cuda(device=0)    
    student_crf = CRF(tagset_size,batch_first=True)
    student_crf.cuda(device=0)

    teacher_lstms=[]
    teacher_crfs=[]
    for lstm_load_path,crf_load_path in zip(lstm_load_paths,crf_load_paths):
        teacher_lstm=cnn_lstm_no_pad_model(word_matrix,word_dim,len(char_index),char_dim,feature_maps,kernels,hidden_dim,tagset_size)    
        teacher_lstm.load_state_dict(torch.load(lstm_load_path))
        teacher_lstm.cuda(device=0)
        teacher_lstms.append(teacher_lstm)
        teacher_crf = CRF(tagset_size,batch_first=True)
        teacher_crf.load_state_dict(torch.load(crf_load_path))
        teacher_crf.cuda(device=0)
        teacher_crfs.append(teacher_crf)
    parameters=[]
    for param in student_lstm.parameters():
        parameters.append(param)
    for param in student_crf.parameters():
        parameters.append(param)
    optimizer=optim.RMSprop(parameters, lr=learn_rate)
    # teacher_loss_function=torch.nn.MSELoss()
    ########获取teacher标签##########    
    if not os.path.exists(predict_distant_label_sim_pkl):
        teacher_lables=[]
        sim_scores=[]
        for idx in range(num_teacher):    
            for index in tqdm(range(distant_count)):
                word,char,lable=distant_features[index].call()        
                teacher_out,teacher_hidden=teacher_lstms[idx](word,char,False)            
                decoded=teacher_crfs[idx].decode(teacher_out)
                decoded=torch.tensor(decoded, dtype=torch.long).cuda()
                teacher_lables.append(decoded)
                if idx==1:
                    sim_score=torch.sum(decoded==teacher_lables[index]).float()
                    sim_score=sim_score/word.size(0)
                    sim_scores.append(sim_score)
        with open(predict_distant_label_sim_pkl, "wb") as f:
            pkl.dump((teacher_lables,sim_scores),f,-1)
    else:
        with open(predict_distant_label_sim_pkl, "rb") as f:
            teacher_lables,sim_scores=pkl.load(f)

    distant_count_old=distant_count
    distant_features=distant_features+distant_features
    distant_count*=2
    print(f'distant_count_old:{distant_count_old}')    
    ########训练和测试##############
    distant_index=list(range(distant_count))
    dev_index=list(range(dev_count))    
    test_index=list(range(test_count))
    max_f_dev=0.0  
    train_step_count=0
    for epoch in range(epoch_num):
        #############训练语料##############
        count=0
        sum_loss=0.0
        np.random.shuffle(distant_index)
        student_lstm.train()
        student_crf.train()
        for teacher_lstm in teacher_lstms:
            teacher_lstm.train()
        total_loss = Variable(torch.FloatTensor([0]).cuda(device=0))
        for index in tqdm(distant_index):
            word,char,lable=distant_features[index].call()
            student_out=student_lstm(word,char,True)
            lable=teacher_lables[index]
            if index<distant_pubtator_count_old:
                teacher_out,teacher_hidden=teacher_lstms[0](word,char,False)
            else:
                teacher_out,teacher_hidden=teacher_lstms[1](word,char,False)
            #计算损失
            lable_loss=student_crf(student_out,lable,reduction='sum')
            student_out=student_out.squeeze(0)
            teacher_out=teacher_out.squeeze(0)
            teacher_loss=L2_soft(student_out,teacher_out)
            both_loss = torch.add(teacher_loss, -1*lable_loss)
            total_loss = torch.add(total_loss, both_loss)
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
                student_lstm.eval()
                student_crf.eval()
                predict=[]
                for index in tqdm(dev_index):
                    word,char,lable=dev_features[index].call()
                    out=student_lstm(word,char,False)
                    decoded=student_crf.decode(out)
                    predict.append(decoded[0])
                ###########写入验证集结果################
                write_file=write_path+'write_dev_'+str(train_step_count)+'.PubTator.txt'
                predict_file=predict_path+'predict_dev_'+str(train_step_count)+'.pkl'
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
                print('验证集:',str(train_step_count)+'\t'+'\t'.join(p))
                dev_record=str(train_step_count)+'\t'+'\t'.join(p)
                torch.save(student_lstm.state_dict(), model_save_path+'/model_lstm'+str(train_step_count)+'.pth')
                torch.save(student_crf.state_dict(), model_save_path+'/model_crf'+str(train_step_count)+'.pth')
                with open(record_path,'a',encoding='utf-8')as w:
                    w.write(dev_record+'\n')
                student_lstm.train()
                student_crf.train()
                for teacher_lstm in teacher_lstms:
                    teacher_lstm.train()
                train_step_count+=1
    print('max_f_dev:',max_f_dev)    
    
