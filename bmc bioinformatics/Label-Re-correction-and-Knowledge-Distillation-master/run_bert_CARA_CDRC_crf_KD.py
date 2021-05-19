#!/usr/bin/python
# -*- coding: UTF-8 -*-

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
from utils import write_predict_result
from model import sequence_label_model
from torchcrf import CRF

np.random.seed(1337)
torch.manual_seed(1337)
torch.cuda.manual_seed(1337)

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

def L2_soft(outputs, targets):
    softmax_outputs = F.softmax(outputs, dim=1)
    softmax_targets = F.softmax(targets, dim=1)
    return F.mse_loss(softmax_outputs, softmax_targets)

if __name__ == '__main__':
    root = r'/media/administrator/程序卷/zheliu/bc5/data/'#my_data/'#luo_data/'
    write_root='my_data_new/'
    # distant_pkl = write_root + r'distant_CDRA.pkl'
    # distant_pkl = write_root + r'distant_CDRC.pkl'    
    # train_pkl = write_root + r'train.pkl'    
    dev_pkl = write_root + r'dev.pkl'
    ori_dev_path= root + r'/original-data/split_CDR_DevelopmentSet.PubTator.txt'
    write_path='/media/administrator/程序卷/zheliu/bc5_bert/predict_CDRA_CDRC_0.5_crf_KD/'
    if not os.path.exists(write_path):
        os.makedirs(write_path)
    predict_path='/media/administrator/程序卷/zheliu/bc5_bert/predict_CDRA_CDRC_0.5_crf_KD/'
    record_path='/media/administrator/程序卷/zheliu/bc5_bert/prf_ner.txt'
    model_save_path='/media/administrator/程序卷/zheliu/bc5_bert/model_CDRA_CDRC_0.5_crf_KD'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    epoch_num=2#60
    batch_size=4#4#8#16#32
    learn_rate=5e-5#5e-4#0.005#1e-3
    hidden_dim=768
    tagset_size=5
    max_seq_length=512
    train_step=250
    
    distant_pkl=predict_path+'teacher_label_new.pkl'
    bert_file='biobert_v1.1_pubmed/'
    #############distant#################
    with open(distant_pkl, "rb") as f:
        distant_data=pkl.load(f)
    print(f'distant data len {len(distant_data)}')
    ##############dev##################
    with open(dev_pkl, "rb") as f:
        dev_data,dev_count_lists,dev_sents=pkl.load(f)
    print(f'dev data len {len(dev_data)}')
    print(f'dev token len {len(dev_sents)}')
    #############test#################
    with open(test_pkl, "rb") as f:
        test_data,test_count_lists,test_sents=pkl.load(f)
    print(f'test data len {len(test_data)}')
    print(f'test token len {len(test_sents)}')
    ###################model####################################
    # bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    model_bert = BertModel.from_pretrained(bert_file)
    model_bert.cuda()
    model_sequence_label=sequence_label_model(hidden_dim,tagset_size)
    model_sequence_label.cuda()
    model_crf=CRF(tagset_size,batch_first=True)
    model_crf.cuda()

    parameters=[]
    for param in model_bert.parameters():
        parameters.append(param)
    for param in model_sequence_label.parameters():
        parameters.append(param)
    for param in model_crf.parameters():
        parameters.append(param)
    
    optimizer=optim.Adam(parameters,learn_rate)
    # loss_cal=torch.nn.CrossEntropyLoss()
    ########训练和测试##############
    train_step_count=0
    max_f_dev=0.0  
    for epoch in range(epoch_num):
        time1 = time.time()
        count =0
        model_bert.train()
        model_sequence_label.train()
        model_crf.train()
        distant_sampler = RandomSampler(distant_data)
        distant_dataloader = DataLoader(distant_data, sampler=distant_sampler, batch_size=batch_size)
        # total_loss = Variable(torch.FloatTensor([0]).cuda(device=0))        
        for batch in tqdm(distant_dataloader):
            batch = tuple(t.cuda() for t in batch)
            input_ids, input_mask, segment_ids, label_ids_soft1,label_ids_soft2,label_ids_hard1,label_ids_hard2 = batch
            all_encoder_layers, _ = model_bert(input_ids, segment_ids, input_mask, output_all_encoded_layers=False)
            out=model_sequence_label(all_encoder_layers,True)
            #获得teacher标签
            loss_crf1=model_crf(out,label_ids_hard1,input_mask.byte(),reduction='sum')
            loss_crf2=model_crf(out,label_ids_hard2,input_mask.byte(),reduction='sum')
            input_mask=input_mask.unsqueeze(2).repeat(1,1,tagset_size).byte()
            student_out=torch.masked_select(out,input_mask).reshape(-1,tagset_size)
            label_ids_soft1=torch.masked_select(label_ids_soft1,input_mask).reshape(-1,tagset_size)
            label_ids_soft2=torch.masked_select(label_ids_soft2,input_mask).reshape(-1,tagset_size)            
            loss_l21=L2_soft(student_out,label_ids_soft1)       
            loss_l22=L2_soft(student_out,label_ids_soft2)                                     
            both_loss1= torch.add(loss_l21,-1*loss_crf1)
            both_loss2= torch.add(loss_l22,-1*loss_crf2)            
            total_loss = torch.add(both_loss1, both_loss2)
            count += 1
            # if count % batch_size == 0:
            # print(total_loss)
            total_loss = total_loss / batch_size
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            # total_loss = Variable(torch.FloatTensor([0]).cuda(device=0))
            # break                              
            if count% train_step==0:
                ################验证##############
                model_bert.eval()
                model_sequence_label.eval()
                model_crf.eval()
                dev_dataloader=DataLoader(dev_data)
                index_num=0
                decodeds=[]
                predict=[]
                for batch in tqdm(dev_dataloader):
                    batch = tuple(t.cuda() for t in batch)
                    input_ids, input_mask, segment_ids, label_ids = batch
                    all_encoder_layers, _ = model_bert(input_ids, segment_ids, input_mask, output_all_encoded_layers=False)
                    out=model_sequence_label(all_encoder_layers,False)
                    decoded=model_crf.decode(out,input_mask.byte())
                    decoded=decoded[0]
                    # decoded=np.argmax(out.squeeze(0).data.cpu().numpy(),axis=1)
                    len_count=torch.sum(input_mask)-1
                    decodeds.extend(decoded[1:len_count])
                    if dev_count_lists[index_num]:
                        predict.append(decodeds)
                        decodeds=[]                        
                    index_num+=1
                ###########写入验证集结果################
                write_file=write_path+'write_dev_'+str(train_step_count)+'.PubTator.txt'
                predict_file=predict_path+'predict_dev_'+str(train_step_count)+'.pkl'
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
                        torch.save(model_bert.state_dict(), model_save_path+'/model_bert'+'.pth')
                        torch.save(model_sequence_label.state_dict(), model_save_path+'/model_sequence_label'+'.pth')        
                        torch.save(model_crf.state_dict(), model_save_path+'/model_crf'+'.pth')                                                                
                        with open(predict_file,'wb')as f:
                            pkl.dump(predict,f,-1)
                print('验证集:',str(train_step_count)+'\t'+'\t'.join(p))
                dev_record=str(train_step_count)+'\t'+'\t'.join(p)
                with open(record_path,'a',encoding='utf-8')as w:
                    w.write(dev_record+'\n')
                model_bert.train()
                model_sequence_label.train()
                model_crf.train()                
                train_step_count+=1
    with open(record_path,'a',encoding='utf-8')as w:
        w.write(str(max_f_dev)+'\n')
    print('max_f_dev:',max_f_dev)    
    
