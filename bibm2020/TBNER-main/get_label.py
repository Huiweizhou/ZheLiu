#!/usr/bin/python
# -*- coding: UTF-8 -*-
import torch
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.optimization import BertAdam
from torch import nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import time
import os
import pickle as pkl
from model import sequence_correct_label_model,sequence_label_model

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id,predict):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.predict=predict

if __name__ == '__main__':
    # distant_pkl='my_data_new/distant_CDWA_predict.pkl'
    # distant_final_pkl='my_data_new/distant_CDCA.pkl'
    # model_bert_predict_path='model_CDWA_correct/model_bert.pth'
    # model_sequence_predict_label='model_CDWA_correct/model_sequence_label.pth'

    # distant_pkl='my_data_new/distant_CDWC_predict.pkl'
    # distant_final_pkl='my_data_new/distant_CDCC.pkl'
    # model_bert_predict_path='model_CDWC_correct/model_bert.pth'
    # model_sequence_predict_label='model_CDWC_correct/model_sequence_label.pth'

    hidden_dim=768
    tagset_size=3
    max_seq_length=512
    project_dim=20
    bert_file='/biobert_v1.1_pubmed/'
    #############distant#################
    with open(distant_pkl, "rb") as f:
        # distant_data,_,_=pkl.load(f)
        distant_data=pkl.load(f)
    print(f'distant data len {len(distant_data)}')  
    ##############获得标签#####################################
    if not os.path.exists(distant_final_pkl):
        model_bert = BertModel.from_pretrained(bert_file)
        model_bert.load_state_dict(torch.load(model_bert_predict_path))   
        model_bert.cuda()
        model_sequence_label=sequence_correct_label_model(hidden_dim,tagset_size,project_dim)
        # model_sequence_label=sequence_label_model(hidden_dim,tagset_size)
        model_sequence_label.load_state_dict(torch.load(model_sequence_predict_label))   
        model_sequence_label.cuda()
        distant_dataloader = DataLoader(distant_data, batch_size=1)
        predict=[]
        for batch in tqdm(distant_dataloader):
            batch = tuple(t.cuda() for t in batch)
            input_ids, input_mask, segment_ids, label_ids,predict_label = batch
            # input_ids, input_mask, segment_ids,predict_label, label_ids = batch            
            # input_ids, input_mask, segment_ids, label_ids = batch            
            all_encoder_layers, _ = model_bert(input_ids, segment_ids, input_mask, output_all_encoded_layers=False)
            out=model_sequence_label(all_encoder_layers,predict_label,False)
            # out=model_sequence_label(all_encoder_layers,label_ids,False)
            
            # out=model_sequence_label(all_encoder_layers,False)
            decoded=np.argmax(out.squeeze(0).data.cpu().numpy(),axis=1)
            predict.append(decoded)
        # with open(distant_predict_pkl, "rb") as f:
        #     predict=pkl.load(f)
        all_input_ids = torch.tensor([f[0][0].data.cpu().numpy() for f in distant_dataloader], dtype=torch.long).cuda()
        all_input_mask = torch.tensor([f[1][0].data.cpu().numpy() for f in distant_dataloader], dtype=torch.long).cuda()
        all_segment_ids = torch.tensor([f[2][0].data.cpu().numpy() for f in distant_dataloader], dtype=torch.long).cuda()
        all_label_ids = torch.tensor(predict, dtype=torch.long).cuda()
        # all_gold_labels = torch.tensor([f[3][0].data.cpu().numpy() for f in distant_dataloader], dtype=torch.long).cuda()        
        distant_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # distant_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,all_gold_labels)
        
        with open(distant_final_pkl, "wb") as f:
            pkl.dump((distant_data), f, -1)
    else:
        with open(distant_final_pkl, "rb") as f:
           distant_data=pkl.load(f)

    #############获得整合标签#################
    
    
    write_root='my_data_new/'

    distant_final_pkl1=write_root+'distant_CDCA.pkl'
    distant_final_pkl2=write_root+'distant_CDCC.pkl'
    intergrate_pkl='my_data_new/diatant_intergrate2.pkl'

    if not os.path.exists(intergrate_pkl):
        ############distant#################
        with open(distant_final_pkl1, "rb") as f:
            # distant_data1,_,_=pkl.load(f)
            distant_data1=pkl.load(f)
            
        with open(distant_final_pkl2, "rb") as f:
            # distant_data2,_,_=pkl.load(f)
            distant_data2=pkl.load(f)
            
        
        distant_dataloader1 = DataLoader(distant_data1, batch_size=1)
        distant_dataloader2 = DataLoader(distant_data2, batch_size=1)
        # for f1,f2 in zip(distant_dataloader1,distant_dataloader2):
        #     print((f1[3][0]==f2[3][0]).float().data.cpu().numpy())
        #     ppp
        all_input_ids = torch.tensor([f[0][0].data.cpu().numpy() for f in distant_dataloader1], dtype=torch.long).cuda()
        print(all_input_ids.size())
        all_input_mask = torch.tensor([f[1][0].data.cpu().numpy() for f in distant_dataloader1], dtype=torch.long).cuda()
        print(all_input_mask.size())        
        all_segment_ids = torch.tensor([f[2][0].data.cpu().numpy() for f in distant_dataloader1], dtype=torch.long).cuda()
        print(all_segment_ids.size())        
        all_label_ids1 = torch.tensor([f[3][0].data.cpu().numpy() for f in distant_dataloader1], dtype=torch.long).cuda()
        print(all_label_ids1.size())        
        all_label_ids2 = torch.tensor([f[3][0].data.cpu().numpy() for f in distant_dataloader2], dtype=torch.long).cuda()
        print(all_label_ids2.size())        
        all_label_sims = torch.tensor([(f1[3][0]==f2[3][0]).float().data.cpu().numpy() for f1,f2 in zip(distant_dataloader1,distant_dataloader2)], dtype=torch.float).cuda()
        print(all_label_sims.size())        
        distant_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids1,all_label_ids2,all_label_sims)
        with open(intergrate_pkl, "wb") as f:
            pkl.dump((distant_data), f, -1)
    else:
        with open(intergrate_pkl, "rb") as f:
           distant_data=pkl.load(f)

    