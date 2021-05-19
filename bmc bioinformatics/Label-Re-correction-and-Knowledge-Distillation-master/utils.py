#!/usr/bin/python
# -*- coding: UTF-8 -*-
import string
import re
from tqdm import tqdm 
def wordNormalize(word):
    '''
    对单词进行清洗,特殊符号归一化
    :param word:
    :return:
    '''
    word = word.strip().lower()
    word = re.sub(u'\s+', '', word, flags=re.U)  # 匹配任何空白字符
    word = word.replace("--", "-")
    word = re.sub("\"+", '"', word)

    if word.isdigit():
        word = '1'
    else:
        temp = word
        for char in word:
            if char not in string.printable:
                temp = temp.replace(char, '*')
        word = temp
    return word

def createCharDict():
    '''
    创建字符字典
    '''
    char2idx = {}
    char2idx['None'] = len(char2idx)  # 0索引用于填充
    for char in string.printable:
        char2idx[char] = len(char2idx)
    char2idx['**'] = len(char2idx)  # 用于那些未收录的字符
    # print(char2idx)
    return char2idx


def write_predict_result(ori_path,sents,predict,write_path):
    count=0
    with open(write_path,'w') as writer:
        with open(ori_path) as reader:
            for line in reader.readlines():
                if '|t|' in line:
                    writer.write(line)
                    line = line.strip('\n').split('|t|')
                    norm_sent=line[1]+' '
                elif '|a|' in line:
                    writer.write(line)
                    line = line.strip('\n').split('|a|')
                    norm_sent+= line[1]
                    sent=sents[count]
                    assert len(sent)==len(predict[count])
                    offset_begin_list=[]                 
                    entity_list=[]
                    type_list=[]
                    predict_sent=predict[count]
                    # print(sent)
                    # print(predict_sent)
                    offset=0
                    entity = ''
                    prex=0
                    for tokenIdx in range(len(predict_sent)):
                        label = predict_sent[tokenIdx]
                        word = sent[tokenIdx]
                        offset_before=offset
                        while norm_sent[offset:offset+len(word)]!=word:                   
                            offset+=1
                        if label == 0 or label == 1 or label == 3:
                            if entity:
                                entity=norm_sent[entity_offset:offset_before]
                                entity_list.append(entity)
                                type_list.append('Chemical' if prex == 1 or prex == 2 else 'Disease')
                                offset_begin_list.append(entity_offset)
                                entity=''
                            if label == 1 or label == 3:
                                entity = word + ' '
                                entity_offset=offset
                            prex = label
                        elif label == 2:
                            if prex == 1 or prex == 2:
                                entity += word + ' '
                                prex = label
                        else:
                            if prex == 3 or prex == 4:
                                entity += word + ' '
                                prex = label    
                        offset+=len(word)
                    for offset,entity,typ in zip(offset_begin_list,entity_list,type_list):
                        writer.write(line[0]+'\t'+str(offset)+'\t'+str(offset+len(entity))+'\t'+norm_sent[offset:offset+len(entity)]+'\t'+typ+'\t'+'-1\n')
                elif line == '\n':
                    writer.write(line)
                    count+=1                    
    # print('\n写入完成\n')

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
    import pickle as pkl
    import os
    root = r'/home/liuzhe/bc5/my_data/'
    ori_test_path= root + r'/original-data/CDR_DevelopmentSet.PubTator.txt'#root + r'/original-data/CDR_TestSet.PubTator.txt'
    test_path =root + r'dev.final.txt'# root + r'test.final.txt'
    test_sents=read_data(test_path)
    write_file='/home/liuzhe/bc5/predict/write_try.PubTator.txt'
    # predict_path='/home/liuzhe/bc5/predict/predict_dev_6.pkl'
    # with open(predict_path, "rb") as f:
        # predict=pkl.load(f)
    dev_path = root + r'dev.pkl'
    with open(dev_path, "rb") as f:
        dev_features,word_index,char_index=pkl.load(f)
    predict=[]
    for dev_fea in dev_features:
        predict.append(dev_fea.lable)
        # print(dev_fea.lable)
    write_predict_result(ori_test_path,test_sents,predict,write_file)
    # 对实体识别进行评估
    os.chdir("/home/liuzhe/bc5/BC5CDR_Evaluation-0.0.3")
    p = os.popen('./eval_mention.sh Pubtator ' + ori_test_path + ' ' + write_file).read()
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
            f1 = ele[1]
    
    print('\t'.join(p))

def compute_precision(guessed, correct):
    correctCount = 0
    count = 0
    idx = 0
    while idx < len(guessed):
        if guessed[idx]== 1 or guessed[idx]==3: #A new chunk starts
            count += 1
            if guessed[idx] == correct[idx]:
                idx += 1
                correctlyFound = True
                while idx < len(guessed) and (guessed[idx] == 2 or guessed[idx] ==4): #Scan until it no longer starts with I
                    if guessed[idx] != correct[idx]:
                        correctlyFound = False
                    idx += 1
                if idx < len(guessed):
                    if correct[idx]== 2 or correct[idx]==4: #The chunk in correct was longer
                        correctlyFound = False
                if correctlyFound:
                    correctCount += 1
            else:
                idx += 1
        else:  
            idx += 1
    
    precision = 0.0
    if count > 0:    
        precision = float(correctCount) / count
    return precision

def sent_prf_cal(predict_sent,gold_sent):
    assert  len(predict_sent)==len(gold_sent)
    prec = compute_precision(predict_sent, gold_sent)
    rec = compute_precision(gold_sent, predict_sent)
    f1=0.0
    if (rec+prec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec)
    return f1
