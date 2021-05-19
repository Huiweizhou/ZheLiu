#!/usr/bin/python
# -*- coding: UTF-8 -*-
from tqdm import tqdm
def change_txt(read_file,write_file):
    '''
    将pubtator格式变成txt的格式的带标签的句子
    '''
    B_tag = ['B`^', 'B``^']#Chemical,Disease
    I_tag = ['^`I', '^``I']
    with open(write_file,'w',encoding='utf-8')as write:    
        with open(read_file,encoding='utf-8')as read:
            annotations=[] 
            for line in tqdm(read.readlines()):
                if '|t|' in line:
                    split_line=line.strip('\n').split('|t|')
                    title=split_line[1]
                elif '|a|'in line:
                    split_line=line.strip('\n').split('|a|')
                    abstract=split_line[1]
                    article=title+' '+abstract
                elif  line == '\n':
                    sort_annotations = sorted(annotations, key=lambda x:int(x[2]), reverse=True)
                    final_annotations=[]
                    final_annotations.append(sort_annotations[0])                    
                    anno_len=len(sort_annotations)
                    for i in range(anno_len-1):
                        if  int(sort_annotations[i][1])<int(sort_annotations[i+1][2]):
                            if (int(sort_annotations[i][2])-int(sort_annotations[i][1])) <(int(sort_annotations[i+1][2])-int(sort_annotations[i+1][1])):
                                final_annotations.pop()
                                final_annotations.append(sort_annotations[i+1]) 
                        else:
                            final_annotations.append(sort_annotations[i+1])
                    # if final_annotations[0][0]=='12079509':
                    #     for ele in final_annotations:
                    #         print('--',ele)
                    for anno in final_annotations:
                        begin=int(anno[1])
                        end=int(anno[2])
                        id=anno[5]
                        id=id.replace('-1','None')
                        if '+' in id:
                            id=id.replace('+','|')
                        if anno[4]=='Chemical':
                            article=article[:begin]+B_tag[0]+id+'^'+article[begin:end]+I_tag[0]+article[end:]
                        else:
                            article=article[:begin]+B_tag[1]+id+'^'+article[begin:end]+I_tag[1]+article[end:]
                    write.write(article+'\n')
                    annotations=[]
                    # break
                else:
                    split_line=line.strip('\n').split('\t')                                        
                    if len(split_line)!=4:
                        annotations.append(split_line)

def change_out(read_file,write_file):
    '''
    将txt格式的句子切分，并且转换成一般的语料输入形式
    '''
    B_tag = ['B`^', 'B``^']#Chemical,Disease
    I_tag = ['^`I', '^``I']
    lables=['O','B-Chemical','I-Chemical','B-Disease','I-Disease']
    with open(write_file,'w',encoding='utf-8')as write:    
        with open(read_file,encoding='utf-8')as read:
            for line in tqdm(read.readlines()):
                line=line.strip('\n')

                # 切分句子
                for special in "!\"#$%()*+,-./:;<=>?@[\\]_{}~":
                    line = line.replace(special, ' '+special+' ')
                line = line.replace('\'', ' \'')
                line =line.replace('    ', ' ').replace('   ', ' ').replace('  ', ' ')

                # 转换成一般的语料输入形式
                line =line.split(' ')
                lable=lables[0]
                for word in line:
                    if word!='':
                        if B_tag[1] in word:
                            lable=lables[3]
                            split_word=word.split('^')
                            word=split_word[2]
                            write.write(word+'\t'+lable+'\n')
                            if len(split_word)==4:
                                lable=lables[0]
                            else:
                                lable=lables[4]
                        elif B_tag[0] in word:
                            lable=lables[1]
                            split_word=word.split('^')
                            word=split_word[2]
                            write.write(word+'\t'+lable+'\n')   
                            if len(split_word)==4:
                                lable=lables[0]
                            else:
                                lable=lables[2]                       
                        elif I_tag[0] in word or I_tag[1] in word:
                            split_word=word.split('^')
                            word=split_word[0]
                            write.write(word+'\t'+lable+'\n')
                            lable=lables[0]                        
                        else:
                            write.write(word+'\t'+lable+'\n')                        
                write.write('\n')    
                
                    

if __name__ == '__main__':
    root = r'data/'

    # 第一步 将pubtator格式变成txt的格式的带标签的句子
    train_path = root + r'original-data/CDR_TrainingSet.PubTator.txt'
    dev_path = root + r'original-data/CDR_DevelopmentSet.PubTator.txt'
    test_path = root + r'original-data/CDR_TestSet.PubTator.txt'    
    distant_CDWA_path = root + r'original-data/CDWA.PubTator.txt'
    distant_CDWC_path = root + r'original-data/CDWC.pubtator.txt'    
    processed_train_path = root + r'train.txt'
    processed_dev_path = root + r'dev.txt'
    processed_test_path = root + r'test.txt'
    processed_distant_CDWA_path = root + r'distant_CDWA.txt'
    processed_distant_CDWC_path = root + r'distant_CDWC.txt'    
    read_files=[train_path,dev_path,test_path,distant_CDWA_path,distant_CDWC_path]
    write_files=[processed_train_path,processed_dev_path,processed_test_path,processed_distant_CDWA_path,processed_distant_CDWC_path]
    for read_file,write_file in zip(read_files,write_files):
        change_txt(read_file,write_file)

    # 第二步 将txt格式的句子切分，并且转换成一般的语料输入形式
    train_path = root + r'train.txt'
    dev_path = root + r'dev.txt'
    test_path = root + r'test.txt'
    distant_CDWA_path = root + r'distant_CDWA.txt'
    distant_CDWC_path = root +r'distant_CDWC.txt'    
    processed_train_path = root + r'train.final.txt'
    processed_dev_path = root + r'dev.final.txt'
    processed_test_path = root + r'test.final.txt'
    processed_distant_CDWA_path = root + r'distant_CDWA.final.txt'
    processed_distant_CDWC_path = root + r'distant_CDWC.final.txt'    
    read_files=[train_path,dev_path,test_path,distant_CDWA_path,distant_CDWC_path]
    write_files=[processed_train_path,processed_dev_path,processed_test_path,processed_distant_CDWA_path,processed_distant_CDWC_path]
    for read_file,write_file in zip(read_files,write_files):
        change_out(read_file,write_file)
    
    
