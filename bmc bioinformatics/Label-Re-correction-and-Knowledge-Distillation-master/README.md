This project is the source code for the paper "Improving Biomedical Named Entity Recognition with Label Re-correction and Knowledge Distillation", which focus on the Chemical-induced Diseases (CID) Relation Extraction subtask in BioCreative V Track 3 CDR Task.

URL for BioCreative V Track 3 CDR Task: http://biocreative.org/tasks/biocreative-v/track3-cdr/

The original data and official evaluation toolkit could be found here.

=============================environmental requirements===========================

python >=3.6

pytorch >= 1.1.0

pytorch-crf >= 0.7.2

tqdm >= 4.36.1

numpy >= 1.17.2

=============================Introduction of the code==========================

preprocessd_data.py:convert the original data with the form of pubtator into the commom  form (e.g. Tricuspid	B-Disease)

processed_data.py:convert the commom form into the BLSTM input form

processed_data_bert.py:convert the commom form into the BERT input form

run_distant.py:train BLSTM-CRF model on the weakly labeled datasets

run_distant_transfer.py: transfer BLSTM-CRF model trained on the weakly labeled dataset to huaman annotated dataset as correct model

run_distant_lable_recorrect.py: use correct model to correct dataset and train BLSTM-CRF model on the corrected model

run_teacher_student.py: use knowledge distillation strategy to train student model with BLSTM-CRF structure

run_bert_crf.py:train BERT-CRF model on the weakly labeled datasets

run_bert_CARA_CDRC_crf_KD.py:use knowledge distillation strategy to train student model with BERT-CRF structure

model.py: model we used

utils.py

=============================Introduction of the datasets==================================

Created datasets will be available soon.
