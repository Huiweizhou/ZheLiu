# Two-perspective-Biomedical-Named-Entity-Recognition-with-Weakly-Labeled-Data-Correction

URL for BioCreative V Track 3 CDR Task: https://biocreative.bioinformatics.udel.edu/tasks/biocreative-v/track-3-cdr/

URL for BioCreative IV Track 2- CHEMDNER Task: https://biocreative.bioinformatics.udel.edu/tasks/biocreative-iv/chemdner/

URL for NCBI disease Task: https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/

The original data and official evaluation toolkit could be found here.

=====================environmental requirements============================

python >=3.6

pytorch >= 1.1.0

pytorch-crf >= 0.7.2

tqdm >= 4.36.1

numpy >= 1.17.2

pytorch_pretrained_bert >= 0.6.2

biobertv1.1: https://github.com/dmis-lab/biobert

=====================Introduction of the code==========================

preprocessd_data.py:convert the original data with the form of pubtator into the commom form (e.g. Tricuspid B-Disease)

processed_data.py:convert the commom form into the BLSTM input form

run_judge.py: run judge model (BLSTM-CRF)

run_knowledge_acquisition.py: run knowledge acquisition module (BLSTM-CRF) with curriculum learning

get_predict.py: acquire the predict of knowledge acquisition module (BLSTM-CRF)

convert_blstm2bert.py: convert the input form of BLSTM into the form of BioBERT

run_noisy_correction.py: run noisy correction module(BioBERT)

get_label.py: use noisy correction module to correct weakly labeled dataset

run_partial_label_integrating.py: run partial label integrating (BioBERT)

=====================weakly dataset==========================

Due to upload memory limitations, please contact the author for corpus.
