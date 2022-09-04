# Multi-task Heterogeneous Network Learning for Therapeutic Synergy Score Predictions 

How to use it:

1. __Download our collected and compiled dataset from https://drive.google.com/file/d/11xEGdgZLXlcxUfjajbrA3eSgIMT0-evr/view?usp=sharing, and put it into your specified file folder as the original data folder. The original data folder is used to generate the model input for each independent repeat.__

2. __Run data_processing_updated_1.ipynb and data_processing_updated_2.ipynb step by step to generate the model input for independent repeats (we have provided a group of data for an independent repeat named as 'fold1' in above link).__
3. __Follow the instruction in each following .py file to execute different methods described in the manuscript:__
  * __NEW_HNEMA_evaluation_ECFP6_CCLE_withAE.py:__ the implementation of Muthene using ECFP6 + selected 60 cell lines (described by CCLE gene expression data) with adverse effect module.
  * __NEW_GIN_evaluation_ECFP6_CCLE_withAE.py:__ the implementation of the Muthene variant using GIN (encoding drug molecular graphs) + selected 60 cell lines (described by CCLE gene expression data) with adverse effect module.
  * __NEW_HNEMA_evaluation_ECFP6_CCLE_withoutAE.py:__ the implementation of Muthene using ECFP6 + selected 60 cell lines (described by CCLE gene expression data) without adverse effect module.

Instead of the framework name in the manuscript (i.e., Muthene), we use HNEMA (Heterogeneous Network Embedding with Meta-path Aggregation) here to define the function.
