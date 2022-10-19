# Improving Therapeutic Synergy Score Predictions with Adverse Effects using Multi-task Heterogeneous Network Embedding 

How to use it:

Basic Environment:
* __Python 3.6.13__
* __Pytorch 1.10.2__
* __CUDA tool kit 11.3.1__

Step1. __Download our collected and compiled dataset from https://drive.google.com/file/d/11xEGdgZLXlcxUfjajbrA3eSgIMT0-evr/view?usp=sharing, and put it into your specified file folder as the original data folder. The original data folder is used to generate the model input for each independent repeat.__

Raw Data:
* __CCLE_expression.csv:__ Downloaded original DepMap gene expression data.
* __drugcomb_alldruginfo_dict.pickle:__ Collected information of involved drugs.
* __drugcomb_synergy_score.csv:__ Collected drug-drug-cell line synergy score samples.
* __twosides_side_effect.csv:__ Collected drug-drug adverse effect samples.
* __drug_target_interaction.csv:__ Collected drug-target interaction samples.
* __target_target_interaction.csv:__ Collected target-target interaction samples.

Step2. __Run data_processing_updated_1.ipynb and data_processing_updated_2.ipynb step/block by step/block to generate the model input for independent repeats from the Raw Data (of the original data folder) In addition, we have provided a group of data (generated by these two .ipynb files) for an independent repeat named as 'fold1' in above link).__

Step3. __Follow the instruction described in each following .py file, to read the model input generated by above two .ipynb files (e.g., files in provided 'fold1') and  run&evaluate different variants described in the manuscript:__
  * __NEW_HNEMA_evaluation_ECFP6_CCLE_withAE.py:__ the implementation of standard Muthene using ECFP6 + selected 60 cell lines (described by CCLE gene expression data) with the adverse effect prediction module.
  * __NEW_GIN_evaluation_ECFP6_CCLE_withAE.py:__ the implementation of the Muthene variant using GIN (encoding drug molecular graphs) + selected 60 cell lines (described by CCLE gene expression data) with the adverse effect prediction module.
  * __NEW_HNEMA_evaluation_ECFP6_CCLE_withoutAE.py:__ the implementation of Muthene variant using ECFP6 + selected 60 cell lines (described by CCLE gene expression data) without the adverse effect prediction module.

Instead of the framework name in the manuscript (i.e., Muthene), we use HNEMA (Heterogeneous Network Embedding with Meta-path Aggregation) here to define the function.
