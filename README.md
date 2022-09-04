# Multi-task Heterogeneous Network Learning for Therapeutic Synergy Score Predictions 

How to use it:

1. __Download our collected and compiled dataset from https://drive.google.com/file/d/11xEGdgZLXlcxUfjajbrA3eSgIMT0-evr/view?usp=sharing, and put it into your specified file folder as the original data folder. The original data folder is used to generate the model input for each independent repeat.__

2. __Run data_processing_updated_1.ipynb and data_processing_updated_2.ipynb step by step to generate the model input for independent repeats (we have provided a group data for a repeat named as fold1 in above link).__
3. 


4. Unzip data.rar to the "data" folder.
5. Tune the hyper-parameters following the instruction in run_model_evaluation.py.
6. Run run_model_evaluation.py to get evaluation results.

Instead of the framework name in the manuscript (i.e., Muthene), we use HNEMA (Heterogeneous Network Embedding with Meta-path Aggregation) here to define the function.
