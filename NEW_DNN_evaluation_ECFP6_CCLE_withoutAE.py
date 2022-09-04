import time
import argparse
import torch
from torch.optim import lr_scheduler
import numpy as np
from model.Auxiliary_networks import DNN_predictor
from utils.pytorchtools import EarlyStopping
from utils.data import load_DNN_DDI_data_te
from utils.tools import index_generator
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import scipy.stats
import random

# the implementation of the DeepSynergy using ECFP6 + selected 60 cell lines (described by CCLE gene expression data) without adverse effects

# Params
random_seed = 1024
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

# some overall fixed parameters
# drug/target/cell line
num_ntype = 3
lr = 0.005
weight_decay = 0.001

# total numbers of drug and target nodes
num_drug = 106
num_target = 12217
num_cellline = 60

only_test = False

# S_mean，synergy_zip，synergy_loewe，synergy_hsa，synergy_bliss (0,1,2,3,4)
# the type of synergy score to be predicted
predicted_te_type = 2

# args.root_prefix, args.epoch, args.patience, args.batch_size, args.repeat, args.hidden_dim_aux, args.layer_list, args.dropout, args.input_dropout
def run_model_DNN_DDI(root_prefix, num_epochs, patience, batch_size, repeat, hidden_dim_aux, layer_list, dropout, input_dropout, whether_CCLE):
    print('whether_CCLE, hidden_dim_aux:', whether_CCLE, hidden_dim_aux)
    name2id_dict, train_val_test_drug_drug_samples, train_val_test_drug_drug_labels, all_drug_morgan, cellline_expression = load_DNN_DDI_data_te(root_prefix)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    train_drug_drug_samples = train_val_test_drug_drug_samples['train_drug_drug_samples']
    print('total training set sample number is:',train_drug_drug_samples.shape[0])

    train_te_labels = torch.tensor(train_val_test_drug_drug_labels['train_te_labels'],dtype=torch.float32).to(device)

    # an extra test about exchanging the val and test set
    val_drug_drug_samples = train_val_test_drug_drug_samples['val_drug_drug_samples']
    test_drug_drug_samples = train_val_test_drug_drug_samples['test_drug_drug_samples']
    val_te_labels = torch.tensor(train_val_test_drug_drug_labels['val_te_labels'],dtype=torch.float32).to(device)
    test_te_labels = torch.tensor(train_val_test_drug_drug_labels['test_te_labels'],dtype=torch.float32).to(device)

    morgan_values = all_drug_morgan.data
    morgan_indices = np.vstack((all_drug_morgan.row, all_drug_morgan.col))
    i = torch.LongTensor(morgan_indices)
    v = torch.FloatTensor(morgan_values)
    shape = all_drug_morgan.shape
    all_drug_morgan = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense().to(device)

    # gene expression data of cell lines
    cellline_expression = torch.tensor(cellline_expression, dtype=torch.float32).to(device)
    cellline2id_dict = name2id_dict[-3]

    mse_list = []
    rmse_list = []
    mae_list = []
    pearson_list = []
    VAL_L0SS=[]
    for _ in range(repeat):
        # def __init__(self, cellline_expression, in_feats, cellline_feats, emd_feats, layer_list, dropout, input_dropout, whether_CCLE=True):
        main_net = DNN_predictor(cellline_expression, all_drug_morgan.shape[1]*2, len(cellline2id_dict), hidden_dim_aux, layer_list, dropout, input_dropout, whether_CCLE)
        main_net.to(device)
        print('main_net structure:', main_net)

        optimizer = torch.optim.Adam(main_net.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        main_net.train()

        tot_params1 = sum([np.prod(p.size()) for p in main_net.parameters()])
        print(f"Total number of parameters in main_net: {tot_params1}")

        if only_test == True:
            temp_prefix = './data/data4training_model/checkpoint/'
            # change it to your trained model
            model_save_path = temp_prefix + 'checkpoint.pt'
        else:
            model_save_path = root_prefix + 'checkpoint/checkpoint_{}.pt'.format(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))

        # every parameter update in the evaluation phase is based on overwriting the old model
        early_stopping = EarlyStopping(patience=patience, verbose=True, save_path=model_save_path)
        # three lists keeping the time of different training phases
        dur1 = []  # data processing before feeding data in an iteration
        dur2 = []  # the training time for an iteration
        dur3 = []  # the time to use grad to update parameters of the model

        train_sample_idx_generator = index_generator(batch_size=batch_size, num_data=len(train_drug_drug_samples))
        val_sample_idx_generator = index_generator(batch_size=batch_size//2, num_data=len(val_drug_drug_samples), shuffle=False)
        test_sample_idx_generator = index_generator(batch_size=batch_size//2, num_data=len(test_drug_drug_samples), shuffle=False)

        te_criterion = torch.nn.MSELoss(reduction='mean')
        print('total epoch number is:',num_epochs)
        if only_test==False:
            for epoch in range(num_epochs):
                t_start = time.time()
                main_net.train()

                for iteration in range(train_sample_idx_generator.num_iterations()):
                    t0 = time.time()

                    # a batch of sample indices for training
                    train_sample_idx_batch = train_sample_idx_generator.next()
                    train_sample_idx_batch.sort()

                    train_drug_drug_batch = train_drug_drug_samples[train_sample_idx_batch].tolist()
                    train_te_labels_batch = train_te_labels[train_sample_idx_batch]
                    train_drug_drug_idx = (np.array(train_drug_drug_batch)[:, :-1].astype(int)).tolist()

                    train_cellline_symbol = (np.array(train_drug_drug_batch)[:, -1]).tolist()
                    train_cellline_idx = [cellline2id_dict[i] for i in train_cellline_symbol] # retrieve the cell line names and transform them to ids in cell line dict

                    train_drug_drug_idx = torch.tensor(train_drug_drug_idx, dtype=torch.int64).to(device)
                    train_cellline_idx = torch.tensor(train_cellline_idx, dtype=torch.int64).to(device) # transform cell line ids to id tensors

                    t1 = time.time()
                    dur1.append(t1 - t0)

                    row_drug_batch, col_drug_batch = train_drug_drug_idx[:, 0], train_drug_drug_idx[:, 1]
                    row_drug_struc_embedding, col_drug_struc_embedding = all_drug_morgan[row_drug_batch], all_drug_morgan[col_drug_batch]

                    te_output = main_net(row_drug_struc_embedding, col_drug_struc_embedding, train_cellline_idx)
                    te_loss = te_criterion(te_output,train_te_labels_batch[:,predicted_te_type].unsqueeze(1))
                    train_total_loss_batch = te_loss

                    t2 = time.time()
                    dur2.append(t2 - t1)
                    # autograd
                    optimizer.zero_grad()
                    train_total_loss_batch.backward()

                    # clip_grad_norm_(self.graph_classifier.parameters(), max_norm=10, norm_type=2)
                    optimizer.step()

                    t3 = time.time()
                    dur3.append(t3 - t2)
                    # print training info
                    if iteration % 100 == 0:
                        print(
                            'Epoch {:05d} | Iteration {:05d} | Train_Loss {:.4f} | Time1(s) {:.4f} | Time2(s) {:.4f} | Time3(s) {:.4f}'.format(
                                epoch, iteration, train_total_loss_batch.item(), np.mean(dur1), np.mean(dur2), np.mean(dur3)))

                # model evaluation
                main_net.eval()
                val_te_loss, val_se_loss, val_total_loss=[],[],[]
                with torch.no_grad():
                    for iteration in range(val_sample_idx_generator.num_iterations()):
                        val_sample_idx_batch = val_sample_idx_generator.next()

                        val_drug_drug_batch = val_drug_drug_samples[val_sample_idx_batch]
                        val_drug_drug_batch_ = val_drug_drug_batch[:,[1,0,2]]

                        val_drug_drug_batch_combined = np.concatenate([val_drug_drug_batch, val_drug_drug_batch_],axis=0).tolist()
                        val_te_labels_batch = val_te_labels[val_sample_idx_batch]

                        val_drug_drug_idx = (np.array(val_drug_drug_batch_combined)[:, :-1].astype(int)).tolist()
                        val_cellline_symbol = (np.array(val_drug_drug_batch_combined)[:, -1]).tolist()
                        val_cellline_idx = [cellline2id_dict[i] for i in val_cellline_symbol]

                        val_drug_drug_idx = torch.tensor(val_drug_drug_idx, dtype=torch.int64).to(device)
                        val_cellline_idx = torch.tensor(val_cellline_idx, dtype=torch.int64).to(device)

                        row_drug_batch, col_drug_batch = val_drug_drug_idx[:, 0], val_drug_drug_idx[:, 1]
                        row_drug_struc_embedding, col_drug_struc_embedding = all_drug_morgan[row_drug_batch], all_drug_morgan[col_drug_batch]
                        te_output = main_net(row_drug_struc_embedding, col_drug_struc_embedding, val_cellline_idx)

                        te_output = (te_output[:te_output.shape[0]//2,:] + te_output[te_output.shape[0]//2:,:])/2
                        te_loss = te_criterion(te_output, val_te_labels_batch[:, predicted_te_type].unsqueeze(1))
                        val_total_loss.append(te_loss)

                    val_total_loss=torch.mean(torch.tensor(val_total_loss))
                    VAL_L0SS.append(val_total_loss.item())
                t_end = time.time()
                print('Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(
                    epoch, val_total_loss.item(), t_end - t_start))

                scheduler.step()
                early_stopping(val_total_loss,
                               {
                                   'main_net': main_net.state_dict(),
                               })
                if early_stopping.early_stop:
                    print('Early stopping based on the validation loss!')
                    break

        # model test
        checkpoint=torch.load(model_save_path)
        main_net.load_state_dict(checkpoint['main_net'])
        main_net.eval()
        test_te_results = []
        test_te_label_list = []
        with torch.no_grad():
            for iteration in range(test_sample_idx_generator.num_iterations()):
                test_sample_idx_batch = test_sample_idx_generator.next()
                test_drug_drug_batch = test_drug_drug_samples[test_sample_idx_batch]
                test_drug_drug_batch_ = test_drug_drug_batch[:,[1,0,2]]
                test_drug_drug_batch_combined = np.concatenate([test_drug_drug_batch,test_drug_drug_batch_],axis=0).tolist()

                test_te_labels_batch = test_te_labels[test_sample_idx_batch]
                test_drug_drug_idx = (np.array(test_drug_drug_batch_combined)[:, :-1].astype(int)).tolist()
                test_cellline_symbol = (np.array(test_drug_drug_batch_combined)[:, -1]).tolist()
                test_cellline_idx = [cellline2id_dict[i] for i in test_cellline_symbol]

                test_drug_drug_idx = torch.tensor(test_drug_drug_idx, dtype=torch.int64).to(device)
                test_cellline_idx = torch.tensor(test_cellline_idx, dtype=torch.int64).to(device)

                row_drug_batch, col_drug_batch = test_drug_drug_idx[:, 0], test_drug_drug_idx[:, 1]
                row_drug_struc_embedding, col_drug_struc_embedding = all_drug_morgan[row_drug_batch], all_drug_morgan[col_drug_batch]
                te_output = main_net(row_drug_struc_embedding, col_drug_struc_embedding, test_cellline_idx)
                te_output = (te_output[:te_output.shape[0] // 2, :] + te_output[te_output.shape[0] // 2:, :]) / 2

                test_te_results.append(te_output)
                test_te_label_list.append(test_te_labels_batch)

            test_te_results = torch.cat(test_te_results)
            test_te_results = test_te_results.cpu().numpy()

            test_te_label_list = torch.cat(test_te_label_list)
            test_te_label_list = test_te_label_list[:,predicted_te_type].unsqueeze(1).cpu().numpy()

        TE_MSE = mean_squared_error(test_te_label_list, test_te_results)
        TE_RMSE = np.sqrt(TE_MSE)
        TE_MAE = mean_absolute_error(test_te_label_list, test_te_results)
        TE_PEARSON = scipy.stats.pearsonr(test_te_label_list.reshape(-1), test_te_results.reshape(-1))

        print('Link Prediction Test')
        print('TE_MSE = {}'.format(TE_MSE))
        print('TE_RMSE = {}'.format(TE_RMSE))
        print('TE_MAE = {}'.format(TE_MAE))
        print('TE_PEARSON and p-value = {},{}'.format(TE_PEARSON[0], TE_PEARSON[1]))

        mse_list.append(TE_MSE)
        rmse_list.append(TE_RMSE)
        mae_list.append(TE_MAE)
        pearson_list.append(TE_PEARSON[0])


    print('----------------------------------------------------------------')
    print('Link Prediction Tests Summary')
    print('MSE_mean = {}, MSE_std = {}'.format(np.mean(mse_list), np.std(mse_list)))
    print('RMSE_mean = {}, RMSE_std = {}'.format(np.mean(rmse_list), np.std(rmse_list)))
    print('MAE_mean = {}, MAE_std = {}'.format(np.mean(mae_list), np.std(mae_list)))
    print('PEARSON_mean = {}, PEARSON_std = {}'.format(np.mean(pearson_list), np.std(pearson_list)))

    # store loss in the evaluation phase
    pd.DataFrame(VAL_L0SS, columns=['VAL_LOSS']).to_csv(root_prefix+'checkpoint/VAL_LOSS.csv')


if __name__ == '__main__':
    # part1
    ap = argparse.ArgumentParser(description='DeepSyngergy testing for drug-drug link prediction')
    ap.add_argument('--root-prefix', type=str,
                    default='D:/B/PROJECT B2_2/dataset/Muthene_dataset/fold1/', # the folder to store the model input for current independent repeat
                    help='root from which to read the orginal input files')
    ap.add_argument('--epoch', type=int, default=20, help='Number of epochs. Default is 5.')
    ap.add_argument('--patience', type=int, default=8, help='Patience. Default is 5.')
    ap.add_argument('--batch-size', type=int, default=32, help='Batch size. Please choose an odd value, because of the way of calculating test labels of our model. Default is 8.')
    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
    # part2
    ap.add_argument('--whether-CCLE', default=[True, True], help='Whether use the real cell line expression data to replace one-hot numerical encoding')
    # whether_CCLE[0]: True: use true cell line expression data. False: use one-hot encoding instead
    # whether_CCLE[1]: True: directly use cell line expression data without dimension reduction ('--hidden-dim-aux' fails) (in DeepSynergy, it should be set to True to keep the model archtectiture described in the original paper)
    # False: use cell line expression data with dimension reduction, the reduced dimension is determined by '--hidden-dim-aux'
    ap.add_argument('--hidden-dim-aux', type=int, default=64,
                    help='Dimension of the node hidden state in the drug model. Default is 64.')
    ap.add_argument('--layer-list', default=[2048, 1024, 512, 1],
                    help='layer neuron units list.')
    ap.add_argument('--dropout', type=float, default=0.5,
                    help='dropout_rate.')
    ap.add_argument('--input-dropout', type=float, default=0.2,
                    help='input_dropout_rate.')

    args = ap.parse_args()

    run_model_DNN_DDI(args.root_prefix, args.epoch, args.patience, args.batch_size, args.repeat, args.hidden_dim_aux, args.layer_list, args.dropout, args.input_dropout, args.whether_CCLE)