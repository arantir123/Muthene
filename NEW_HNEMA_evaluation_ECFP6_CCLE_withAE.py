# Instead of the framework name in the manuscript (i.e., Muthene), we use HNEMA (Heterogeneous Network Embedding with Meta-path Aggregation) here to define the function.
# Besides, we sincerely thank Fu et al. open the source code of MAGNN at https://github.com/cynricfu/MAGNN. MAGNN helps us to finish message passing of nodes on heterogeneous network.
import time
import argparse
import torch
from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_norm_
import numpy as np
from model.Auxiliary_networks import side_effect_predictor, therapeutic_effect_DNN_predictor
from model.HNEMA_link_prediction import HNEMA_link_prediction
from utils.pytorchtools import EarlyStopping
from utils.data import load_HNEMA_DDI_data_te
from utils.tools import index_generator, parse_minibatch
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import random
import itertools
import pandas as pd
import scipy.stats
import copy


# fix random seed
random_seed = 1024
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
print('random_seed:', random_seed)

# some overall fixed parameters
# drug/target/cell line
num_ntype = 3
# for the main_net
dropout_rate = 0.5
lr = 0.005
weight_decay = 0.001

# the implementation of Muthene using ECFP6 + selected 60 cell lines (described by CCLE gene expression data) with adverse effect module

# the aim of use_masks is to mask drug-drug pairs occurring in the batch, which contains these pairs as the known samples
use_masks = [[False, False, False, True],
             [False, False, False, True]]
# while in val/test set, such masks are not needed
no_masks = [[False] * 4, [False] * 4]

# total numbers of drug and target nodes
num_drug = 106
num_target = 12217
num_cellline = 60

involved_metapaths = [
    [(0, 1, 0), (0, 1, 1, 0), (0, 1, 1, 1, 0), (0, 'te', 0)]]

# for the case that just load model for test
only_test = False

# the type of synergy score to be predicted
# S_mean, synergy_zip, synergy_loewe, synergy_hsa, synergy_bliss (corresponding to 0,1,2,3,4, respectively)
predicted_te_type = 2

def run_model_HNEMA_DDI(root_prefix, hidden_dim_main, num_heads_main, attnvec_dim_main, rnn_type_main,
                        num_epochs, patience, batch_size, neighbor_samples, repeat, attn_switch_main, rnn_concat_main,
                        hidden_dim_aux, loss_ratio_te, loss_ratio_se, layer_list, pred_in_dropout, pred_out_dropout, output_concat, whether_CCLE, args):

    print('loss_ratio_te, loss_ratio_se, output_concat, whether_CCLE, hidden_dim_aux, rnn_type_main:', loss_ratio_te, loss_ratio_se, output_concat, whether_CCLE, hidden_dim_aux, rnn_type_main)
    adjlists_ua, edge_metapath_indices_list_ua, adjM, type_mask, name2id_dict, train_val_test_drug_drug_samples, train_val_test_drug_drug_labels, all_drug_morgan, cellline_expression = load_HNEMA_DDI_data_te(root_prefix)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    features_list = []
    in_dims = []

    # based on type mask, to generate one-hot encoding for each type of nodes (drug/target/cell line) in the heterogeneous network
    for i in range(num_ntype):
        dim = (type_mask == i).sum()
        in_dims.append(dim)
        indices = np.vstack((np.arange(dim), np.arange(dim)))
        indices = torch.LongTensor(indices)
        values = torch.FloatTensor(np.ones(dim))
        features_list.append(torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device))

    # ECFP6 of drugs
    morgan_values = all_drug_morgan.data
    morgan_indices = np.vstack((all_drug_morgan.row, all_drug_morgan.col))
    i = torch.LongTensor(morgan_indices)
    v = torch.FloatTensor(morgan_values)
    shape = all_drug_morgan.shape
    all_drug_morgan = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense().to(device)

    # gene expression data of cell lines
    cellline_expression = torch.tensor(cellline_expression, dtype=torch.float32).to(device)

    loss_ratio_te = torch.tensor(loss_ratio_te, dtype=torch.float32).to(device)
    loss_ratio_se = torch.tensor(loss_ratio_se, dtype=torch.float32).to(device)

    train_drug_drug_samples = train_val_test_drug_drug_samples['train_drug_drug_samples']

    # scaler = MinMaxScaler()
    train_te_temp_labels = train_val_test_drug_drug_labels['train_te_labels'][:, predicted_te_type].reshape(-1,1)
    # scaler.fit(train_te_temp_labels)
    # train_te_temp_labels = scaler.transform(train_te_temp_labels)
    train_te_labels = torch.tensor(train_te_temp_labels, dtype=torch.float32).to(device)
    train_se_labels = torch.tensor(train_val_test_drug_drug_labels['train_se_labels'],dtype=torch.float32).to(device)

    # an extra test about exchanging the val and test set
    val_drug_drug_samples = train_val_test_drug_drug_samples['val_drug_drug_samples']
    test_drug_drug_samples = train_val_test_drug_drug_samples['test_drug_drug_samples']

    val_te_temp_labels = train_val_test_drug_drug_labels['val_te_labels'][:, predicted_te_type].reshape(-1, 1)
    # test_te_temp_labels = scaler.transform(test_te_temp_labels)
    val_te_labels = torch.tensor(val_te_temp_labels,dtype=torch.float32).to(device)
    val_se_labels = torch.tensor(train_val_test_drug_drug_labels['val_se_labels'],dtype=torch.float32).to(device)

    test_te_temp_labels = train_val_test_drug_drug_labels['test_te_labels'][:, predicted_te_type].reshape(-1, 1)
    # val_te_temp_labels = scaler.transform(val_te_temp_labels)
    test_te_labels = torch.tensor(test_te_temp_labels,dtype=torch.float32).to(device)
    test_se_labels = torch.tensor(train_val_test_drug_drug_labels['test_se_labels'],dtype=torch.float32).to(device)

    # atomnum2id_dict = name2id_dict[-1]
    se_symbol2id_dict = name2id_dict[-2]
    cellline2id_dict = name2id_dict[-3]

    mse_list = []
    rmse_list = []
    mae_list = []
    pearson_list = []
    VAL_L0SS=[]
    for _ in range(repeat):
        main_net = HNEMA_link_prediction(
            [4], in_dims[:-1], hidden_dim_main, hidden_dim_main, num_heads_main, attnvec_dim_main, rnn_type_main,
            dropout_rate, attn_switch_main, rnn_concat_main, args)
        main_net.to(device)

        # def __init__(self, in_feats, h_feats, dropout_rate=0.0)
        se_net = side_effect_predictor(hidden_dim_main + all_drug_morgan.shape[1], len(se_symbol2id_dict))
        se_net.to(device)

        te_layer_list = copy.deepcopy(layer_list)
        te_layer_list.append(1)
        print('The hidden unit number for each layer in TE prediction:', te_layer_list)
        # def __init__(self, cellline_expression, cellline_feats, in_feats, emd_feats, layer_list, output_concat=False, concat_feats=0, dropout=0.0, input_dropout=0.0, whether_CCLE=True)
        te_net = therapeutic_effect_DNN_predictor(cellline_expression, len(cellline2id_dict),
                                                  hidden_dim_main + all_drug_morgan.shape[1], hidden_dim_aux, te_layer_list,
                                                  output_concat, len(se_symbol2id_dict), pred_out_dropout, pred_in_dropout, whether_CCLE)

        te_net.to(device)
        sigmoid = torch.nn.Sigmoid()
        print('te_net structure:', te_net)

        # optimizer = torch.optim.SGD(
        optimizer = torch.optim.Adam(
            itertools.chain(main_net.parameters(), te_net.parameters(), se_net.parameters()),
            lr=lr, weight_decay=weight_decay)

        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        main_net.train()
        se_net.train()
        te_net.train()

        tot_params1 = sum([np.prod(p.size()) for p in main_net.parameters()])
        print(f"Total number of parameters in main_net: {tot_params1}")
        tot_params2 = sum([np.prod(p.size()) for p in se_net.parameters()])
        print(f"Total number of parameters in se_net: {tot_params2}")
        tot_params3 = sum([np.prod(p.size()) for p in te_net.parameters()])
        print(f"Total number of parameters in te_net: {tot_params3}")
        print(f"Total number of parameters in Muthene: {tot_params1 + tot_params2 + tot_params3}")

        if only_test == True:
            temp_prefix = './data/data4training_model/checkpoint/'
            # change it to your trained model
            model_save_path = temp_prefix + 'checkpoint.pt'
        else:
            model_save_path = root_prefix + 'checkpoint/checkpoint_{}.pt'.format(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))

        early_stopping = EarlyStopping(patience=patience, verbose=True, save_path=model_save_path)
        # three lists keeping the time of different training phases
        dur1 = []  # data processing before feeding data in an iteration
        dur2 = []  # the training time for an iteration
        dur3 = []  # the time to use grad to update parameters of the model

        train_sample_idx_generator = index_generator(batch_size=batch_size, num_data=len(train_drug_drug_samples))
        # reason for batch_size=batch_size//2: to generate the drug-drug pairs with the opposite drug order in val/test phases
        val_sample_idx_generator = index_generator(batch_size=batch_size//2, num_data=len(val_drug_drug_samples), shuffle=False)
        test_sample_idx_generator = index_generator(batch_size=batch_size//2, num_data=len(test_drug_drug_samples), shuffle=False)

        te_criterion = torch.nn.MSELoss(reduction='mean')
        se_criterion = torch.nn.BCELoss(reduction='mean')

        print('total epoch number is:',num_epochs)
        print('current loss_ratio_te and loss_ratio_se are:', loss_ratio_te, loss_ratio_se)
        if only_test == False:
            for epoch in range(num_epochs):
                t_start = time.time()
                main_net.train()
                se_net.train()
                te_net.train()

                for iteration in range(train_sample_idx_generator.num_iterations()):
                    t0 = time.time()

                    train_sample_idx_batch = train_sample_idx_generator.next()
                    train_sample_idx_batch.sort()

                    train_drug_drug_batch = train_drug_drug_samples[train_sample_idx_batch].tolist()
                    train_te_labels_batch = train_te_labels[train_sample_idx_batch]
                    train_se_labels_batch = train_se_labels[train_sample_idx_batch]
                    train_drug_drug_idx = (np.array(train_drug_drug_batch)[:, :-1].astype(int)).tolist()

                    train_cellline_symbol = (np.array(train_drug_drug_batch)[:, -1]).tolist()
                    train_cellline_idx = [cellline2id_dict[i] for i in train_cellline_symbol]

                    train_g_lists, train_indices_lists, train_idx_batch_mapped_lists = parse_minibatch(adjlists_ua, edge_metapath_indices_list_ua, train_drug_drug_idx, device, neighbor_samples,use_masks, num_drug)

                    t1 = time.time()
                    dur1.append(t1 - t0)

                    [row_drug_embedding, col_drug_embedding], _, [row_drug_atten, col_drug_atten] = main_net((train_g_lists, features_list, type_mask[:num_drug + num_target], train_indices_lists, train_idx_batch_mapped_lists))

                    train_drug_drug_idx = torch.tensor(train_drug_drug_idx, dtype=torch.int64).to(device)
                    train_cellline_idx = torch.tensor(train_cellline_idx, dtype=torch.int64).to(device)
                    row_drug_batch, col_drug_batch = train_drug_drug_idx[:, 0], train_drug_drug_idx[:, 1]
                    row_drug_struc_embedding, col_drug_struc_embedding = all_drug_morgan[row_drug_batch], all_drug_morgan[col_drug_batch]

                    row_drug_composite_embedding = torch.cat((row_drug_embedding, row_drug_struc_embedding), axis=1)
                    col_drug_composite_embedding = torch.cat((col_drug_embedding, col_drug_struc_embedding), axis=1)

                    se_output = sigmoid(se_net(row_drug_composite_embedding, col_drug_composite_embedding))
                    if output_concat==True:
                        se_output_ = se_output.clone().detach()
                        te_output = te_net(row_drug_composite_embedding, col_drug_composite_embedding, train_cellline_idx, se_output_)
                    else:
                        te_output = te_net(row_drug_composite_embedding, col_drug_composite_embedding, train_cellline_idx)

                    te_loss = te_criterion(te_output, train_te_labels_batch)
                    se_loss = se_criterion(se_output, train_se_labels_batch)
                    train_total_loss_batch = loss_ratio_te * te_loss + loss_ratio_se * se_loss

                    t2 = time.time()
                    dur2.append(t2 - t1)
                    # autograd
                    optimizer.zero_grad()
                    train_total_loss_batch.backward()
                    # clip_grad_norm_(itertools.chain(main_net.parameters(), drug_net.parameters(), te_net.parameters(), se_net.parameters()), max_norm=10, norm_type=2)
                    optimizer.step()
                    t3 = time.time()
                    dur3.append(t3 - t2)
                    if iteration % 100 == 0:
                        print(
                            'Epoch {:05d} | Iteration {:05d} | Train_Loss {:.4f} | Time1(s) {:.4f} | Time2(s) {:.4f} | Time3(s) {:.4f}'.format(
                                epoch, iteration, train_total_loss_batch.item(), np.mean(dur1), np.mean(dur2), np.mean(dur3)))

                # model evaluation
                main_net.eval()
                se_net.eval()
                te_net.eval()
                val_te_loss, val_se_loss, val_total_loss=[],[],[]
                with torch.no_grad():
                    for iteration in range(val_sample_idx_generator.num_iterations()):
                        val_sample_idx_batch = val_sample_idx_generator.next()
                        val_drug_drug_batch = val_drug_drug_samples[val_sample_idx_batch]
                        # for generating drug-drug pairs with the opposite drug order
                        val_drug_drug_batch_ = val_drug_drug_batch[:,[1,0,2]]
                        val_drug_drug_batch_combined = np.concatenate([val_drug_drug_batch,val_drug_drug_batch_],axis=0).tolist()

                        val_te_labels_batch = val_te_labels[val_sample_idx_batch]
                        val_se_labels_batch = val_se_labels[val_sample_idx_batch]
                        val_drug_drug_idx = (np.array(val_drug_drug_batch_combined)[:, :-1].astype(int)).tolist()
                        val_cellline_symbol = (np.array(val_drug_drug_batch_combined)[:, -1]).tolist()
                        val_cellline_idx = [cellline2id_dict[i] for i in val_cellline_symbol]

                        val_g_lists, val_indices_lists, val_idx_batch_mapped_lists = parse_minibatch(adjlists_ua, edge_metapath_indices_list_ua, val_drug_drug_idx, device, neighbor_samples, no_masks, num_drug)

                        [row_drug_embedding, col_drug_embedding], _, [row_drug_atten, col_drug_atten] = main_net((val_g_lists, features_list, type_mask[:num_drug + num_target], val_indices_lists, val_idx_batch_mapped_lists))

                        val_drug_drug_idx = torch.tensor(val_drug_drug_idx, dtype=torch.int64).to(device)
                        val_cellline_idx = torch.tensor(val_cellline_idx, dtype=torch.int64).to(device)
                        row_drug_batch, col_drug_batch = val_drug_drug_idx[:, 0], val_drug_drug_idx[:, 1]
                        row_drug_struc_embedding, col_drug_struc_embedding = all_drug_morgan[row_drug_batch], all_drug_morgan[col_drug_batch]

                        row_drug_composite_embedding = torch.cat((row_drug_embedding, row_drug_struc_embedding), axis=1)
                        col_drug_composite_embedding = torch.cat((col_drug_embedding, col_drug_struc_embedding), axis=1)

                        se_output = sigmoid(se_net(row_drug_composite_embedding, col_drug_composite_embedding))
                        if output_concat == True:
                            se_output_ = se_output.clone().detach()
                            te_output = te_net(row_drug_composite_embedding, col_drug_composite_embedding, val_cellline_idx, se_output_)
                        else:
                            te_output = te_net(row_drug_composite_embedding, col_drug_composite_embedding, val_cellline_idx)

                        # calculate the averaging results of the drug pairs with the opposite drug order
                        se_output = (se_output[:se_output.shape[0]//2,:] + se_output[se_output.shape[0]//2:,:])/2
                        te_output = (te_output[:te_output.shape[0]//2,:] + te_output[te_output.shape[0]//2:,:])/2
                        te_loss = te_criterion(te_output, val_te_labels_batch)
                        se_loss = se_criterion(se_output, val_se_labels_batch)
                        val_total_loss.append(loss_ratio_te * te_loss + loss_ratio_se * se_loss)

                    val_total_loss=torch.mean(torch.tensor(val_total_loss))
                    VAL_L0SS.append(val_total_loss.item())
                t_end = time.time()
                print('Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(
                    epoch, val_total_loss.item(), t_end - t_start))

                scheduler.step()
                early_stopping(val_total_loss,
                               {
                                   'main_net': main_net.state_dict(),
                                   'se_net': se_net.state_dict(),
                                   'te_net': te_net.state_dict()
                               })
                if early_stopping.early_stop:
                    print('Early stopping based on the validation loss!')
                    break

        # model test
        print('The name of loaded model is:', model_save_path)
        checkpoint=torch.load(model_save_path)
        main_net.load_state_dict(checkpoint['main_net'])
        se_net.load_state_dict(checkpoint['se_net'])
        te_net.load_state_dict(checkpoint['te_net'])

        main_net.eval()
        se_net.eval()
        te_net.eval()
        test_te_results, test_se_results = [], []
        test_te_label_list, test_se_label_list = [], []
        with torch.no_grad():
            for iteration in range(test_sample_idx_generator.num_iterations()):
                test_sample_idx_batch = test_sample_idx_generator.next()
                test_drug_drug_batch = test_drug_drug_samples[test_sample_idx_batch]
                test_drug_drug_batch_ = test_drug_drug_batch[:,[1,0,2]] # also test the drug-drug-cell line pairs with the reverse direction of drug pairs
                test_drug_drug_batch_combined = np.concatenate([test_drug_drug_batch,test_drug_drug_batch_],axis=0).tolist()

                test_te_labels_batch = test_te_labels[test_sample_idx_batch]
                test_se_labels_batch = test_se_labels[test_sample_idx_batch]
                test_drug_drug_idx = (np.array(test_drug_drug_batch_combined)[:, :-1].astype(int)).tolist()
                test_cellline_symbol = (np.array(test_drug_drug_batch_combined)[:, -1]).tolist()
                test_cellline_idx = [cellline2id_dict[i] for i in test_cellline_symbol]

                test_g_lists, test_indices_lists, test_idx_batch_mapped_lists = parse_minibatch(
                    adjlists_ua, edge_metapath_indices_list_ua, test_drug_drug_idx, device, neighbor_samples,
                    no_masks, num_drug)

                [row_drug_embedding, col_drug_embedding], _, [row_drug_atten, col_drug_atten] = main_net((test_g_lists, features_list, type_mask[:num_drug + num_target], test_indices_lists, test_idx_batch_mapped_lists))

                test_drug_drug_idx = torch.tensor(test_drug_drug_idx, dtype=torch.int64).to(device)
                test_cellline_idx = torch.tensor(test_cellline_idx, dtype=torch.int64).to(device)
                row_drug_batch, col_drug_batch = test_drug_drug_idx[:, 0], test_drug_drug_idx[:, 1]
                row_drug_struc_embedding, col_drug_struc_embedding = all_drug_morgan[row_drug_batch], all_drug_morgan[col_drug_batch]

                row_drug_composite_embedding = torch.cat((row_drug_embedding, row_drug_struc_embedding), axis=1)
                col_drug_composite_embedding = torch.cat((col_drug_embedding, col_drug_struc_embedding), axis=1)

                se_output = sigmoid(se_net(row_drug_composite_embedding, col_drug_composite_embedding))
                if output_concat == True:
                    se_output_ = se_output.clone().detach()
                    te_output = te_net(row_drug_composite_embedding, col_drug_composite_embedding, test_cellline_idx, se_output_)
                else:
                    te_output = te_net(row_drug_composite_embedding, col_drug_composite_embedding, test_cellline_idx)

                se_output = (se_output[:se_output.shape[0]//2,:] + se_output[se_output.shape[0]//2:,:])/2
                te_output = (te_output[:te_output.shape[0]//2,:] + te_output[te_output.shape[0]//2:,:])/2
                test_te_results.append(te_output)
                test_te_label_list.append(test_te_labels_batch)
                test_se_results.append(se_output)
                test_se_label_list.append(test_se_labels_batch)

            test_te_results = torch.cat(test_te_results)
            test_te_results = test_te_results.cpu().numpy()
            # test_te_results = scaler.inverse_transform(test_te_results)
            test_se_results = torch.cat(test_se_results)
            test_se_results = test_se_results.cpu().numpy()

            test_te_label_list = torch.cat(test_te_label_list)
            test_te_label_list = test_te_label_list.cpu().numpy()
            # test_te_label_list = scaler.inverse_transform(test_te_label_list)
            test_se_label_list = torch.cat(test_se_label_list)
            test_se_label_list = test_se_label_list.cpu().numpy()

        print('the size of test_te_results, test_se_results:', test_te_results.shape, test_se_results.shape)
        print('the size of test_te_label_list, test_se_label_list:', test_te_label_list.shape, test_se_label_list.shape)
        TE_MSE = mean_squared_error(test_te_label_list, test_te_results)
        TE_RMSE = np.sqrt(TE_MSE)
        TE_MAE = mean_absolute_error(test_te_label_list, test_te_results)
        # coefficient and 2-tailed p-value
        TE_PEARSON = scipy.stats.pearsonr(test_te_label_list.reshape(-1), test_te_results.reshape(-1))

        print('Link Prediction Test')
        print('TE_MSE = {}'.format(TE_MSE))
        print('TE_RMSE = {}'.format(TE_RMSE))
        print('TE_MAE = {}'.format(TE_MAE))
        print('TE_PEARSON and p-value = {},{}'.format(TE_PEARSON[0],TE_PEARSON[1]))

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

    pd.DataFrame(VAL_L0SS, columns=['VAL_LOSS']).to_csv(
        root_prefix+'checkpoint/VAL_LOSS.csv')


# do some research on the component features of the TE predictor, some potential operations for improving performance:
# 1. add the drug embedding generated by meta-paths to a relatively large value
# 2. do not use Linear to process cell line expression data
# 3. use different initialization for linear layers


if __name__ == '__main__':
    # part1 (for meta-path embedding generation)
    ap = argparse.ArgumentParser(description='Muthene testing for drug-drug link prediction')
    ap.add_argument('--root-prefix', type=str,
                    default='C:/Users/Arantir/Desktop/Muthene_dataset/fold1/', # the folder to store the model input for current independent repeat
                    help='root from which to read the original input files')
    ap.add_argument('--hidden-dim-main', type=int, default=64,
                    help='Dimension of the node hidden state in the main model. Default is 64.')
    ap.add_argument('--num-heads-main', type=int, default=8,
                    help='Number of the attention heads in the main model. Default is 8.')
    ap.add_argument('--attnvec-dim-main', type=int, default=128,
                    help='Dimension of the attention vector in the main model. Default is 128.')
    ap.add_argument('--rnn-type-main', default='bi-gru',
                    help='Type of the aggregator in the main model. Default is bi-gru.')
    ap.add_argument('--epoch', type=int, default=20, help='Number of epochs. Default is 20.')
    ap.add_argument('--patience', type=int, default=8, help='Patience. Default is 8.')
    ap.add_argument('--batch-size', type=int, default=32,
                    help='Batch size. Please choose an odd value, because of the way of calculating val/test labels of our model. Default is 32.')
    ap.add_argument('--samples', type=int, default=100,
                    help='Number of neighbors sampled in the parse function of main model. Default is 100.')
    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
    # if it is set to False, the GAT layer will ignore the feature of the central node itself
    ap.add_argument('--attn-switch-main', default=True,
                    help='whether need to consider the feature of the central node when using GAT layer in the main model')
    ap.add_argument('--rnn-concat-main', default=False,
                    help='whether need to concat the feature extracted from rnn with the embedding from GAT layer in the main model')
    # part2 (for other modules in HNEMA)
    ap.add_argument('--whether-CCLE', default=[True, False],
                    help='Whether use the real cell line expression data to replace one-hot numerical encoding')
    # whether_CCLE[0]: True: use true cell line expression data. False: use one-hot encoding instead
    # whether_CCLE[1]: True: directly use cell line expression data without dimension reduction ('--hidden-dim-aux' fails)
    # False: use cell line expression data with dimension reduction, the reduced dimension is determined by '--hidden-dim-aux'
    ap.add_argument('--hidden-dim-aux', type=int, default=64,
                    help='Dimension of generated cell line embeddings. Default is 64.')
    # loss-ratio-te is a relatively sensitive important hyper-parameter in our experiments, 1 or 10 usually generates good results
    ap.add_argument('--loss-ratio-te', type=float, default=1,
                    help='The weight percentage of therapeutic effect loss in the total loss')
    ap.add_argument('--loss-ratio-se', type=float, default=1,
                    help='The weight percentage of adverse effect loss in the total loss')
    ap.add_argument('--layer-list', default=[2048, 1024, 512], # default = [2048, 1024, 512]
                    help='layer neuron units list for the DNN TE predictor.')
    ap.add_argument('--pred_in_dropout', type=float, default=0.2,
                    help='The input dropout rate of the DNN TE predictor')
    ap.add_argument('--pred_out_dropout', type=float, default=0.5,
                    help='The output dropout rate of the DNN TE predictor')
    ap.add_argument('--output_concat', default=True,
                    help='Whether put the adverse effect output into therapeutiec effect prediction')

    args = ap.parse_args()
    run_model_HNEMA_DDI(args.root_prefix, args.hidden_dim_main, args.num_heads_main, args.attnvec_dim_main, args.rnn_type_main, args.epoch,
                        args.patience, args.batch_size, args.samples, args.repeat, args.attn_switch_main, args.rnn_concat_main, args.hidden_dim_aux,
                        args.loss_ratio_te, args.loss_ratio_se, args.layer_list, args.pred_in_dropout, args.pred_out_dropout, args.output_concat, args.whether_CCLE, args)
