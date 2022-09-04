import torch.nn as nn
import torch.nn.functional as F
import torch
from dgl.nn import GINConv, GraphConv
import dgl


# for generating drug structural embedding
class GIN4drug_struc(nn.Module):

    def __init__(self, in_feats, h_feats):
        super(GIN4drug_struc, self).__init__()
        # in_feats: total number of atom types
        self.embedding = nn.Embedding(in_feats, h_feats)
        self.lin1 = torch.nn.Linear(h_feats, h_feats)
        self.lin2 = torch.nn.Linear(h_feats, h_feats)
        self.conv1 = GINConv(self.lin1, 'sum')
        self.conv2 = GINConv(self.lin2, 'sum')

    def forward(self, g, in_feat):
        # indices for retrieving embeddings
        h = self.embedding(in_feat)
        h = self.conv1(g, h)
        h = F.relu(h)
        h = self.conv2(g, h)
        g.ndata['h'] = h
        return dgl.mean_nodes(g, 'h')


class side_effect_predictor(nn.Module):

    def __init__(self, in_feats, h_feats, dropout_rate=0.0):
        # in_feats: dimension of drug embedding (from ECFP6 + from DTI network)
        # h_feats: number of side effects
        super(side_effect_predictor, self).__init__()
        self.lin1 = torch.nn.Linear(in_feats * 2, h_feats)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = lambda x: x

        # *** could try extra initialization for all linear layers here ***

    def forward(self, drug_embedding1, drug_embedding2):
        input = torch.cat([drug_embedding1, drug_embedding2], axis=1)
        se_output = self.lin1(self.dropout(input))
        return se_output


class DNN_predictor(nn.Module):
    # cellline_feats: only for generating the cell line look-up table under the case whether_CCLE=[False, False]
    # in_feats: total embedding dimension for a drug-drug pair based on ECFP6 (also could include an adverse effect label dimension)
    def __init__(self, cellline_expression, in_feats, cellline_feats, emd_feats, layer_list, dropout, input_dropout, whether_CCLE=[True, True]):
        print('DNN predictor hyper-paramters:', in_feats, cellline_feats, emd_feats, layer_list, dropout, input_dropout, whether_CCLE)
        super(DNN_predictor, self).__init__()
        self.whether_CCLE = whether_CCLE
        self.emd_feats = emd_feats

        # cell line encoding:
        # whether_CCLE[0]: True: use true cell line expression data. False: use one-hot encoding instead
        # whether_CCLE[1]:
        # True: directly use cell line expression data without dimension reduction ('--hidden-dim-aux' fails)
        # False: use cell line expression data with dimension reduction, the reduced dimension is determined by '--hidden-dim-aux'

        if self.whether_CCLE[0] == True:
            self.cellline_expression = cellline_expression
            self.emd_feats = self.cellline_expression.size(1)
            if self.whether_CCLE[1] == False:
                self.emd_feats = emd_feats
                self.cellline_transform = nn.Linear(self.cellline_expression.size(1), self.emd_feats, bias=True) # for feature reduction
        else:
            self.cellline_transform = nn.Embedding(cellline_feats, self.emd_feats) # cell line number --> len(cellline2id_dict) * cell line dimension

        self.linears = nn.ModuleList()
        for i in range(len(layer_list)):
            if i == 0:
                print('Neurons in first layer of DNN predictor:', in_feats + self.emd_feats, 'cell line dimension:', self.emd_feats)
                self.linears.append(torch.nn.Linear(in_feats + self.emd_feats, layer_list[i]))
                self.linears.append(torch.nn.ReLU())
                self.linears.append(nn.Dropout(input_dropout))
            elif i == len(layer_list) - 1:
                self.linears.append(torch.nn.Linear(layer_list[i - 1], layer_list[i]))
            else:
                self.linears.append(torch.nn.Linear(layer_list[i - 1], layer_list[i]))
                self.linears.append(torch.nn.ReLU())
                self.linears.append(nn.Dropout(dropout))

    def forward(self, drug_embedding1, drug_embedding2, cellline_idx, se_labels_batch=None):
        # cell line encoding, cellline_idx: cell line ids of a batch based on the cell line dict
        if (self.whether_CCLE[0] == True) and (self.whether_CCLE[1] == True):
            # use cell line information directly
            cellline_embedding = self.cellline_expression[cellline_idx]
        elif (self.whether_CCLE[0] == True) and (self.whether_CCLE[1] == False):
            cellline_embedding_encoding = self.cellline_transform(self.cellline_expression)
            cellline_embedding = cellline_embedding_encoding[cellline_idx]
        else:
            cellline_embedding = self.cellline_transform(cellline_idx)

        if se_labels_batch != None: # in the case for fusing adverse effect labels
            input = torch.cat((drug_embedding1, drug_embedding2, cellline_embedding, se_labels_batch), axis=1)
        else:
            input = torch.cat((drug_embedding1, drug_embedding2, cellline_embedding), axis=1)

        for layer in self.linears:
            input = layer(input)
        return input


class therapeutic_effect_DNN_predictor(nn.Module):
    # cellline_feats: only for generating the cell line look-up table under the case whether_CCLE=[False, False]. emd_feats: cell line defined dimension
    # in_feats: total embedding dimension for one drug (ECFP6 + metapath embedding)
    # layer_list: hidden unit number for each layer (except for input feature number)
    # control whether to concatenate adverse effect outputs
    def __init__(self, cellline_expression, cellline_feats, in_feats, emd_feats, layer_list, output_concat=False, concat_feats=0, dropout=0.0, input_dropout=0.0, whether_CCLE=[True, True]):
        print('TE predictor hyper-paramters:', cellline_feats, in_feats, emd_feats, layer_list, output_concat, concat_feats, dropout, input_dropout, whether_CCLE)
        super(therapeutic_effect_DNN_predictor, self).__init__()
        self.whether_CCLE = whether_CCLE
        self.emd_feats = emd_feats

        # cell line encoding:
        # whether_CCLE[0]: True: use true cell line expression data. False: use one-hot encoding instead
        # whether_CCLE[1]:
        # True: directly use cell line expression data without dimension reduction ('--hidden-dim-aux' fails)
        # False: use cell line expression data with dimension reduction, the reduced dimension is determined by '--hidden-dim-aux'

        # need to explain how to leverage cell line related information in detail
        if self.whether_CCLE[0] == True:
            self.cellline_expression = cellline_expression
            self.emd_feats = self.cellline_expression.size(1)
            if self.whether_CCLE[1] == False:
                self.emd_feats = emd_feats
                self.cellline_transform = nn.Linear(self.cellline_expression.size(1), self.emd_feats, bias=True) # for feature reduction
        else:
            self.cellline_transform = nn.Embedding(cellline_feats, self.emd_feats) # cell line number * cell line dimension

        # drug-drug-cell line pair encoding:
        self.linears = nn.ModuleList()
        for i in range(len(layer_list)):
            if i == 0: # the first layer
                if output_concat == True:
                    print('Neurons in first layer of TE predictor:', in_feats * 2 + self.emd_feats + concat_feats, 'cell line dimension:', self.emd_feats)
                    self.linears.append(torch.nn.Linear(in_feats * 2 + self.emd_feats + concat_feats, layer_list[i]))
                else:
                    print('Neurons in first layer of TE predictor:', in_feats * 2 + self.emd_feats, 'cell line dimension:', self.emd_feats)
                    self.linears.append(torch.nn.Linear(in_feats * 2 + self.emd_feats, layer_list[i]))
                self.linears.append(torch.nn.ReLU())
                self.linears.append(nn.Dropout(input_dropout))
            elif i == len(layer_list) - 1: # the last layer
                self.linears.append(torch.nn.Linear(layer_list[i - 1], layer_list[i]))
            else: # the intermediate layers
                self.linears.append(torch.nn.Linear(layer_list[i - 1], layer_list[i]))
                self.linears.append(torch.nn.ReLU())
                self.linears.append(nn.Dropout(dropout))

        # *** could try extra initialization for all linear layers here ***
        # for fc in self.linears:
        #     if isinstance(fc, nn.Linear):
        #         nn.init.xavier_normal_(fc.weight, gain=1.414)
        # if (self.whether_CCLE[0] == True) and (self.whether_CCLE[1] == False):
        #     nn.init.xavier_normal_(self.cellline_transform.weight, gain=1.414)

    def forward(self, drug_embedding1, drug_embedding2, cellline_idx, se_output=None):
        # cell line encoding
        if (self.whether_CCLE[0] == True) and (self.whether_CCLE[1] == True):
            # use cell line information directly
            cellline_embedding = self.cellline_expression[cellline_idx]
        elif (self.whether_CCLE[0] == True) and (self.whether_CCLE[1] == False):
            cellline_embedding_encoding = self.cellline_transform(self.cellline_expression)
            cellline_embedding = cellline_embedding_encoding[cellline_idx]
        else:
            cellline_embedding = self.cellline_transform(cellline_idx)

        # feature concatenation
        if se_output != None:
            input = torch.cat((drug_embedding1, drug_embedding2, cellline_embedding, se_output), axis=1)
        else:
            input = torch.cat((drug_embedding1, drug_embedding2, cellline_embedding), axis=1)

        # drug-drug-cell line pair encoding
        for layer in self.linears:
            input = layer(input)
        return input


class side_effect_DNN_predictor(nn.Module):

    def __init__(self, in_feats, layer_list, dropout=0.0, input_dropout=0.0):
        super(side_effect_DNN_predictor, self).__init__()
        self.linears = nn.ModuleList()
        for i in range(len(layer_list)):
            if i == 0:
                self.linears.append(torch.nn.Linear(in_feats * 2, layer_list[i]))
                self.linears.append(torch.nn.ReLU())
                self.linears.append(nn.Dropout(input_dropout))
            elif i == len(layer_list) - 1:
                self.linears.append(torch.nn.Linear(layer_list[i - 1], layer_list[i]))
            else:
                self.linears.append(torch.nn.Linear(layer_list[i - 1], layer_list[i]))
                self.linears.append(torch.nn.ReLU())
                self.linears.append(nn.Dropout(dropout))

    def forward(self, drug_embedding1, drug_embedding2):
        input = torch.cat([drug_embedding1, drug_embedding2], axis=1)
        for layer in self.linears:
            input = layer(input)
        return input


# for automatically balancing weights of two tasks
class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int,the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum
