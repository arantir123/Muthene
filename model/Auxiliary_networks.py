import torch.nn as nn
import torch.nn.functional as F
import torch
from dgl.nn import GINConv, GraphConv
import dgl


# for generating drug structural embedding
class GIN4drug_struc(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GIN4drug_struc, self).__init__()
        # in_feats应该是原子的种类数(例如10)
        self.embedding = nn.Embedding(in_feats, h_feats)
        self.lin1 = torch.nn.Linear(h_feats, h_feats)
        self.lin2 = torch.nn.Linear(h_feats, h_feats)
        # 原论文中说sum拥有最好性能
        self.conv1 = GINConv(self.lin1, 'sum')
        self.conv2 = GINConv(self.lin2, 'sum')

    def forward(self, g, in_feat):
        # in_feat是一组经过字典转换的原子标签
        # 字典：{5: 0, 6: 1, 7: 2, 8: 3, 9: 4, 15: 5, 16: 6, 17: 7, 35: 8, 79: 9}
        # 转换完的结果：[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], torch.Size([10])
        h = self.embedding(in_feat)
        h = self.conv1(g, h)
        h = F.relu(h)
        h = self.conv2(g, h)
        g.ndata['h'] = h
        return dgl.mean_nodes(g, 'h')


class side_effect_predictor(nn.Module):
    def __init__(self, in_feats, h_feats, dropout_rate=0.0):
        super(side_effect_predictor, self).__init__()
        self.lin1 = torch.nn.Linear(in_feats * 2, h_feats)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = lambda x: x

    def forward(self, drug_embedding1, drug_embedding2):
        input = torch.cat([drug_embedding1, drug_embedding2], axis=1)
        se_output = self.lin1(self.dropout(input))
        return se_output


class DNN_predictor(nn.Module):
    def __init__(self, in_feats, cellline_feats, emd_feats, layer_list, dropout, input_dropout):
        super(DNN_predictor, self).__init__()
        self.linears = nn.ModuleList()
        for i in range(len(layer_list)):
            if i == 0:
                self.linears.append(torch.nn.Linear(in_feats + emd_feats, layer_list[i]))
                self.linears.append(torch.nn.ReLU())
                self.linears.append(nn.Dropout(input_dropout))
            elif i == len(layer_list) - 1:
                self.linears.append(torch.nn.Linear(layer_list[i - 1], layer_list[i]))
            else:
                self.linears.append(torch.nn.Linear(layer_list[i - 1], layer_list[i]))
                self.linears.append(torch.nn.ReLU())
                self.linears.append(nn.Dropout(dropout))
        # for cell line embedding
        self.embedding = nn.Embedding(cellline_feats, emd_feats)

    def forward(self, drug_embedding1, drug_embedding2, cellline_idx, se_labels_batch=None):
        cellline_embedding = self.embedding(cellline_idx)
        if se_labels_batch != None:
            input = torch.cat((drug_embedding1, drug_embedding2, cellline_embedding, se_labels_batch), axis=1)
        else:
            input = torch.cat((drug_embedding1, drug_embedding2, cellline_embedding), axis=1)
        for layer in self.linears:
            input = layer(input)
        return input


class therapeutic_effect_DNN_predictor(nn.Module):
    # cellline_feats是cell line的总个数
    # in_feats是除了cell line数据之外的单个药物输入总维度
    # emd_feats是cell line的embedding维度
    # h_feats是要预测的类别总数，或者要回归的指标总数
    # layer_list是除了第一层之外的隐单元个数
    def __init__(self, cellline_feats, in_feats, emd_feats, layer_list, output_concat=False, concat_feats=0, dropout=0.0, input_dropout=0.0):
        print('TE predictor hyper-paramters:', cellline_feats, in_feats, emd_feats, layer_list, output_concat, concat_feats, dropout, input_dropout)
        super(therapeutic_effect_DNN_predictor, self).__init__()
        self.embedding = nn.Embedding(cellline_feats, emd_feats)
        self.linears = nn.ModuleList()
        for i in range(len(layer_list)):
            if i == 0:
                print('neurons in first layer of TE predictor:', in_feats * 2 + emd_feats + concat_feats)
                if output_concat == True:
                    self.linears.append(torch.nn.Linear(in_feats * 2 + emd_feats + concat_feats, layer_list[i]))
                else:
                    self.linears.append(torch.nn.Linear(in_feats * 2 + emd_feats, layer_list[i]))
                self.linears.append(torch.nn.ReLU())
                self.linears.append(nn.Dropout(input_dropout))
            elif i == len(layer_list) - 1:
                self.linears.append(torch.nn.Linear(layer_list[i - 1], layer_list[i]))
            else:
                self.linears.append(torch.nn.Linear(layer_list[i - 1], layer_list[i]))
                self.linears.append(torch.nn.ReLU())
                self.linears.append(nn.Dropout(dropout))

    def forward(self, drug_embedding1, drug_embedding2, cellline_idx, se_output=None):
        cellline_embedding = self.embedding(cellline_idx)
        if se_output != None:
            input = torch.cat((drug_embedding1, drug_embedding2, cellline_embedding, se_output), axis=1)
        else:
            input = torch.cat((drug_embedding1, drug_embedding2, cellline_embedding), axis=1)
        for layer in self.linears:
            input = layer(input)
        return input


class side_effect_DNN_predictor(nn.Module):
    # h_feats是要预测的类别总数，或者要回归的指标总数
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
