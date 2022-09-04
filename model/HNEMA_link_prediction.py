import torch
import torch.nn as nn
import numpy as np
from model.base_HNEMA import HNEMA_ctr_ntype_specific, HNEMA_ctr_ntype_specific_transformer


# for link prediction task
class HNEMA_lp_layer(nn.Module):
    def __init__(self,
                 num_metapaths_list,
                 in_dim,
                 out_dim,
                 num_heads,
                 attn_vec_dim,
                 rnn_type='bi-gru',
                 attn_drop=0.5,
                 attn_switch=False,
                 rnn_concat=False):
        super(HNEMA_lp_layer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads

        # drug/target specific layers
        self.drug_layer = HNEMA_ctr_ntype_specific(num_metapaths_list[0],
                                                   in_dim,
                                                   num_heads,
                                                   attn_vec_dim,
                                                   rnn_type,
                                                   attn_drop,
                                                   use_minibatch=True,
                                                   attn_switch=attn_switch,
                                                   rnn_concat=rnn_concat)

        # note that the actual input dimension should consider the number of heads as multiple head outputs are concatenated together
        if (rnn_concat == True):
            self.fc_drug = nn.Linear(in_dim * num_heads * 2, out_dim, bias=True)
            # self.fc_target = nn.Linear(in_dim * num_heads * 2, out_dim, bias=True)
        else:
            self.fc_drug = nn.Linear(in_dim * num_heads, out_dim, bias=True)
            # self.fc_target = nn.Linear(in_dim * num_heads, out_dim, bias=True)

        nn.init.xavier_normal_(self.fc_drug.weight, gain=1.414)
        # nn.init.xavier_normal_(self.fc_target.weight, gain=1.414)

    def forward(self, inputs):
        g_lists, features, type_mask, edge_metapath_indices_lists, target_idx_lists = inputs
        # drug/target specific layers
        h_drug1, atten_drug1 = self.drug_layer(
            (g_lists[0], features, type_mask, edge_metapath_indices_lists[0], target_idx_lists[0]))
        h_drug2, atten_drug2 = self.drug_layer(
            (g_lists[1], features, type_mask, edge_metapath_indices_lists[1], target_idx_lists[1]))

        logits_drug1 = self.fc_drug(h_drug1)
        logits_drug2 = self.fc_drug(h_drug2)
        return [logits_drug1, logits_drug2], [h_drug1, h_drug2], [atten_drug1, atten_drug2]


class HNEMA_link_prediction(nn.Module):
    def __init__(self,
                 num_metapaths_list,
                 feats_dim_list,
                 hidden_dim,
                 out_dim,
                 num_heads,
                 attn_vec_dim,
                 rnn_type='bi-gru',
                 dropout_rate=0.5,
                 attn_switch=False,
                 rnn_concat=False,
                 args=None):
        super(HNEMA_link_prediction, self).__init__()
        self.hidden_dim = hidden_dim
        self.args = args

        # node type specific transformation
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True) for feats_dim in feats_dim_list])
        # feature dropout after transformation
        if dropout_rate > 0:
            self.feat_drop = nn.Dropout(dropout_rate)
        else:
            self.feat_drop = lambda x: x
        # initialization of fc layers
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        # HNEMA_lp layers
        self.layer = HNEMA_lp_layer(num_metapaths_list,
                                     hidden_dim,
                                     out_dim,
                                     num_heads,
                                     attn_vec_dim,
                                     rnn_type,
                                     attn_drop=dropout_rate,
                                     attn_switch=attn_switch,
                                     rnn_concat=rnn_concat)

    def forward(self, inputs):
        g_lists, features_list, type_mask, edge_metapath_indices_lists, target_idx_lists = inputs

        # node type specific transformation
        transformed_features = torch.zeros(type_mask.shape[0], self.hidden_dim, device=features_list[0].device)
        for i, fc in enumerate(self.fc_list):
            node_indices = np.where(type_mask == i)[0]
            transformed_features[node_indices] = fc(features_list[i])
        # create a matrix storing all node features of the dataset
        transformed_features = self.feat_drop(transformed_features)

        # hidden layers
        [logits_drug1, logits_drug2], [h_drug1, h_drug2], [atten_drug1, atten_drug2] = self.layer(
            (g_lists, transformed_features, type_mask, edge_metapath_indices_lists, target_idx_lists))

        return [logits_drug1, logits_drug2], [h_drug1, h_drug2], [atten_drug1, atten_drug2]
