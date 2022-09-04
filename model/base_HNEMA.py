import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax
import copy
from model.Trans_encoder import MultiHeadedAttention, PositionwiseFeedForward, Encoder, EncoderLayer


class HNEMA_metapath_specific(nn.Module):
    def __init__(self,
                 out_dim,
                 num_heads,
                 rnn_type='bi-gru',
                 attn_drop=0.5,
                 alpha=0.01,
                 use_minibatch=False,
                 attn_switch=False,
                 rnn_concat=False):
        super(HNEMA_metapath_specific, self).__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.rnn_type = rnn_type
        self.use_minibatch = use_minibatch
        self.attn_switch = attn_switch
        self.rnn_concat = rnn_concat

        # rnn-like metapath instance aggregator
        # consider multiple attention heads
        if rnn_type == 'bi-lstm':
            print('current rnn type is:', rnn_type)
            self.rnn = nn.LSTM(out_dim, num_heads * out_dim // 2, bidirectional=True)
        elif rnn_type == 'lstm':
            print('current rnn type is:', rnn_type)
            self.rnn = nn.LSTM(out_dim, num_heads * out_dim)
        elif rnn_type == 'bi-gru':
            print('current rnn type is:', rnn_type)
            self.rnn = nn.GRU(out_dim, num_heads * out_dim // 2, bidirectional=True)
        elif rnn_type == 'gru':
            print('current rnn type is:', rnn_type)
            self.rnn = nn.GRU(out_dim, num_heads * out_dim)
        elif rnn_type == 'mean':
            self.trans_out1 = nn.Linear(out_dim, num_heads * out_dim, bias=False)
            nn.init.xavier_normal_(self.trans_out1.weight, gain=1.414)
        elif rnn_type == 'transformer':
            c = copy.deepcopy
            # attn = MultiHeadedAttention(num_heads, out_dim)
            attn = MultiHeadedAttention(1, out_dim)

            # the size of fc equals to the output of multi-attention layer
            # num_heads: 64, out_dim:8, fc_hidden_state, dropout

            # ff = PositionwiseFeedForward(num_heads * out_dim, 512, 0.1)
            ff = PositionwiseFeedForward(out_dim, 512, 0.1)

            # self.skip_proj = nn.Linear(out_dim, num_heads * out_dim, bias=False)
            # the second parameter represents the number of encoder block of transformer extractor
            self.rnn = Encoder(EncoderLayer(out_dim, c(attn), c(ff), 0.1), 6)

            self.trans_out1 = nn.Linear(out_dim, num_heads * out_dim, bias=False)
            self.trans_out2 = nn.Linear(out_dim, num_heads * out_dim, bias=False)

            for p in self.rnn.parameters():
                if (p.dim() > 1):
                    nn.init.xavier_uniform_(p)

            # nn.init.xavier_normal_(self.skip_proj.weight, gain=1.414)
            nn.init.xavier_normal_(self.trans_out1.weight, gain=1.414)
            nn.init.xavier_normal_(self.trans_out2.weight, gain=1.414)

        # elif rnn_type == 'gat':
        #     self.trans_out1 = nn.Linear(out_dim, num_heads * out_dim, bias=False)
        #     nn.init.xavier_normal_(self.trans_out1.weight, gain=1.414)

        # node-level attention
        # attention considers the center node embedding or not
        if self.attn_switch:
            # self.attn1 = nn.Linear(out_dim, num_heads, bias=False)
            self.attn1 = nn.Linear(num_heads * out_dim, num_heads, bias=False)
            self.attn2 = nn.Parameter(torch.empty(size=(1, num_heads, out_dim)))
        else:
            self.attn = nn.Parameter(torch.empty(size=(1, num_heads, out_dim)))
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.softmax = edge_softmax
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

        # weight initialization
        if self.attn_switch:
            nn.init.xavier_normal_(self.attn1.weight, gain=1.414)
            nn.init.xavier_normal_(self.attn2.data, gain=1.414)
        else:
            nn.init.xavier_normal_(self.attn.data, gain=1.414)

    def edge_softmax(self, g):
        attention = self.softmax(g, g.edata.pop('a'))
        g.edata['a_drop'] = self.attn_drop(attention)

    def message_passing(self, edges):
        ft = edges.data['eft'] * edges.data['a_drop']
        return {'ft': ft}

    def message_passing_bi_lstm(self, edges):
        avg = edges.data['eft']
        return {'avg': avg}

    def forward(self, inputs):
        if self.use_minibatch:
            g, features, type_mask, edge_metapath_indices, target_idx = inputs
        else:
            g, features, type_mask, edge_metapath_indices = inputs

        # Embedding layer
        # use torch.nn.functional.embedding or torch.embedding here
        # do not use torch.nn.embedding
        # edata: E x Seq x out_dim
        edata = F.embedding(edge_metapath_indices, features)

        # apply rnn to metapath-based feature sequence
        if self.rnn_type == 'bi-lstm':
            # the size of output is  [sequence length, batch size, num_heads (e.g.,8) * out_dim (e.g.,64) ]
            output, (hidden, _) = self.rnn(edata.permute(1, 0, 2))

            if (self.attn_switch == True):
                # source and target node embeddings contain the separate word embedding through the bidirectional learning
                # hidden contains comprehensive information extracted from the whole sequence
                # source_node_embed = output[0]
                target_node_embed = output[-1]

                # in this case, torch.split equals to tensor.reshape
                target_node_embed = target_node_embed.reshape(-1, self.out_dim, self.num_heads).permute(
                    0, 2, 1).reshape(-1, self.num_heads * self.out_dim).unsqueeze(dim=0)
                hidden = hidden.permute(1, 0, 2).reshape(-1, self.out_dim, self.num_heads).permute(
                    0, 2, 1).reshape(-1, self.num_heads * self.out_dim).unsqueeze(dim=0)

            else:
                hidden = hidden.permute(1, 0, 2).reshape(-1, self.out_dim, self.num_heads).permute(
                    0, 2, 1).reshape(-1, self.num_heads * self.out_dim).unsqueeze(dim=0)

        elif self.rnn_type == 'bi-gru':
            output, hidden = self.rnn(edata.permute(1, 0, 2))
            if (self.attn_switch == True):
                # source and target node embeddings contain the separate word embedding through the bidirectional learning
                # hidden contains comprehensive information extracted from the whole sequence
                # source_node_embed = output[0]
                target_node_embed = output[-1]
                # print(output.size())
                # torch.Size([3, 62, 512])

                # in this case, torch.split equals to tensor.reshape
                target_node_embed = target_node_embed.reshape(-1, self.out_dim, self.num_heads).permute(
                    0, 2, 1).reshape(-1, self.num_heads * self.out_dim).unsqueeze(dim=0)
                hidden = hidden.permute(1, 0, 2).reshape(-1, self.out_dim, self.num_heads).permute(
                    0, 2, 1).reshape(-1, self.num_heads * self.out_dim).unsqueeze(dim=0)

            else:
                hidden = hidden.permute(1, 0, 2).reshape(-1, self.out_dim, self.num_heads).permute(
                    0, 2, 1).reshape(-1, self.num_heads * self.out_dim).unsqueeze(dim=0)

        elif self.rnn_type == 'lstm':
            output, (hidden, _) = self.rnn(edata.permute(1, 0, 2))
            if (self.attn_switch == True):
                target_node_embed = output[-1]
                target_node_embed = target_node_embed.reshape(-1, self.out_dim, self.num_heads).permute(
                    0, 2, 1).reshape(-1, self.num_heads * self.out_dim).unsqueeze(dim=0)

        elif self.rnn_type == 'gru':
            output, hidden = self.rnn(edata.permute(1, 0, 2))
            if (self.attn_switch == True):
                target_node_embed = output[-1]
                target_node_embed = target_node_embed.reshape(-1, self.out_dim, self.num_heads).permute(
                    0, 2, 1).reshape(-1, self.num_heads * self.out_dim).unsqueeze(dim=0)

        elif self.rnn_type == 'mean':
            hidden = torch.mean(edata, dim=1)
            hidden = torch.cat([hidden] * self.num_heads, dim=1)
            hidden = hidden.unsqueeze(dim=0)

            if (self.attn_switch == True):
                edata = self.trans_out1(edata)
                target_node_embed = edata[:, -1, :]
                target_node_embed = target_node_embed.reshape(-1, self.out_dim, self.num_heads).permute(
                    0, 2, 1).reshape(-1, self.num_heads * self.out_dim).unsqueeze(dim=0)

        elif self.rnn_type == 'transformer':
            output = self.rnn(edata, None)
            target_node_embed = output[:, -1, :]

            target_node_embed = self.trans_out1(target_node_embed)
            hidden = self.leaky_relu(self.trans_out2(output.mean(dim=1)))

            target_node_embed = target_node_embed.reshape(-1, self.out_dim, self.num_heads).permute(
                0, 2, 1).reshape(-1, self.num_heads * self.out_dim).unsqueeze(dim=0)
            hidden = hidden.reshape(-1, self.out_dim, self.num_heads).permute(
                0, 2, 1).reshape(-1, self.num_heads * self.out_dim).unsqueeze(dim=0)

        # elif self.rnn_type == 'gat':
        #     edata = self.trans_out1(edata)
        #     source_node_embed = edata[:, 0, :]
        #     target_node_embed = edata[:, -1, :]
        #     source_node_embed = source_node_embed.reshape(-1, self.out_dim, self.num_heads).permute(
        #         0, 2, 1).reshape(-1, self.num_heads * self.out_dim).unsqueeze(dim=0)
        #     target_node_embed = target_node_embed.reshape(-1, self.out_dim, self.num_heads).permute(
        #         0, 2, 1).reshape(-1, self.num_heads * self.out_dim).unsqueeze(dim=0)
        #     hidden = source_node_embed

        eft = hidden.permute(1, 0, 2).view(-1, self.num_heads, self.out_dim)  # E x num_heads x out_dim

        if self.attn_switch:
            center_node_feat = target_node_embed.squeeze(dim=0)
            a1 = self.attn1(center_node_feat)  # E x num_heads
            # self.attn2 = nn.Parameter(torch.empty(size=(1, num_heads, out_dim))), eft=E x num_heads x out_dim
            # shape = (N, NH, FOUT) * (1, NH, FOUT) -> (N, NH, 1) -> (N, NH) because sum squeezes the last dimension
            a2 = (eft * self.attn2).sum(dim=-1)  # E x num_heads
            a = (a1 + a2).unsqueeze(dim=-1)  # E x num_heads x 1
        else:
            a = (eft * self.attn).sum(dim=-1).unsqueeze(dim=-1)  # E x num_heads x 1

        a = self.leaky_relu(a)
        # switch the device
        g = g.to(torch.device('cuda:0'))
        # g = g.to(torch.device('cpu'))

        g.edata.update({'eft': eft, 'a': a})
        self.edge_softmax(g)

        # compute the aggregated node features scaled by the dropped, unnormalized attention values.
        # Send messages along all the edges of the specified type and update all the nodes of the corresponding destination type.
        g.update_all(self.message_passing, fn.sum('ft', 'ft'))
        ret = g.ndata['ft']  # E x num_heads x out_dim

        if (self.rnn_concat == True):
            g.update_all(self.message_passing_bi_lstm, fn.mean('avg', 'avg'))
            aux = g.ndata['avg']

        if self.use_minibatch:
            if (self.rnn_concat == True):
                return torch.cat([ret[target_idx], aux[target_idx]], dim=-1)
            else:
                return ret[target_idx]
        else:
            return ret


class HNEMA_ctr_ntype_specific(nn.Module):
    def __init__(self,
                 num_metapaths,
                 out_dim,
                 num_heads,
                 attn_vec_dim,
                 rnn_type='bi-gru',
                 attn_drop=0.5,
                 use_minibatch=False,
                 attn_switch=False,
                 rnn_concat=False):
        super(HNEMA_ctr_ntype_specific, self).__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.use_minibatch = use_minibatch
        self.rnn_concat = rnn_concat

        # metapath-specific layers
        self.metapath_layers = nn.ModuleList()
        for i in range(num_metapaths):
            self.metapath_layers.append(HNEMA_metapath_specific(out_dim,
                                                                num_heads,
                                                                rnn_type,
                                                                attn_drop=attn_drop,
                                                                use_minibatch=use_minibatch,
                                                                attn_switch=attn_switch,
                                                                rnn_concat=rnn_concat))

        # metapath-level attention
        # note that the actual input dimension should consider the number of heads
        # as multiple head outputs are concatenated together
        if (self.rnn_concat == True):
            self.fc1 = nn.Linear(out_dim * num_heads * 2, attn_vec_dim, bias=True)
            self.fc2 = nn.Linear(attn_vec_dim, 1, bias=False)
        else:
            self.fc1 = nn.Linear(out_dim * num_heads, attn_vec_dim, bias=True)
            self.fc2 = nn.Linear(attn_vec_dim, 1, bias=False)

        # weight initialization
        nn.init.xavier_normal_(self.fc1.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc2.weight, gain=1.414)

    def forward(self, inputs):
        if self.use_minibatch:
            g_list, features, type_mask, edge_metapath_indices_list, target_idx_list = inputs

            # metapath-specific layers
            if self.rnn_concat == True:
                metapath_outs = [
                    F.elu(metapath_layer((g, features, type_mask, edge_metapath_indices, target_idx)).view(-1,
                                                                                                           self.num_heads * self.out_dim * 2))
                    for g, edge_metapath_indices, target_idx, metapath_layer in
                    zip(g_list, edge_metapath_indices_list, target_idx_list, self.metapath_layers)]
            else:
                metapath_outs = [
                    F.elu(metapath_layer((g, features, type_mask, edge_metapath_indices, target_idx)).view(-1,
                                                                                                           self.num_heads * self.out_dim))
                    for g, edge_metapath_indices, target_idx, metapath_layer in
                    zip(g_list, edge_metapath_indices_list, target_idx_list, self.metapath_layers)]

        else:
            g_list, features, type_mask, edge_metapath_indices_list = inputs

            # metapath-specific layers
            metapath_outs = [F.elu(
                metapath_layer((g, features, type_mask, edge_metapath_indices)).view(-1, self.num_heads * self.out_dim))
                for g, edge_metapath_indices, metapath_layer in
                zip(g_list, edge_metapath_indices_list, self.metapath_layers)]

        beta = []
        # all the metapaths share the same fc1 and fc2
        for metapath_out in metapath_outs:
            fc1 = torch.tanh(self.fc1(metapath_out))
            # calculate the mean value of this metapath
            fc1_mean = torch.mean(fc1, dim=0)
            fc2 = self.fc2(fc1_mean)
            beta.append(fc2)

        beta = torch.cat(beta, dim=0)
        beta = F.softmax(beta, dim=0)
        beta = torch.unsqueeze(beta, dim=-1)
        beta = torch.unsqueeze(beta, dim=-1)

        metapath_outs = [torch.unsqueeze(metapath_out, dim=0) for metapath_out in metapath_outs]

        metapath_outs = torch.cat(metapath_outs, dim=0)

        h = torch.sum(beta * metapath_outs, dim=0)
        return h, beta


class HNEMA_ctr_ntype_specific_transformer(nn.Module):
    def __init__(self,
                 num_metapaths,
                 etypes_list,
                 out_dim,
                 num_heads,
                 attn_vec_dim,
                 rnn_type='transformer',
                 attn_drop=0.5,
                 use_minibatch=False,
                 attn_switch=False,
                 rnn_concat=False):
        super(HNEMA_ctr_ntype_specific_transformer, self).__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.use_minibatch = use_minibatch
        self.rnn_concat = rnn_concat

        # metapath-specific layers
        self.metapath_layers = nn.ModuleList()
        for i in range(num_metapaths):
            self.metapath_layers.append(HNEMA_metapath_specific(etypes_list[i],
                                                                out_dim,
                                                                num_heads,
                                                                rnn_type,
                                                                attn_drop=attn_drop,
                                                                use_minibatch=use_minibatch,
                                                                attn_switch=attn_switch,
                                                                rnn_concat=rnn_concat))

        # metapath-level attention
        # note that the acutal input dimension should consider the number of heads
        # as multiple head outputs are concatenated together
        if (self.rnn_concat == True):
            self.fc1 = nn.Linear(out_dim * num_heads * 2, attn_vec_dim, bias=True)
            self.fc2 = nn.Linear(attn_vec_dim, 1, bias=False)
            self.metapath_fuse = Attention_fuse(out_dim * num_heads * 2, attn_vec_dim)
        else:
            self.fc1 = nn.Linear(out_dim * num_heads, attn_vec_dim, bias=True)
            self.fc2 = nn.Linear(attn_vec_dim, 1, bias=False)
            self.metapath_fuse = Attention_fuse(out_dim * num_heads, attn_vec_dim)

        # weight initialization
        nn.init.xavier_normal_(self.fc1.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc2.weight, gain=1.414)

    def forward(self, inputs):
        if self.use_minibatch:
            g_list, features, type_mask, edge_metapath_indices_list, target_idx_list = inputs

            # metapath-specific layers
            if self.rnn_concat == True:
                metapath_outs = [
                    F.elu(metapath_layer((g, features, type_mask, edge_metapath_indices, target_idx)).view(-1,
                                                                                                           self.num_heads * self.out_dim * 2))
                    for g, edge_metapath_indices, target_idx, metapath_layer in
                    zip(g_list, edge_metapath_indices_list, target_idx_list, self.metapath_layers)]
            else:
                metapath_outs = [
                    F.elu(metapath_layer((g, features, type_mask, edge_metapath_indices, target_idx)).view(-1,
                                                                                                           self.num_heads * self.out_dim))
                    for g, edge_metapath_indices, target_idx, metapath_layer in
                    zip(g_list, edge_metapath_indices_list, target_idx_list, self.metapath_layers)]


        else:
            g_list, features, type_mask, edge_metapath_indices_list = inputs

            # metapath-specific layers
            metapath_outs = [F.elu(
                metapath_layer((g, features, type_mask, edge_metapath_indices)).view(-1, self.num_heads * self.out_dim))
                for g, edge_metapath_indices, metapath_layer in
                zip(g_list, edge_metapath_indices_list, self.metapath_layers)]

        metapath_outs = torch.tensor([item.cpu().detach().numpy() for item in metapath_outs]).cuda()
        # add non-linearity to fusing node features of different view
        # Q = metapath_outs, K = metapath_outs, V = metapath_outs
        h = self.metapath_fuse(metapath_outs, metapath_outs, metapath_outs)
        return h


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Attention_fuse(nn.Module):
    def __init__(self, dim_model, attn_vec_dim):
        super(Attention_fuse, self).__init__()
        self.linears = clones(nn.Linear(dim_model, attn_vec_dim), 2)

        for linear in self.linears:
            nn.init.xavier_normal_(linear.weight, gain=1.414)

    def forward(self, query, key, value):
        metapath_num = query.size(0)
        feature_dim = query.size(-1)
        # flatten the sequence dim of query and key
        query, key = [l(x) for l, x in zip(self.linears, (query, key))]
        query = query.reshape(metapath_num, -1)
        key = key.reshape(metapath_num, -1)
        value = value.reshape(metapath_num, -1)
        attention = torch.mm(query, torch.t(key))
        d_k = torch.tensor(key.size(-1), dtype=torch.float32)
        attention = torch.div(attention, torch.sqrt(d_k))
        attention = F.softmax(attention, dim=-1)
        output = torch.matmul(attention, value).reshape(metapath_num, -1, feature_dim)
        output = torch.mean(output, dim=0)
        # print(attention)
        return output
