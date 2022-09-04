import numpy as np
import scipy
import pickle
import pandas as pd
from dgl.data import DGLDataset
import torch
import dgl
import os


def load_HNEMA_DDI_data_te(prefix='D:/B/PROJECT B2_2/dataset/generated_2/after_process/'): # your folder to store the required original data
    print('the path of source file is :', prefix)

    # read drug adjlist files using relative index
    in_file = open(prefix + '0/0-1-0.adjlist', 'r')
    adjlist00 = [line.strip() for line in in_file]
    adjlist00 = adjlist00
    in_file.close()
    in_file = open(prefix + '0/0-1-1-0.adjlist', 'r')
    adjlist01 = [line.strip() for line in in_file]
    adjlist01 = adjlist01
    in_file.close()
    in_file = open(prefix + '0/0-1-1-1-0.adjlist', 'r')
    adjlist02 = [line.strip() for line in in_file]
    adjlist02 = adjlist02
    in_file.close()
    in_file = open(prefix + '0/0-te-0.adjlist', 'r')
    adjlist03 = [line.strip() for line in in_file]
    adjlist03 = adjlist03
    in_file.close()

    # read the metapath instance files (stored as absolute index)
    in_file = open(prefix + '0/0-1-0_idx.pickle', 'rb')
    idx00 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '0/0-1-1-0_idx.pickle', 'rb')
    idx01 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '0/0-1-1-1-0_idx.pickle', 'rb')
    idx02 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '0/0-te-0_idx.pickle', 'rb')
    idx03 = pickle.load(in_file)
    in_file.close()

    # read adjacency matrix storing the all required interactions required by training
    adjM = scipy.sparse.load_npz(prefix + 'adjM.npz')
    # type_mask is a mask storing all nodes' types
    type_mask = np.load(prefix + 'node_types.npy')

    in_file = open(prefix + 'drug2relid_dict.pickle', 'rb')
    drug2id_dict = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + 'target2relid_dict.pickle', 'rb')
    target2id_dict = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + 'cellline2relid_dict.pickle', 'rb')
    cellline2id_dict = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + 'se_symbol2id_dict.pickle', 'rb')
    se_symbol2id_dict = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + 'atomnum2id_dict.pickle', 'rb')
    atomnum2id_dict = pickle.load(in_file)
    in_file.close()

    train_val_test_drug_drug_samples = np.load(prefix + 'train_val_test_drug_drug_samples.npz')
    train_val_test_drug_drug_labels = np.load(prefix + 'train_val_test_drug_drug_labels.npz')
    # ECFP6
    all_drug_morgan = scipy.sparse.load_npz(prefix + 'ECFP6_DNN_coomatrix.npz')
    # CCLE cell line expression data
    if os.path.exists(prefix + 'cellline_expression_normalized.npy'):
        cellline_expression = np.load(prefix + 'cellline_expression_normalized.npy', allow_pickle=True)
    else:
        cellline_expression = np.zeros((len(cellline2id_dict), 1024))
        print('an empty cell line expression data is generated.')

    return [[adjlist00, adjlist01, adjlist02, adjlist03],[adjlist00, adjlist01, adjlist02, adjlist03]], \
           [[idx00, idx01, idx02, idx03],[idx00, idx01, idx02, idx03]], \
           adjM, type_mask, \
           [drug2id_dict,target2id_dict,cellline2id_dict,se_symbol2id_dict,atomnum2id_dict], \
           train_val_test_drug_drug_samples, train_val_test_drug_drug_labels, all_drug_morgan, cellline_expression


def load_DNN_DDI_data_te(prefix='D:/B/PROJECT B2_2/dataset/generated_2/after_process/'): # your folder to store the required original data
    print('the path of source file is :', prefix)

    in_file = open(prefix + 'drug2relid_dict.pickle', 'rb')
    drug2id_dict = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + 'target2relid_dict.pickle', 'rb')
    target2id_dict = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + 'cellline2relid_dict.pickle', 'rb')
    cellline2id_dict = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + 'se_symbol2id_dict.pickle', 'rb') # adverse effect name to ids
    se_symbol2id_dict = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + 'atomnum2id_dict.pickle', 'rb') # atom numbers to ids
    atomnum2id_dict = pickle.load(in_file)
    in_file.close()

    train_val_test_drug_drug_samples = np.load(prefix + 'train_val_test_drug_drug_samples.npz')
    train_val_test_drug_drug_labels = np.load(prefix + 'train_val_test_drug_drug_labels.npz')
    # ECFP6
    all_drug_morgan = scipy.sparse.load_npz(prefix + 'ECFP6_DNN_coomatrix.npz')
    # CCLE cell line expression data
    if os.path.exists(prefix + 'cellline_expression_normalized.npy'):
        cellline_expression = np.load(prefix + 'cellline_expression_normalized.npy', allow_pickle=True)
    else:
        cellline_expression = np.zeros((len(cellline2id_dict), 1024))
        print('An empty cell line expression data is generated.')

    return [drug2id_dict,target2id_dict,cellline2id_dict,se_symbol2id_dict,atomnum2id_dict], \
           train_val_test_drug_drug_samples, train_val_test_drug_drug_labels, all_drug_morgan, cellline_expression


# extra data for HNE-GIN (stored in './data/data4training_model/')
class DrugStrucDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='drugstruc')

    def process(self):
        # the path for reading drug molecular graphs
        prefix='./data/data4training_model/'
        print('the path of drug structure file is :', prefix)
        edges = pd.read_csv(prefix + 'drug_graph_edges.csv')
        properties = pd.read_csv(prefix + 'drug_graph_properties.csv')
        nodes = pd.read_csv(prefix + 'drug_graph_nodes.csv')
        self.graphs = []
        self.labels = []

        # Create a graph for each graph ID from the edges table.
        # First process the properties table into two dictionaries with graph IDs as keys.
        # The label and number of nodes are values.
        label_dict = {}
        num_nodes_dict = {}
        for _, row in properties.iterrows():
            label_dict[row['graph_id']] = row['label']
            num_nodes_dict[row['graph_id']] = row['num_nodes']

        # For the edges, first group the table by graph IDs.
        edges_group = edges.groupby('graph_id')
        nodes_group = nodes.groupby('graph_id')

        # For each graph ID...
        for graph_id in edges_group.groups:
            # Find the edges as well as the number of nodes and its label.
            edges_of_id = edges_group.get_group(graph_id)
            nodes_of_id = nodes_group.get_group(graph_id)
            src = edges_of_id['src'].to_numpy()
            dst = edges_of_id['dst'].to_numpy()
            num_nodes = num_nodes_dict[graph_id]
            label = label_dict[graph_id]
            atom_features = torch.from_numpy(nodes_of_id['atom_num'].to_numpy())
            atom_features = atom_features.type(torch.float32)

            # Create a graph and add it to the list of graphs and labels.
            # create dgl graph for each molecule
            g = dgl.graph((src, dst), num_nodes=num_nodes)
            g.ndata['atom_num'] = atom_features
            self.graphs.append(g)
            self.labels.append(label)

        # Convert the label list to tensor for saving.
        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)


# Save graphs
# dgl.save_graphs('graph.dgl', g)
# dgl.save_graphs('graphs.dgl', [g, sg1, sg2])
# Load graphs
# (g,), _ = dgl.load_graphs('graph.dgl')
# (g, sg1, sg2), _ = dgl.load_graphs('graphs.dgl'