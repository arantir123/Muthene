import torch
import dgl
import numpy as np

def parse_adjlist(adjlist, edge_metapath_indices, samples=None, exclude=None, offset=None, mode=None):
    edges = []
    nodes = set()
    result_indices = []

    for row, indices in zip(adjlist, edge_metapath_indices):
        row_parsed = list(map(int, row.split(' ')))
        nodes.add(row_parsed[0])
        # if meta-path based neighbors exist
        if len(row_parsed) > 1:
            # sampling neighbors
            if samples is None:
                if exclude is not None:
                    if mode == 0:
                        mask = [False if [u1, a1 - offset] in exclude or [u2, a2 - offset] in exclude else True for
                                u1, a1, u2, a2 in indices[:, [0, 1, -1, -2]]]
                    else:
                        mask = [False if [u1, a1 - offset] in exclude or [u2, a2 - offset] in exclude else True for
                                a1, u1, a2, u2 in indices[:, [0, 1, -1, -2]]]
                    neighbors = np.array(row_parsed[1:])[mask]
                    result_indices.append(indices[mask])
                else:
                    neighbors = row_parsed[1:]
                    result_indices.append(indices)
            else:
                # undersampling frequent neighbors
                unique, counts = np.unique(row_parsed[1:], return_counts=True)
                p = []
                for count in counts:
                    p += [(count ** (3 / 4)) / count] * count
                p = np.array(p)
                p = p / p.sum()

                samples = min(samples, len(row_parsed) - 1)
                sampled_idx = np.sort(np.random.choice(len(row_parsed) - 1, samples, replace=False, p=p))

                if exclude is not None:
                    if mode == 0:
                        mask = [False if [u1, a1 - offset] in exclude or [u2, a2 - offset] in exclude else True for
                                u1, a1, u2, a2 in indices[sampled_idx][:, [0, 1, -1, -2]]]
                    else:
                        mask = [False if [u1, a1 - offset] in exclude or [u2, a2 - offset] in exclude else True for
                                a1, u1, a2, u2 in indices[sampled_idx][:, [0, 1, -1, -2]]]

                    neighbors = np.array([row_parsed[i + 1] for i in sampled_idx])[mask]
                    result_indices.append(indices[sampled_idx][mask])
                else:
                    neighbors = [row_parsed[i + 1] for i in sampled_idx]
                    result_indices.append(indices[sampled_idx])

        # for the case that a node does not have any neighbors
        else:
            neighbors = [row_parsed[0]]
            indices = np.array([[row_parsed[0]] * indices.shape[1]])
            if mode == 1:
                indices += offset
            result_indices.append(indices)

        for dst in neighbors:
            nodes.add(dst)
            edges.append((row_parsed[0], dst))

    mapping = {map_from: map_to for map_to, map_from in enumerate(sorted(nodes))}
    # use mapping to transform edge ids
    edges = list(map(lambda tup: (mapping[tup[0]], mapping[tup[1]]), edges))
    result_indices = np.vstack(result_indices)

    return edges, result_indices, len(nodes), mapping


def parse_minibatch(adjlists_ua, edge_metapath_indices_list_ua, drug_target_batch, device, samples=None,
                        use_masks=None, offset=None):
    # first parameter: meta-path based neighbors for each drug node
    # second parameter: nodes in every meta-path instance stored using relative ids
    # third parameter: node ids of drug-drug pairs in current batch
    g_lists = [[], []]
    result_indices_lists = [[], []]
    idx_batch_mapped_lists = [[], []]
    # the loop for iterating the drug node and target node
    for mode, (adjlists, edge_metapath_indices_list) in enumerate(zip(adjlists_ua, edge_metapath_indices_list_ua)):
        # the loop for iterating every metapath of one type of node
        # the order of adjlist and indices are the same
        for adjlist, indices, use_mask in zip(adjlists, edge_metapath_indices_list, use_masks[mode]):
            if use_mask:
                # samples=100
                edges, result_indices, num_nodes, mapping = parse_adjlist(
                    [adjlist[row[mode]] for row in drug_target_batch],
                    [indices[row[mode]] for row in drug_target_batch], samples, drug_target_batch, offset, mode)

            else:
                edges, result_indices, num_nodes, mapping = parse_adjlist(
                    [adjlist[row[mode]] for row in drug_target_batch],
                    [indices[row[mode]] for row in drug_target_batch], samples, offset=offset, mode=mode)

            # Multigraph means that there can be multiple edges between two nodes.
            # Multigraphs are graphs that can have multiple (directed) edges between the same pair of nodes, including self loops. For instance, two authors can coauthor a paper in different years, resulting in edges with different features.
            g = dgl.DGLGraph(multigraph=True)
            g.add_nodes(num_nodes)

            if len(edges) > 0:
                sorted_index = sorted(range(len(edges)), key=lambda i: edges[i])
                g.add_edges(*list(zip(*[(edges[i][1], edges[i][0]) for i in sorted_index])))

                result_indices = torch.LongTensor(result_indices[sorted_index]).to(device)

            else:
                result_indices = torch.LongTensor(result_indices).to(device)

            g_lists[mode].append(g)
            result_indices_lists[mode].append(result_indices)
            idx_batch_mapped_lists[mode].append(np.array([mapping[row[mode]] for row in drug_target_batch]))

    # print(g_lists,len(g_lists))
    # print(result_indices_lists,len(result_indices_lists))
    # print(idx_batch_mapped_lists,len(idx_batch_mapped_lists))
    return g_lists, result_indices_lists, idx_batch_mapped_lists


class index_generator:
    def __init__(self, batch_size, num_data=None, indices=None, shuffle=True):
        if num_data is not None:
            self.num_data = num_data
            self.indices = np.arange(num_data)
        if indices is not None:
            self.num_data = len(indices)
            self.indices = np.copy(indices)
        self.batch_size = batch_size
        self.iter_counter = 0
        self.shuffle = shuffle
        if shuffle:
            np.random.shuffle(self.indices)

    def next(self):
        if self.num_iterations_left() <= 0:
            self.reset()
        self.iter_counter += 1
        return np.copy(self.indices[(self.iter_counter - 1) * self.batch_size:self.iter_counter * self.batch_size])

    def num_iterations(self):
        return int(np.ceil(self.num_data / self.batch_size))

    def num_iterations_left(self):
        return self.num_iterations() - self.iter_counter

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.iter_counter = 0