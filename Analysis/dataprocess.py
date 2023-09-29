import torch
from torch.utils.data import Dataset as BaseDataset
from torch import nn
import numpy as np

from collections import defaultdict, Counter
import os
import os.path as osp
import copy
import time
import pickle as pkl
from tqdm import tqdm

import torch_geometric.transforms as T
from torch_geometric.utils import degree, train_test_split_edges, remove_isolated_nodes, add_self_loops, to_undirected
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from torch_scatter import scatter_add

from ogb.linkproppred import PygLinkPropPredDataset
import networkx as nx


def load_planetoid_dataset(name):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = Planetoid(root=path, name=name)

    dataset.transform = T.NormalizeFeatures()

    return dataset


def load_data(args):
    if args.dataset in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = load_planetoid_dataset(args.dataset)
        data = dataset[0]

        data.edge_index = remove_isolated_nodes(data.edge_index)[0]

        if os.path.exists(os.getcwd() + '/data/' + args.dataset + '/edge.pkl'):
            edge = pkl.load(open(os.getcwd() + '/data/' + args.dataset + '/edge.pkl', 'rb'))
            neg_edge = pkl.load(open(os.getcwd() + '/data/' + args.dataset + '/neg_edge.pkl', 'rb'))
        else:
            transform = T.RandomLinkSplit(is_undirected=True, neg_sampling_ratio = 1.0, num_val = 0.1, num_test = 0.2)
            train_data, val_data, test_data = transform(data)
            train_edge, val_edge, test_edge = train_data.edge_label_index, val_data.edge_label_index, test_data.edge_label_index

            edge = {'Train': train_edge[:, :train_edge.shape[1]//2].t(), \
                    'Val': val_edge[:, :val_edge.shape[1]//2].t(),\
                    'Test': test_edge[:, :test_edge.shape[1]//2].t()}

            neg_edge = {'Train': train_edge[:, train_edge.shape[1]//2:].t(), \
                        'Val': val_edge[:, val_edge.shape[1]//2:].t(),\
                        'Test': test_edge[:, test_edge.shape[1]//2:].t()}

            pkl.dump(edge, open(os.getcwd() + '/data/' + args.dataset + '/edge.pkl', 'wb'))
            pkl.dump(neg_edge, open(os.getcwd() + '/data/' + args.dataset + '/neg_edge.pkl', 'wb'))


    elif args.dataset in ['ogbl-collab', 'ogbl-citation2']:
        dataset = PygLinkPropPredDataset(name = args.dataset)

        data = dataset[0]

        split_edge = dataset.get_edge_split()

        if args.dataset == 'ogbl-collab':
            edge = {'Train': torch.unique(split_edge['train']['edge'], dim = 0),
                    'Val': torch.unique(split_edge['valid']['edge'], dim = 0),
                    'Test': torch.unique(split_edge['test']['edge'], dim = 0)}

            neg_edge = {'Train': torch.unique(split_edge['valid']['edge_neg'], dim = 0),
                        'Val': torch.unique(split_edge['valid']['edge_neg'], dim = 0),
                        'Test': torch.unique(split_edge['test']['edge_neg'], dim = 0)}

            edge_index = to_undirected(edge['Train'].t(), edge_attr = None, num_nodes = data.x.shape[0])[0]
        
        elif args.dataset == 'ogbl-citation2':
            data.edge_index = to_undirected(edge_index = data.edge_index, num_nodes = data.num_nodes)

            torch.manual_seed(12345)
            idx = torch.randperm(split_edge['train']['source_node'].numel())[:86596]
            split_edge['eval_train'] = {
                'source_node': split_edge['train']['source_node'][idx],
                'target_node': split_edge['train']['target_node'][idx],
                'target_node_neg': split_edge['valid']['target_node_neg'],
            }


            edge = {'Train': torch.stack([split_edge['train']['source_node'], split_edge['train']['target_node']]).t(),\
                    'Val': torch.stack([split_edge['valid']['source_node'], split_edge['valid']['target_node']]).t(),\
                    'Test': torch.stack([split_edge['test']['source_node'], split_edge['test']['target_node']]).t()}

            neg_edge = None
            
    elif args.dataset in ['Reptile', 'Vole', 'Wiki_co_read']:
        edge = np.loadtxt(os.getcwd() + '/data/' + args.dataset + '/edge.txt', dtype = int)
        edge_set = list(set([(node1.item(), node2.item()) for node1, node2 in edge]))
        edge_set = np.array(edge_set, dtype = int)

        num_nodes = np.max(edge) + 1
        edge = torch.tensor(edge)

        edge_index = to_undirected(edge.t(), edge_attr = None, num_nodes = num_nodes)[0]

        if args.dataset in ['Reptile', 'Vole']:
            x = torch.ones((num_nodes, 64))
        else:
            x = torch.load(os.getcwd() + '/data/' + args.dataset + '/document_feature.pt')

        data = Data(x = x, edge_index=edge_index)
        data.edge_index = remove_isolated_nodes(data.edge_index)[0]

        if os.path.exists(os.getcwd() + '/data/' + args.dataset + '/edge.pkl'):
            edge = pkl.load(open(os.getcwd() + '/data/' + args.dataset + '/edge.pkl', 'rb'))
            neg_edge = pkl.load(open(os.getcwd() + '/data/' + args.dataset + '/neg_edge.pkl', 'rb'))
        else:
            transform = T.RandomLinkSplit(is_undirected=True, neg_sampling_ratio = 1.0, num_val = 0.1, num_test = 0.2)
            train_data, val_data, test_data = transform(data)
            train_edge, val_edge, test_edge = train_data.edge_label_index, val_data.edge_label_index, test_data.edge_label_index

            edge = {'Train': train_edge[:, :train_edge.shape[1]//2].t(), \
                    'Val': val_edge[:, :val_edge.shape[1]//2].t(),\
                    'Test': test_edge[:, :test_edge.shape[1]//2].t()}

            neg_edge = {'Train': train_edge[:, train_edge.shape[1]//2:].t(), \
                        'Val': val_edge[:, val_edge.shape[1]//2:].t(),\
                        'Test': test_edge[:, test_edge.shape[1]//2:].t()}

            pkl.dump(edge, open(os.getcwd() + '/data/' + args.dataset + '/edge.pkl', 'wb'))
            pkl.dump(neg_edge, open(os.getcwd() + '/data/' + args.dataset + '/neg_edge.pkl', 'wb'))



    if os.path.exists(os.getcwd() + '/data/' + args.dataset + '/adj_set_dict.pkl'):
        adj_set_dict = pkl.load(
            open(os.getcwd() + '/data/' + args.dataset + '/adj_set_dict.pkl', 'rb'))
    else:
        adj_set_dict = cal_adj_set_dict(edge, args.dataset)

    deg = {key: [len(adj_set_dict[key][i]) for i in range(data.x.shape[0])] for key in adj_set_dict}
    
    # print(args.tc, args.tc_layer)
    if os.path.exists(os.getcwd() + '/data/' + args.dataset + '/' + str(args.tc) + '_' + str(args.tc_layer) + '.pkl'):
        tc = pkl.load(open(os.getcwd() + '/data/' + args.dataset + '/' + str(args.tc) + '_' + str(args.tc_layer) + '.pkl', 'rb'))
        atc = pkl.load(open(os.getcwd() + '/data/' + args.dataset + '/' + str(args.tc) + '_' + str(args.tc_layer) + '.pkl', 'rb'))
    else:
        if args.tc == 'tc':
            compute_tc = cal_tc
        elif args.tc == 'appro_tc':
            compute_tc = cal_appro_tc
        elif args.tc == 'appro_tc_dot':
            compute_tc = cal_appro_tc_dot

        start = time.time()
        tc = compute_tc(adj_set_dict, data.x.shape[0], deg['Train'], edge['Train'], args.tc_layer)
        print(time.time() - start)

        pkl.dump(tc, open(os.getcwd() + '/data/' +
                           args.dataset + '/' + str(args.tc) + '_' + str(args.tc_layer) + '.pkl', 'wb'))
    
    if os.path.exists(os.getcwd() + '/data/' + args.dataset + '/density_{}.pkl'.format(args.tc_layer)):
        density = pkl.load(open(os.getcwd() + '/data/' + args.dataset + '/density_{}.pkl'.format(args.tc_layer), 'rb'))
    else:
        density = cal_density(adj_set_dict, data.x.shape[0], deg['Train'], edge['Train'], args.tc_layer)
        pkl.dump(density, open(os.getcwd() + '/data/' + args.dataset + '/density_{}.pkl'.format(args.tc_layer), 'wb'))
    



    eval_node = {key: torch.unique(edge[key]) for key in edge}
    test_node = eval_node[args.eval_node_type]

    data.edge_index = to_undirected(
        edge['Train'].t(), num_nodes=data.x.shape[0])

    return data, edge, adj_set_dict, neg_edge, test_node, eval_node, deg, tc, atc, density, data.x.shape[0], edge['Train'].shape[0] + edge['Val'].shape[0] + edge['Test'].shape[0]


def cal_adj_set_dict(edge, dataset):
    adj_set_dict = {'Train': defaultdict(set),
                     'Val': defaultdict(set),
                     'Test': defaultdict(set),
                     'Train_val': defaultdict(set),
                     'Train_val_test': defaultdict(set),
                    }

    for node1, node2 in edge['Train']:
        node1, node2 = node1.item(), node2.item()

        adj_set_dict['Train'][node1].add(node2)
        adj_set_dict['Train'][node2].add(node1)

        adj_set_dict['Train_val'][node1].add(node2)
        adj_set_dict['Train_val'][node2].add(node1)

        adj_set_dict['Train_val_test'][node1].add(node2)
        adj_set_dict['Train_val_test'][node2].add(node1)

    for node1, node2 in edge['Val']:
        node1, node2 = node1.item(), node2.item()

        adj_set_dict['Train_val'][node1].add(node2)
        adj_set_dict['Train_val'][node2].add(node1)

        adj_set_dict['Train_val_test'][node1].add(node2)
        adj_set_dict['Train_val_test'][node2].add(node1)

        adj_set_dict['Val'][node1].add(node2)
        adj_set_dict['Val'][node2].add(node1)

    for node1, node2 in edge['Test']:
        node1, node2 = node1.item(), node2.item()

        adj_set_dict['Train_val_test'][node1].add(node2)
        adj_set_dict['Train_val_test'][node2].add(node1)

        adj_set_dict['Test'][node1].add(node2)
        adj_set_dict['Test'][node2].add(node1)

    pkl.dump(adj_set_dict, open(os.getcwd() + '/data/' + dataset + '/adj_set_dict.pkl', 'wb'))

    return adj_set_dict


def cal_adj_list_dict(edge, dataset):
    adj_list_dict = {'Train': defaultdict(list),
                     'Val': defaultdict(list),
                     'Test': defaultdict(list),
                     'Train_val': defaultdict(list),
                     'Train_val_test': defaultdict(list),
                    }

    for node1, node2 in edge['Train']:
        node1, node2 = node1.item(), node2.item()

        adj_list_dict['Train'][node1].append(node2)
        adj_list_dict['Train'][node2].append(node1)

        adj_list_dict['Train_val'][node1].append(node2)
        adj_list_dict['Train_val'][node2].append(node1)

        adj_list_dict['Train_val_test'][node1].append(node2)
        adj_list_dict['Train_val_test'][node2].append(node1)

    for node1, node2 in edge['Val']:
        node1, node2 = node1.item(), node2.item()

        adj_list_dict['Train_val'][node1].append(node2)
        adj_list_dict['Train_val'][node2].append(node1)

        adj_list_dict['Train_val_test'][node1].append(node2)
        adj_list_dict['Train_val_test'][node2].append(node1)

        adj_list_dict['Val'][node1].append(node2)
        adj_list_dict['Val'][node2].append(node1)

    for node1, node2 in edge['Test']:
        node1, node2 = node1.item(), node2.item()

        adj_list_dict['Train_val_test'][node1].append(node2)
        adj_list_dict['Train_val_test'][node2].append(node1)

        adj_list_dict['Test'][node1].append(node2)
        adj_list_dict['Test'][node2].append(node1)

    pkl.dump(adj_list_dict, open(os.getcwd() + '/data/' + dataset + '/adj_list_dict.pkl', 'wb'))

    return adj_list_dict



def cal_density(adj_list_dict, n_nodes, deg, edge_index, K):
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    #transform edge_index into the list of tuples
    edge_tuples = [(node1, node2) for node1, node2 in edge_index.tolist()]
    G.add_edges_from(edge_tuples)

    def recursion_nei(sub_nodes, train_adj_list, k, K, prev_nei):
        if k >= K:
            return

        cur_nei = []
        for node in prev_nei:
            cur_nei.extend(train_adj_list[node])

        sub_nodes.extend(cur_nei)

        next_nei = recursion_nei(sub_nodes, train_adj_list, k + 1, K, cur_nei)

        return
    
    density = []
    for node1 in tqdm(range(n_nodes)):
        sub_nodes = [node1]
        recursion_nei(sub_nodes, adj_list_dict['Train'], 0, K, [node1])

        sub_nodes = list(set(sub_nodes))

        sub_G = G.subgraph(sub_nodes)
        
        if sub_G.number_of_nodes() <= 1:
            density.append(0)
        else:
            density.append(sub_G.number_of_edges() / (sub_G.number_of_nodes() * (sub_G.number_of_nodes() - 1) / 2))
    
    return density

def cal_node_topo_metric(edge_index, n_nodes):
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    #transform edge_index into the list of tuples
    edge_tuples = [(node1, node2) for node1, node2 in edge_index.tolist()]
    G.add_edges_from(edge_tuples)

    node_deg_central = nx.degree_centrality(G)
    node_deg_central = [item for key, item in node_deg_central.items()]

    node_bt_central = nx.betweenness_centrality(G)
    node_bt_central = [item for key, item in node_bt_central.items()]

    node_eig_central = nx.eigenvector_centrality(G)
    node_eig_central = [item for key, item in node_eig_central.items()]

    return node_deg_central, node_bt_central, node_eig_central


def cal_tc(adj_list_dict, n_nodes, deg, edge_index, K):
    train_train_tc, train_val_tc, train_test_tc = [], [], []

    def recursion_nei(adj_hop, train_adj_list, k, K, prev_nei):
        if k >= K:
            return

        cur_nei = []
        for node in prev_nei:
            cur_nei.extend(train_adj_list[node])

        adj_hop[k + 1] = cur_nei

        next_nei = recursion_nei(adj_hop, train_adj_list, k + 1, K, cur_nei)

        return

    beta = 1 / np.mean(deg)

    #add a tqdm bar
    for node1 in tqdm(range(n_nodes)):
        train_nei, val_nei, test_nei = adj_list_dict['Train'][
            node1], adj_list_dict['Val'][node1], adj_list_dict['Test'][node1]

        deg_train, deg_val, deg_test = len(
            train_nei), len(val_nei), len(test_nei)

        adj_hop1 = {}
        recursion_nei(adj_hop1, adj_list_dict['Train'], 0, K, [node1])

        nei_1_hop_weight = [Counter(adj_hop1[key]) for key in adj_hop1]
        nei_1_hop = [set(adj_hop1[key]) for key in adj_hop1]

        if deg_val > 0 and deg_train > 0:
            counts = []

            for node2 in val_nei:
                adj_hop2 = {}
                recursion_nei(adj_hop2, adj_list_dict['Train'], 0, K, [node2])

                nei_2_hop_weight = [Counter(adj_hop2[key]) for key in adj_hop2]
                nei_2_hop = [set(adj_hop2[key]) for key in adj_hop2]


                inters, norms = [], []

                for i in range(len(nei_1_hop)):
                    for j in range(len(nei_2_hop)):
                        tmp_nei_1, tmp_nei_2 = nei_1_hop[i], nei_2_hop[j]
                        tmp_nei_1_w, tmp_nei_2_w = nei_1_hop_weight[i], nei_2_hop_weight[j]

                        inters.append(sum([tmp_nei_1_w.get(tmp, 0) for tmp in tmp_nei_1.intersection(tmp_nei_2)]) * beta ** (i + j))
                        norms.append(sum(list(tmp_nei_1_w.values())) * beta ** (i + j))

                counts.append(sum(inters) / sum(norms))

            train_val_tc.append(np.mean(counts))

        else:
            train_val_tc.append(0)

        if deg_test > 0 and deg_train > 0:
            counts = []

            for node2 in test_nei:
                adj_hop2 = {}
                recursion_nei(adj_hop2, adj_list_dict['Train'], 0, K, [node2])

                nei_2_hop_weight = [Counter(adj_hop2[key]) for key in adj_hop2]
                nei_2_hop = [set(adj_hop2[key]) for key in adj_hop2]


                inters, norms = [], []

                for i in range(len(nei_1_hop)):
                    for j in range(len(nei_2_hop)):
                        tmp_nei_1, tmp_nei_2 = nei_1_hop[i], nei_2_hop[j]
                        tmp_nei_1_w, tmp_nei_2_w = nei_1_hop_weight[i], nei_2_hop_weight[j]

                        inters.append(sum([tmp_nei_1_w.get(tmp, 0) for tmp in tmp_nei_1.intersection(tmp_nei_2)]) * beta ** (i + j))
                        norms.append(sum(list(tmp_nei_1_w.values())) * beta ** (i + j))

                counts.append(sum(inters) / sum(norms))

            train_test_tc.append(np.mean(counts))

        else:
            train_test_tc.append(0)

        if deg_train > 1:
            counts = []

            for node2 in train_nei:
                adj_hop2 = {}
                recursion_nei(adj_hop2, adj_list_dict['Train'], 0, K, [node2])

                nei_2_hop_weight = [Counter(adj_hop2[key]) for key in adj_hop2]
                nei_2_hop = [set(adj_hop2[key]) for key in adj_hop2]


                inters, norms = [], []

                for i in range(len(nei_1_hop)):
                    for j in range(len(nei_2_hop)):
                        tmp_nei_1, tmp_nei_2 = nei_1_hop[i], nei_2_hop[j]
                        tmp_nei_1_w, tmp_nei_2_w = nei_1_hop_weight[i], nei_2_hop_weight[j]

                        inters.append(sum([tmp_nei_1_w.get(tmp, 0) for tmp in tmp_nei_1.intersection(tmp_nei_2)]) * beta ** (i + j))

                        if i == 0 and j == 0:
                            norms.append((sum(list(tmp_nei_1_w.values())) - 1) * beta ** (i + j))
                        else:
                            norms.append((sum(list(tmp_nei_1_w.values()))) * beta ** (i + j))

                counts.append(sum(inters) / sum(norms))

            train_train_tc.append(np.mean(counts))
        else:
            train_train_tc.append(0)

    lcc = {'train_train': train_train_tc,
           'train_val': train_val_tc, 'train_test': train_test_tc}

    return lcc



def cal_appro_tc(adj_list_dict, n_nodes, deg, edge_index, K):
    norm_edge_index, norm_edge_weight, deg = normalize_edge_sage(edge_index, n_nodes)

    train_train_tc, train_val_tc, train_test_tc = [], [], []

    cos = nn.CosineSimilarity(dim = 1, eps = 1e-6)

    num_edge = norm_edge_index[0].shape[0]

    weight = (deg/(2*num_edge))**(0)

    random_emb = torch.tensor(np.random.normal(loc=0.0, scale=1/64, size=(n_nodes, 64)))*weight.view(-1, 1)
    row, col = norm_edge_index[0], norm_edge_index[1]

    prop_embs = [scatter_add(random_emb[row]*norm_edge_weight.unsqueeze(-1), col, dim = 0, dim_size = n_nodes)]
    for k in range(1, K):
        prop_embs.append(scatter_add(prop_embs[-1][row]*norm_edge_weight.unsqueeze(-1), col, dim = 0, dim_size = n_nodes))

    prop_emb = torch.stack(prop_embs).mean(dim = 0)

    for key in adj_list_dict:
        for key2 in adj_list_dict[key]:
            adj_list_dict[key][key2] = list(adj_list_dict[key][key2])

    for node1 in range(n_nodes):
        if len(adj_list_dict['Test'][node1]) > 0:
            nei_emb = prop_emb[adj_list_dict['Test'][node1]]
            core_emb = prop_emb[node1].repeat(nei_emb.shape[0], 1)

            train_test_tc.append(cos(core_emb, nei_emb).mean())
        else:
            train_test_tc.append(0)


        if len(adj_list_dict['Val'][node1]) > 0:
            nei_emb = prop_emb[adj_list_dict['Val'][node1]]
            core_emb = prop_emb[node1].repeat(nei_emb.shape[0], 1)

            train_val_tc.append(cos(core_emb, nei_emb).mean())
        else:
            train_val_tc.append(0)


        if len(adj_list_dict['Train'][node1]) > 0:
            nei_emb = prop_emb[adj_list_dict['Train'][node1]]
            core_emb = prop_emb[node1].repeat(nei_emb.shape[0], 1)

            train_train_tc.append(cos(core_emb, nei_emb).mean())
        else:
            train_train_tc.append(0)


    lcc = {'train_train': train_train_tc,
           'train_test': train_test_tc, 'train_val': train_val_tc}

    return lcc



def cal_appro_tc_dot(adj_list_dict, n_nodes, deg, edge_index, K):
    norm_edge_index, norm_edge_weight, deg = normalize_edge_sage(edge_index, n_nodes)

    train_train_tc, train_val_tc, train_test_tc = [], [], []


    num_edge = norm_edge_index[0].shape[0]

    weight = (deg/(2*num_edge))**(0)

    random_emb = torch.tensor(np.random.normal(loc=0.0, scale=1/64, size=(n_nodes, 64)))*weight.view(-1, 1)
    row, col = norm_edge_index[0], norm_edge_index[1]

    prop_embs = [scatter_add(random_emb[row]*norm_edge_weight.unsqueeze(-1), col, dim = 0, dim_size = n_nodes)]
    for k in range(1, K):
        prop_embs.append(scatter_add(prop_embs[-1][row]*norm_edge_weight.unsqueeze(-1), col, dim = 0, dim_size = n_nodes))

    prop_emb = torch.stack(prop_embs).mean(dim = 0)

    for key in adj_list_dict:
        for key2 in adj_list_dict[key]:
            adj_list_dict[key][key2] = list(adj_list_dict[key][key2])

    for node1 in range(n_nodes):
        if len(adj_list_dict['Test'][node1]) > 0:
            nei_emb = prop_emb[adj_list_dict['Test'][node1]]
            core_emb = prop_emb[node1].repeat(nei_emb.shape[0], 1)

            train_test_tc.append(torch.mul(core_emb, nei_emb).sum(dim = 1).mean())
        else:
            train_test_tc.append(0)


        if len(adj_list_dict['Val'][node1]) > 0:
            nei_emb = prop_emb[adj_list_dict['Val'][node1]]
            core_emb = prop_emb[node1].repeat(nei_emb.shape[0], 1)

            train_val_tc.append(torch.mul(core_emb, nei_emb).sum(dim = 1).mean())
        else:
            train_val_tc.append(0)


        if len(adj_list_dict['Train'][node1]) > 0:
            nei_emb = prop_emb[adj_list_dict['Train'][node1]]
            core_emb = prop_emb[node1].repeat(nei_emb.shape[0], 1)

            train_train_tc.append(torch.mul(core_emb, nei_emb).sum(dim = 1).mean())
        else:
            train_train_tc.append(0)


    lcc = {'train_train': train_train_tc,
           'train_test': train_test_tc, 'train_val': train_val_tc}

    return lcc



def normalize_edge_sage(edge_index, n_node):
    edge_index = to_undirected(edge_index.t(), num_nodes=n_node)

    edge_index, _ = add_self_loops(
        edge_index, num_nodes=n_node)

    row, col = edge_index
    deg = degree(col)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.
    edge_weight = deg_inv_sqrt[col] * deg_inv_sqrt[col]

    return edge_index, edge_weight, deg


def normalize_edge_gcn(edge_index, n_node):
    edge_index = to_undirected(edge_index.t(), num_nodes=n_node)

    edge_index, _ = add_self_loops(
        edge_index, num_nodes=n_node)

    row, col = edge_index
    deg = degree(col)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    return edge_index, edge_weight



class General_dataset(BaseDataset):
    def __init__(self, n_edges):
        self.n_edges = n_edges

    def _get_feed_dict(self, index):
        feed_dict = index

        return feed_dict

    def __len__(self):
        return self.n_edges

    def __getitem__(self, index):
        return self._get_feed_dict(index)

    def collate_batch(self, feed_dicts):

        return torch.tensor(feed_dicts)




if __name__ == '__main__':
    adj_list_dict = {'Train': {0 : [1, 2, 3], 1: [], 2: [], 3: []}, \
                     'Val': {0 : [1, 2, 3], 1: [], 2: [], 3: []}, \
                     'Test': {0 : [1, 2, 3], 1: [], 2: [], 3: []}}
    n_nodes = 4
    deg = [3, 1, 1, 1]
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 0, 0]]).t()
    K = 1
    print(cal_density(adj_list_dict, n_nodes, deg, edge_index, K))