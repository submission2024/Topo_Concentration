from torch_geometric.nn import GCNConv, SAGEConv
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=True, normalize=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True, normalize=False))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, cached=True, normalize=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(MLP, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                nn.Linear(hidden_channels, hidden_channels))
        self.convs.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x)
        return x


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x):
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lins[-1](x)

        return torch.sigmoid(x)

    def score(self, x):
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)

        x = self.lins[-1](x)

        return torch.sigmoid(x)


class LightGCN(nn.Module):
    def __init__(self, args):
        super(LightGCN, self).__init__()

        self.args = args
        self.gcn = self._init_model()

    def _init_model(self):
        return GraphConv(self.args)

    def reset_parameters(self):
        initializer = nn.init.xavier_uniform_
        self.embeds = nn.Parameter(initializer(torch.empty(
            self.args.n_nodes, self.args.n_hidden)))

    def batch_generate(self, node, pos_node, neg_node, adj_sp_norm):
        embs = self.gcn(self.embeds, adj_sp_norm)
        embs = self.pooling(embs)

        node_embs = embs[node]
        pos_item_embs = embs[pos_node]
        neg_item_embs = embs[neg_node]

        return node_embs, pos_item_embs, neg_item_embs

    def forward(self, node, pos_node, neg_node, adj_sp_norm):
        node_embs, pos_item_embs, neg_item_embs = self.batch_generate(
            node, pos_node, neg_node, adj_sp_norm)

        return node_embs, pos_item_embs, neg_item_embs, self.embeds[node], self.embeds[pos_node], self.embeds[neg_node]

    def pooling(self, embeddings):
        # [-1, n_hops, channel]
        if self.args.aggr == 'mean':
            return embeddings.mean(dim=1)
        elif self.args.aggr == 'sum':
            return embeddings.sum(dim=1)
        elif self.args.aggr == 'concat':
            return embeddings.view(embeddings.shape[0], -1)
        else:
            return embeddings[:, -1, :]

    def generate(self, adj_sp_norm):
        node_embs = self.gcn(self.embeds, adj_sp_norm)

        node_embs = self.pooling(node_embs)

        return node_embs


class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """

    def __init__(self, args):
        super(GraphConv, self).__init__()
        self.args = args

    def forward(self, embed, adj_sp_norm):
        # user_embed: [n_users, channel]
        # item_embed: [n_items, channel]

        # all_embed: [n_users+n_items, channel]
        agg_embed = embed
        embs = [embed]

        for hop in range(self.args.n_layers):
            # print(aj_sp_norm, agg_embed.device)
            agg_embed = adj_sp_norm.matmul(agg_embed)
            embs.append(agg_embed)

        embs = torch.stack(embs, dim=1)  # [n_entity, n_hops+1, emb_size]

        return embs



class CN(nn.Module):
    def __init__(self, adj_list_dict, args):
        super(CN, self).__init__()

        self.args = args

        self.train_adj = adj_list_dict['Train']
        self.val_adj = adj_list_dict['Val']
        self.test_adj = adj_list_dict['Test']

    def forward(self, nodes1, nodes2):
        intersections = []

        for i in range(nodes1.shape[0]):
            node1, node2 = nodes1[i].item(), nodes2[i].item()

            intersections.append(len(self.train_adj[node1].intersection(self.train_adj[node2])))

        return torch.tensor(intersections)


class Heuristics(nn.Module):
    def __init__(self, args):
        super(Heuristics, self).__init__()

        self.adj_matrix = args.adj_matrix

        if args.model == 'RA':
            multiplier = 1/self.adj_matrix.sum(axis = 0)
            multiplier[np.isinf(multiplier)] = 0

            self.adj_matrix = self.adj_matrix.multiply(multiplier).tocsr()
        elif args.model == 'AA':
            multiplier = 1/np.log(self.adj_matrix.sum(axis = 0))
            multiplier[np.isinf(multiplier)] = 0

            self.adj_matrix = self.adj_matrix.multiply(multiplier).tocsr()

    def forward(self, nodes1, nodes2):
        # print(self.adj_matrix)
        cn_score = np.array(np.sum(self.adj_matrix[nodes1].multiply(self.adj_matrix[nodes2]), 1)).flatten()

        return torch.tensor(cn_score)

