from utils import *
import torch
from torch.utils.data import DataLoader
from torch_scatter import scatter, scatter_max, scatter_add, scatter_mean
from collections import defaultdict
import pickle as pkl
import os
from torch_sparse import SparseTensor
import torch_sparse
from torch_sparse import SparseTensor
from torch_geometric.utils import softmax
from scipy.stats import rankdata


def train_gcn(encoder, predictor, opts, adj_t, x, dataloader, train_edge, train_val_adjlist, args):
    encoder.train()
    predictor.train()

    total_loss, count = 0, 0
    for batch in dataloader:
        node, pos_node = train_edge[batch, 0], train_edge[batch, 1]

        neg_node = torch.randint(
            0, args.n_nodes, size=(node.shape[0], args.n_neg))

        for j in range(node.shape[0]):
            for k in range(args.n_neg):
                while neg_node[j, k] in train_val_adjlist:
                    neg_node[j, k] = torch.randint(0, args.n_nodes)

        neg_node = neg_node.view(-1)

        h = encoder(x, adj_t)

        node_emb, pos_emb, neg_emb = h[node], h[pos_node], h[neg_node]

        pos_score = predictor(node_emb * pos_emb)
        pos_loss = -torch.log(pos_score + 1e-15).mean()

        neg_score = predictor(node_emb * neg_emb)
        neg_loss = -torch.log(1 - neg_score + 1e-15).mean()

        loss = pos_loss + neg_loss

        for opt in opts:
            opt.zero_grad()

        loss.backward()

        # torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        # torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        for opt in opts:
            opt.step()

        total_loss += loss.item() * node.shape[0]
        count += node.shape[0]

    return total_loss / count


@torch.no_grad()
def eval_gcn(encoder, predictor, adj_t, x, pos_dataloaders, neg_dataloaders, evaluator, edge, neg_edge, test_node, adj_list_dict, args):
    encoder.eval()
    predictor.eval()

    ress = {'Train': {'Hit@K': []},
            'Val': {'Hit@K': []},
            'Test': {'Hit@K': []}}

    h = encoder(x, adj_t)

    # eval_per_edge
    for key in pos_dataloaders:
        pos_preds = []
        neg_preds = []

        for batch in pos_dataloaders[key]:
            node, pos_node = edge[key][batch, 0], edge[key][batch, 1]
            node_emb, pos_emb = h[node], h[pos_node]

            pos_preds += [predictor(node_emb * pos_emb).squeeze().cpu()]

        pos_preds = torch.cat(pos_preds, dim=0)

        for batch in neg_dataloaders[key]:
            node, neg_node = neg_edge[key][batch, 0], neg_edge[key][batch, 1]
            node_emb, neg_emb = h[node], h[neg_node]

            neg_preds += [predictor(node_emb * neg_emb).squeeze().cpu()]

        neg_preds = torch.cat(neg_preds, dim=0)

        # How hit@K is computed
        top_neg_score = torch.topk(neg_preds, k=args.topks[-1])[0]
        for k in args.topks:
            threshold = top_neg_score[k - 1]
            mask = (pos_preds > threshold).float()

            ress[key]['Hit@K'].append((mask.sum() / pos_preds.shape[0]).item())

    return ress


@torch.no_grad()
def eval_gcn_comprehensive(encoder, predictor, train_adj_t, x, pos_dataloaders, neg_dataloaders, evaluator, edge, neg_edge, test_node, adj_list_dict, args, run):
    encoder.eval()
    predictor.eval()

    ress = {'Train': {'Hit@K': []},
            'Val': {'Hit@K': []},
            'Test': {'Hit@K': []}}

    with torch.no_grad():
        h = encoder(x, train_adj_t)

        # eval_per_edge
        for key in pos_dataloaders:
            pos_preds = []
            neg_preds = []

            for batch in pos_dataloaders[key]:
                node, pos_node = edge[key][batch, 0], edge[key][batch, 1]
                node_emb, pos_emb = h[node], h[pos_node]

                pos_preds += [predictor(node_emb * pos_emb).squeeze().cpu()]

            pos_preds = torch.cat(pos_preds, dim=0)

            for batch in neg_dataloaders[key]:
                node, neg_node = neg_edge[key][batch,
                                               0], neg_edge[key][batch, 1]
                node_emb, neg_emb = h[node], h[neg_node]

                neg_preds += [predictor(node_emb * neg_emb).squeeze().cpu()]

            neg_preds = torch.cat(neg_preds, dim=0)

            for k in args.topks:
                evaluator.K = k
                ress[key]['Hit@K'].append(evaluator.eval(
                    {'y_pred_pos': pos_preds, 'y_pred_neg': neg_preds})['hits@' + str(k)])

        # eval_per_node
        ratings_list = []
        groundTruth_nodes_list = []
        users_list = []
        mrr_list= []
        score_diff_pos_neg_list = []

        print(len(test_node) // args.test_batch_size)

        rating_batch_list = []

        for count, batch_node in enumerate(DataLoader(test_node, batch_size=args.test_batch_size, shuffle=False)):
            print(count)
            batch_emb = h[batch_node]

            simi = (batch_emb.unsqueeze(1) * h.unsqueeze(0)
                    ).view(batch_node.shape[0] * h.shape[0], -1)

            rating_batch = predictor(simi).detach().cpu().squeeze().view(
                batch_node.shape[0], h.shape[0])  # (bsz * node) * dim -> bsz*node -> bsz * node

            if args.eval_node_type == 'Train':
                clicked_nodes = [np.array([])]

                groundTruth_nodes = [list(adj_list_dict['Train'][node.item()]) for node in batch_node]

            elif args.eval_node_type == 'Val':
                clicked_nodes = [np.array(list(adj_list_dict['Train'][node.item()])) for node in batch_node]

                groundTruth_nodes = [list(adj_list_dict['Val'][node.item()]) for node in batch_node]

            elif args.eval_node_type == 'Test':
                clicked_nodes = [np.concatenate((list(adj_list_dict['Train'][node.item(
                )]), list(adj_list_dict['Val'][node.item()]))) for node in batch_node]

                groundTruth_nodes = [list(adj_list_dict['Test']
                                     [node.item()]) for node in batch_node]

            interaction_matrix = -np.ones((rating_batch.shape[0], h.shape[0]))


            exclude_index, exclude_nodes = [], []
            for range_i, nodes in enumerate(clicked_nodes):
                exclude_index.extend([range_i] * len(nodes))
                exclude_nodes.extend(nodes)

                interaction_matrix[range_i, nodes.astype(int)] = 0

            for range_i, nodes in enumerate(groundTruth_nodes):
                interaction_matrix[range_i, nodes] = 1

            #cal_score_diff
            pos_mask = (interaction_matrix == 1)*1.
            neg_mask = (interaction_matrix == -1)*1.
            pos_score_ave = (rating_batch*pos_mask).sum(axis = 1)/pos_mask.sum(axis = 1)
            neg_score_ave = (rating_batch*neg_mask).sum(axis = 1)/neg_mask.sum(axis = 1)


            if args.dataset != 'ogbl-collab':
                rating_batch[exclude_index, exclude_nodes] = -(1 << 10)

            rating_K = torch.topk(rating_batch, k=max(args.topks))[1]

            rating_ranking = rankdata(-np.array(rating_batch), method = 'ordinal', axis = 1)
            for i in range(len(groundTruth_nodes)):
                rank = min(rating_ranking[i][list(groundTruth_nodes[i])])
                mrr_list.append(1/rank)

            ratings_list.append(rating_K)
            groundTruth_nodes_list.append(groundTruth_nodes)
            users_list.append(batch_node.tolist())

            rating_batch_list.append(rating_K)


    rating_list = torch.cat(
        rating_batch_list, dim=0).detach().cpu().numpy()

    recall_list, ndcg_list, hit_ratio_list, precision_list, F1_list = [], [], [], [], []

    for users, X in zip(users_list, zip(ratings_list, groundTruth_nodes_list)):
        recalls, ndcgs, hit_ratios, precisions, F1s = test_one_batch_group(
            X, args.topks)

        recall_list.append(recalls)
        ndcg_list.append(ndcgs)
        hit_ratio_list.append(hit_ratios)
        precision_list.append(precisions)
        F1_list.append(F1s)

    recall_list = np.concatenate(recall_list)
    ndcg_list = np.concatenate(ndcg_list)
    hit_ratio_list = np.concatenate(hit_ratio_list)
    precision_list = np.concatenate(precision_list)
    F1_list = np.concatenate(F1_list)
    mrr_list = np.array(mrr_list)

    np.save(os.getcwd() + '/res/' + args.dataset + '/' + args.model_name + '/' + str(run) +
            '_recall_list_' + args.eval_node_type + '.npy', recall_list)
    np.save(os.getcwd() + '/res/' + args.dataset + '/' + args.model_name + '/' + str(run) +
            '_ndcg_list_' + args.eval_node_type + '.npy', ndcg_list)
    np.save(os.getcwd() + '/res/' + args.dataset + '/' + args.model_name + '/' + str(run) +
            '_hit_ratio_list_' + args.eval_node_type + '.npy', hit_ratio_list)
    np.save(os.getcwd() + '/res/' + args.dataset + '/' + args.model_name + '/' + str(run) +
            '_precision_list_' + args.eval_node_type + '.npy', precision_list)
    np.save(os.getcwd() + '/res/' + args.dataset + '/' +
            args.model_name + '/' + str(run) + '_F1_list_' + args.eval_node_type + '.npy', F1_list)
    np.save(os.getcwd() + '/res/' + args.dataset + '/' +
            args.model_name + '/' + str(run) + '_mrr_list_' + args.eval_node_type + '.npy', mrr_list)

    return ress






def train_lightgcn(encoder, optimizer_encoder, adj_sp_norm, dataloader, train_edge, train_val_adjlist, args):
    encoder.train()
    train_edge = train_edge.to(args.device)

    total_loss, count = 0, 0
    for batch in dataloader:
        node, pos_node = train_edge[batch, 0], train_edge[batch, 1]

        neg_node = torch.randint(
            0, args.n_nodes, size=(node.shape[0], args.n_neg))

        for j in range(node.shape[0]):
            for k in range(args.n_neg):
                while neg_node[j, k] in train_val_adjlist:
                    neg_node[j, k] = torch.randint(0, args.n_nodes)

        node_prop, pos_prop, neg_prop, node_0, pos_0, neg_0 = encoder(
            node, pos_node, neg_node, adj_sp_norm)

        bpr_loss = cal_bpr_loss(node_prop, pos_prop, neg_prop)
        l2_loss = cal_l2_loss(node_0, pos_0, neg_0, node_0.shape[0])

        loss = bpr_loss + args.l2_coeff * l2_loss

        optimizer_encoder.zero_grad()
        loss.backward()
        optimizer_encoder.step()

        total_loss += loss.item() * node.shape[0]
        count += node.shape[0]

    return total_loss / count


@torch.no_grad()
def eval_lightgcn(encoder, adj_sp_norm, pos_dataloaders, neg_dataloaders, evaluator, edge, neg_edge, adj_list_dict, args):
    encoder.eval()

    ress = {'Train': {'Hit@K': []},
            'Val': {'Hit@K': []},
            'Test': {'Hit@K': []}}

    h = encoder.generate(adj_sp_norm)
    score = {}
    score2 = {}

    # eval_per_edge
    for key in pos_dataloaders:
        pos_preds = []
        neg_preds = []
        
        tmp_edge = [[], []]
        for batch in pos_dataloaders[key]:
            node, pos_node = edge[key][batch, 0], edge[key][batch, 1]
            node_emb, pos_emb = h[node], h[pos_node]

            pos_preds += [(node_emb * pos_emb).sum(dim=1).cpu()]
            tmp_edge[0] += node.tolist()
            tmp_edge[1] += pos_node.tolist()

        pos_preds = torch.cat(pos_preds, dim=0)
        tmp_edge = torch.tensor(tmp_edge)

        tmp_edge2 = [[], []]
        for batch in neg_dataloaders[key]:
            node, neg_node = neg_edge[key][batch, 0], neg_edge[key][batch, 1]
            node_emb, neg_emb = h[node], h[neg_node]

            neg_preds += [(node_emb * neg_emb).sum(dim=1).cpu()]
            tmp_edge2[0] += node.tolist()
            tmp_edge2[1] += neg_node.tolist()

        neg_preds = torch.cat(neg_preds, dim=0)
        tmp_edge2 = torch.tensor(tmp_edge2)

        # How hit@K is computed
        top_neg_score = torch.topk(neg_preds, k=args.topks[-1])[0]
        for k in args.topks:
            threshold = top_neg_score[k - 1]
            mask = (pos_preds > threshold).float()

            ress[key]['Hit@K'].append((mask.sum() / pos_preds.shape[0]).item())

        
        head, tail = tmp_edge[0], tmp_edge[1]
        score[key] = scatter_mean(pos_preds, tail, dim = 0).numpy()

        head, tail = tmp_edge2[0], tmp_edge2[1]
        score2[key] = scatter_mean(neg_preds, tail, dim = 0).numpy()

    return ress, score, score2


@torch.no_grad()
def eval_lightgcn_comprehensive(encoder, adj_sp_norm, pos_dataloaders, neg_dataloaders, evaluator, edge, neg_edge, adj_list_dict, test_node, args, run):
    encoder.eval()

    ress = {'Train': {'Hit@K': []},
            'Val': {'Hit@K': []},
            'Test': {'Hit@K': []}}

    with torch.no_grad():
        h = encoder.generate(adj_sp_norm)

        # eval_per_edge
        for key in pos_dataloaders:
            pos_preds = []
            neg_preds = []

            for batch in pos_dataloaders[key]:
                node, pos_node = edge[key][batch, 0], edge[key][batch, 1]
                node_emb, pos_emb = h[node], h[pos_node]

                pos_preds += [(node_emb * pos_emb).sum(dim=1).cpu()]

            pos_preds = torch.cat(pos_preds, dim=0)

            for batch in neg_dataloaders[key]:
                node, neg_node = neg_edge[key][batch,
                                               0], neg_edge[key][batch, 1]
                node_emb, neg_emb = h[node], h[neg_node]

                neg_preds += [(node_emb * neg_emb).sum(dim=1).cpu()]

            neg_preds = torch.cat(neg_preds, dim=0)

            for k in args.topks:
                evaluator.K = k
                ress[key]['Hit@K'].append(evaluator.eval(
                    {'y_pred_pos': pos_preds, 'y_pred_neg': neg_preds})['hits@' + str(k)])

        # eval_per_node
        ratings_list = []
        groundTruth_nodes_list = []
        users_list = []
        mrr_list= []

        rating_batch_list = []

        for count, batch_node in enumerate(DataLoader(test_node, batch_size=args.test_batch_size, shuffle=False)):
            # print(count)
            batch_emb = h[batch_node]

            rating_batch = torch.matmul(batch_emb, h.t()).detach().cpu()

            if args.eval_node_type == 'Train':
                clicked_nodes = [np.array([])]

                groundTruth_nodes = [list(adj_list_dict['Train'][node.item()]) for node in batch_node]

            elif args.eval_node_type == 'Val':
                clicked_nodes = [np.array(list(adj_list_dict['Train'][node.item()])) for node in batch_node]

                groundTruth_nodes = [list(adj_list_dict['Val'][node.item()]) for node in batch_node]

            elif args.eval_node_type == 'Test':
                clicked_nodes = [np.concatenate((list(adj_list_dict['Train'][node.item(
                )]), list(adj_list_dict['Val'][node.item()]))) for node in batch_node]

                groundTruth_nodes = [list(adj_list_dict['Test']
                                     [node.item()]) for node in batch_node]

            interaction_matrix = -np.ones((rating_batch.shape[0], h.shape[0]))
            exclude_index, exclude_nodes = [], []
            for range_i, nodes in enumerate(clicked_nodes):
                exclude_index.extend([range_i] * len(nodes))
                exclude_nodes.extend(nodes)

                interaction_matrix[range_i, nodes.astype(int)] = 0

            for range_i, nodes in enumerate(groundTruth_nodes):
                # print(range_i, nodes)
                interaction_matrix[range_i, nodes] = 1

            if args.dataset != 'ogbl-collab':
                rating_batch[exclude_index, exclude_nodes] = -(1 << 10)
            
            rating_K = torch.topk(rating_batch, k=max(args.topks))[1]

            rating_ranking = rankdata(-np.array(rating_batch), method = 'ordinal', axis = 1)
            for i in range(len(groundTruth_nodes)):
                rank = min(rating_ranking[i][list(groundTruth_nodes[i])])
                mrr_list.append(1/rank)


            ratings_list.append(rating_K)
            groundTruth_nodes_list.append(groundTruth_nodes)
            users_list.append(batch_node.tolist())

            rating_batch_list.append(rating_K)


    rating_list = torch.cat(rating_batch_list, dim=0).detach().cpu().numpy()
    recall_list, ndcg_list, hit_ratio_list, precision_list, F1_list = [], [], [], [], []

    for users, X in zip(users_list, zip(ratings_list, groundTruth_nodes_list)):
        recalls, ndcgs, hit_ratios, precisions, F1s = test_one_batch_group(
            X, args.topks)

        recall_list.append(recalls)
        ndcg_list.append(ndcgs)
        hit_ratio_list.append(hit_ratios)
        precision_list.append(precisions)
        F1_list.append(F1s)

    recall_list = np.concatenate(recall_list)
    ndcg_list = np.concatenate(ndcg_list)
    hit_ratio_list = np.concatenate(hit_ratio_list)
    precision_list = np.concatenate(precision_list)
    F1_list = np.concatenate(F1_list)
    mrr_list = np.array(mrr_list)


    np.save(os.getcwd() + '/res/' + args.dataset + '/' + args.model_name + '/' + str(run) +
            '_recall_list_' + args.eval_node_type + '.npy', recall_list)
    np.save(os.getcwd() + '/res/' + args.dataset + '/' + args.model_name + '/' + str(run) +
            '_ndcg_list_' + args.eval_node_type + '.npy', ndcg_list)
    np.save(os.getcwd() + '/res/' + args.dataset + '/' + args.model_name + '/' + str(run) +
            '_hit_ratio_list_' + args.eval_node_type + '.npy', hit_ratio_list)
    np.save(os.getcwd() + '/res/' + args.dataset + '/' + args.model_name + '/' + str(run) +
            '_precision_list_' + args.eval_node_type + '.npy', precision_list)
    np.save(os.getcwd() + '/res/' + args.dataset + '/' +
            args.model_name + '/' + str(run) + '_F1_list_' + args.eval_node_type + '.npy', F1_list)
    np.save(os.getcwd() + '/res/' + args.dataset + '/' +
            args.model_name + '/' + str(run) + '_mrr_list_' + args.eval_node_type + '.npy', mrr_list)

    return ress





def eval_heuristics_comprehensive(predictor, pos_dataloaders, neg_dataloaders, evaluator, edge, neg_edge, test_node, adj_list_dict, args):
    ress = {'Train': {'Hit@K': []},
            'Val': {'Hit@K': []},
            'Test': {'Hit@K': []}}

    with torch.no_grad():
        # eval_per_edge
        for key in pos_dataloaders:
            pos_preds = []
            neg_preds = []

            for batch in pos_dataloaders[key]:
                node, pos_node = edge[key][batch, 0], edge[key][batch, 1]
                scores = predictor(node, pos_node)

                pos_preds += [scores]

            pos_preds = torch.cat(pos_preds, dim=0)

            for batch in neg_dataloaders[key]:
                node, neg_node = neg_edge[key][batch, 0], neg_edge[key][batch, 1]
                scores = predictor(node, neg_node)

                neg_preds += [scores]

            neg_preds = torch.cat(neg_preds, dim=0)

            for k in args.topks:
                evaluator.K = k
                ress[key]['Hit@K'].append(evaluator.eval(
                    {'y_pred_pos': pos_preds, 'y_pred_neg': neg_preds})['hits@' + str(k)])

        # eval_per_node
        ratings_list = []
        groundTruth_nodes_list = []
        users_list = []
        mrr_list= []

        rating_batch_list = []

        full_nodes = torch.arange(args.n_nodes)

        for count, batch_node in enumerate(DataLoader(test_node, batch_size=args.test_batch_size, shuffle=False)):
            print(count)
            a_grid, b_grid = torch.meshgrid(batch_node, full_nodes)
            node_pairs = torch.stack((a_grid, b_grid), dim=-1).view(-1, 2)

            rating_batch = predictor(node_pairs[:, 0], node_pairs[:, 1]).view(
                batch_node.shape[0], -1)


            if args.eval_node_type == 'Train':
                clicked_nodes = [np.array([])]

                groundTruth_nodes = [list(adj_list_dict['Train'][node.item()]) for node in batch_node]

            elif args.eval_node_type == 'Val':
                clicked_nodes = [np.array(list(adj_list_dict['Train'][node.item()])) for node in batch_node]

                groundTruth_nodes = [list(adj_list_dict['Val'][node.item()]) for node in batch_node]

            elif args.eval_node_type == 'Test':
                clicked_nodes = [np.concatenate((list(adj_list_dict['Train'][node.item(
                )]), list(adj_list_dict['Val'][node.item()]))) for node in batch_node]

                groundTruth_nodes = [list(adj_list_dict['Test']
                                     [node.item()]) for node in batch_node]

            exclude_index, exclude_nodes = [], []
            for range_i, nodes in enumerate(clicked_nodes):
                exclude_index.extend([range_i] * len(nodes))
                exclude_nodes.extend(nodes)


            if args.dataset not in  ['ogbl-collab']:
                rating_batch[exclude_index, exclude_nodes] = -(1 << 10)
            rating_K = torch.topk(rating_batch, k=max(args.topks))[1]

            rating_ranking = rankdata(-np.array(rating_batch), method = 'ordinal', axis = 1)
            for i in range(len(groundTruth_nodes)):
                rank = min(rating_ranking[i][list(groundTruth_nodes[i])])
                mrr_list.append(1/rank)


            ratings_list.append(rating_K)
            groundTruth_nodes_list.append(groundTruth_nodes)
            users_list.append(batch_node.tolist())

            rating_batch_list.append(rating_K)


        rating_list = torch.cat(rating_batch_list, dim=0).detach().cpu().numpy()
        recall_list, ndcg_list, hit_ratio_list, precision_list, F1_list = [], [], [], [], []

        for users, X in zip(users_list, zip(ratings_list, groundTruth_nodes_list)):
            recalls, ndcgs, hit_ratios, precisions, F1s = test_one_batch_group(
                X, args.topks)

            recall_list.append(recalls)
            ndcg_list.append(ndcgs)
            hit_ratio_list.append(hit_ratios)
            precision_list.append(precisions)
            F1_list.append(F1s)

        recall_list = np.concatenate(recall_list)
        ndcg_list = np.concatenate(ndcg_list)
        hit_ratio_list = np.concatenate(hit_ratio_list)
        precision_list = np.concatenate(precision_list)
        F1_list = np.concatenate(F1_list)
        mrr_list = np.array(mrr_list)


        np.save(os.getcwd() + '/res/' + args.dataset + '/' + args.model_name + '/recall_list_' + args.eval_node_type + '.npy', recall_list)
        np.save(os.getcwd() + '/res/' + args.dataset + '/' + args.model_name + '/ndcg_list_' + args.eval_node_type + '.npy', ndcg_list)
        np.save(os.getcwd() + '/res/' + args.dataset + '/' + args.model_name + '/hit_ratio_list_' + args.eval_node_type + '.npy', hit_ratio_list)
        np.save(os.getcwd() + '/res/' + args.dataset + '/' + args.model_name + '/precision_list_' + args.eval_node_type + '.npy', precision_list)
        np.save(os.getcwd() + '/res/' + args.dataset + '/' + args.model_name + '/F1_list_' + args.eval_node_type + '.npy', F1_list)

        np.save(os.getcwd() + '/res/' + args.dataset + '/' + args.model_name + '/mrr_list_' + args.eval_node_type + '.npy', mrr_list)

    return ress
