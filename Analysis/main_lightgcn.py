import time
from tqdm import tqdm
from parse_LightGCN import parse_args
import torch
import os
from torch.utils.data import DataLoader
from ogb.linkproppred import Evaluator

from utils import *
from model import LightGCN
from dataprocess import General_dataset, load_data, normalize_edge_gcn
from learn import train_lightgcn, eval_lightgcn, eval_lightgcn_comprehensive
import math
from torch_sparse import SparseTensor
import pickle as pkl


def run(encoder, edge, adj_list_dict, neg_edge, train_edge_index, train_edge_weight, test_node, args):
    pbar = tqdm(range(args.runs), unit='run')
    evaluator = Evaluator(name='ogbl-collab')

    train_hits, val_hits, test_hits = [], [], []

    if not args.load:
        for run in pbar:
            seed_everything(args.seed + run)

            encoder.reset_parameters()
            encoder = encoder.to(args.device)

            opt_encoder = torch.optim.Adam(
                encoder.parameters(), lr=args.encoder_lr)

            # Initilize dataloader for training train dataset
            train_train_dataset = General_dataset(
                n_edges=edge['Train'].shape[0])
            train_dataloader = DataLoader(train_train_dataset, batch_size=args.batch_size, shuffle=True,
                                          num_workers=args.num_workers, collate_fn=train_train_dataset.collate_batch, pin_memory=args.pin_memory)

            # Initilize dataloader for evaluating train/val/test datasets
            eval_pos_val_dataset, eval_pos_test_dataset = General_dataset(
                n_edges=edge['Val'].shape[0]), General_dataset(n_edges=edge['Test'].shape[0])
            eval_neg_train_dataset, eval_neg_val_dataset, eval_neg_test_dataset = General_dataset(n_edges=neg_edge['Train'].shape[0]), General_dataset(
                n_edges=neg_edge['Val'].shape[0]), General_dataset(n_edges=neg_edge['Test'].shape[0])
            eval_pos_dataloaders = {'Train': DataLoader(train_train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=train_train_dataset.collate_batch, pin_memory=args.pin_memory),
                                    'Val': DataLoader(eval_pos_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=eval_pos_val_dataset.collate_batch, pin_memory=args.pin_memory),
                                    'Test': DataLoader(eval_pos_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=eval_pos_test_dataset.collate_batch, pin_memory=args.pin_memory)}
            eval_neg_dataloaders = {'Train': DataLoader(eval_neg_train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=eval_neg_train_dataset.collate_batch, pin_memory=args.pin_memory),
                                    'Val': DataLoader(eval_neg_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=eval_neg_val_dataset.collate_batch, pin_memory=args.pin_memory),
                                    'Test': DataLoader(eval_neg_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=eval_neg_test_dataset.collate_batch, pin_memory=args.pin_memory)}

            train_adj_t = SparseTensor(
                row=train_edge_index[0], col=train_edge_index[1], value=train_edge_weight, is_sorted=False)

            train_adj_t = train_adj_t.to(args.device)

            if args.train:
                best_val_hit100 = -math.inf

                scores, scores2 = [], []
                for epoch in range(1, 1 + args.epochs):
                    loss = train_lightgcn(encoder, opt_encoder, train_adj_t, train_dataloader,
                                          edge['Train'], adj_list_dict['Train_val'], args)

                    if epoch % args.eval_steps == 0:
                        ress, score, score2 = eval_lightgcn(encoder, train_adj_t, eval_pos_dataloaders,
                                             eval_neg_dataloaders, evaluator, edge, neg_edge, adj_list_dict, args)

                        if ress['Val']['Hit@K'][4] > best_val_hit100:
                            best_val_hit100 = ress['Val']['Hit@K'][4]
                            test_hit = ress['Test']['Hit@K']
                            best_val_hit = ress['Val']['Hit@K']

                            # print(epoch, test_hit, best_val_hit)

                            if args.save:
                                torch.save(encoder.state_dict(), os.getcwd() + '/model/' + args.dataset +
                                           '/' + args.model_name + '/' + args.encoder_name + '_' + str(run) + '.pkl')
                        
                        scores.append(score)
                        scores2.append(score2)

                val_hits.append(best_val_hit)
                test_hits.append(test_hit)
            pkl.dump(scores, open('./res/' + args.dataset + '/' + args.model_name + '/' + str(run) + '_score.pkl', 'wb'))
            pkl.dump(scores2, open('./res/' + args.dataset + '/' + args.model_name + '/' + str(run) + '_score2.pkl', 'wb'))


            if not args.train and args.save:
                encoder.load_state_dict(torch.load(
                    os.getcwd() + '/model/' + args.dataset + '/' + args.model_name + '/' + args.encoder_name + '_' + str(run) + '.pkl'))

                ress = eval_lightgcn_comprehensive(encoder, train_adj_t, eval_pos_dataloaders,
                                                   eval_neg_dataloaders, evaluator, edge, neg_edge, adj_list_dict, test_node, args, run)

                train_hits.append(ress['Train']['Hit@K'])
                val_hits.append(ress['Val']['Hit@K'])
                test_hits.append(ress['Test']['Hit@K'])

                np.save('./res/' + args.dataset + '/' + args.model_name + '/train_link_level_performance.npy', train_hits)
                np.save('./res/' + args.dataset + '/' + args.model_name + '/val_link_level_performance.npy', val_hits)
                np.save('./res/' + args.dataset + '/' + args.model_name + '/test_link_level_performance.npy', test_hits)

                print(np.array(train_hits).mean(axis=0), np.array(train_hits).std(axis=0))
                print(np.array(val_hits).mean(axis=0), np.array(val_hits).std(axis=0))
                print(np.array(test_hits).mean(axis=0), np.array(test_hits).std(axis=0))

    else:
        recalls, ndcgs, hits, f1s, precisions, mrrs, scores = [], [], [], [], [], [], []

        for run in pbar:
            recalls.append(np.load('./res/' + args.dataset + '/' +
                                   args.model_name + '/' + str(run) + '_recall_list_' + args.eval_node_type + '.npy'))
            ndcgs.append(np.load('./res/' + args.dataset + '/' +
                                 args.model_name + '/' + str(run) + '_ndcg_list_' + args.eval_node_type + '.npy'))
            hits.append(np.load('./res/' + args.dataset + '/' +
                                args.model_name + '/' + str(run) + '_hit_ratio_list_' + args.eval_node_type + '.npy'))
            f1s.append(np.load('./res/' + args.dataset + '/' +
                               args.model_name + '/' + str(run) + '_F1_list_' + args.eval_node_type + '.npy'))
            precisions.append(np.load('./res/' + args.dataset + '/' +
                                      args.model_name + '/' + str(run) + '_precision_list_' + args.eval_node_type + '.npy'))
            mrrs.append(np.load('./res/' + args.dataset + '/' +
                                      args.model_name + '/' + str(run) + '_mrr_list_' + args.eval_node_type + '.npy'))


        metrics = [recalls, ndcgs, hits, f1s, precisions, mrrs]
        keys = ['Recall', 'NDCG', 'Hits', 'F1', 'Precision', 'MRR']

        print('Node-level evaluation')
        for key, metric in zip(keys, metrics):
            metric = np.stack(metric, axis=0)  # run * num_node * K
            metric = metric.mean(axis=1)

            print(key, metric.mean(axis=0), metric.std(axis=0))

        print('Link-level evaluation')
        link_train = np.load('./res/' + args.dataset + '/' + args.model_name + '/train_link_level_performance.npy')
        link_val = np.load('./res/' + args.dataset + '/' + args.model_name + '/val_link_level_performance.npy')
        link_test = np.load('./res/' + args.dataset + '/' + args.model_name + '/test_link_level_performance.npy')
        print('Train:', np.mean(link_train, axis = 0), np.std(link_train, axis = 0))
        print('Val:', np.mean(link_val, axis = 0), np.std(link_val, axis = 0))
        print('Test:', np.mean(link_test, axis = 0), np.std(link_test, axis = 0))


if __name__ == '__main__':
    args = parse_args()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.path = os.getcwd()

    seed_everything(args.seed)
    path_everything(args.dataset, args.model_name)

    """build dataset"""
    data, edge, adj_list_dict, neg_edge, test_node, eval_node, args.deg, args.tc, args.atc, args.density, args.n_nodes, args.n_edges = load_data(args)

    print('# of nodes:', args.n_nodes)
    print('# of edges:', args.n_edges)
    print('# of training edges:', edge['Train'].shape[0])
    print('# of validation edges:', edge['Val'].shape[0])
    print('# of testing edges:', edge['Test'].shape[0])
    print('Network density:', args.n_edges * 2 / (args.n_nodes * (args.n_nodes - 1)))

    for key in args.tc:
        print('TC-' + key, np.mean(np.array(args.tc[key])[test_node.numpy()]))

    clicked_set = adj_list_dict['Train_val']

    """build model"""
    encoder = LightGCN(args)

    train_edge_index = edge['Train']
    train_edge_index, train_edge_weight = normalize_edge_gcn(train_edge_index, args.n_nodes)

    run(encoder, edge, adj_list_dict,
        neg_edge, train_edge_index, train_edge_weight, test_node, args)
