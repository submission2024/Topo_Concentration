import time
from tqdm import tqdm
from parse_GCN import parse_args
import torch
from torch import nn
import os
from torch.utils.data import DataLoader
from ogb.linkproppred import Evaluator

from utils import *
from model import GCN, LinkPredictor
from dataprocess import load_data, General_dataset, normalize_edge_gcn
from learn import train_gcn, eval_gcn, eval_gcn_comprehensive
import math
from torch_sparse import SparseTensor


def run(encoder, predictor, edge, adj_list_dict, neg_edge, data, test_node, args):
    pbar = tqdm(range(args.runs), unit='run')
    evaluator = Evaluator(name='ogbl-collab')

    train_hits, val_hits, test_hits = [], [], []

    if not args.load:
        for run in pbar:
            seed_everything(args.seed + run)

            encoder.reset_parameters()
            predictor.reset_parameters()

            if args.model in ['GCN']:
                opt_encoder = torch.optim.Adam(
                    encoder.parameters(), lr=args.encoder_lr)
                opt_predictor = torch.optim.Adam(
                    predictor.parameters(), lr=args.predictor_lr)

                opts = [opt_encoder, opt_predictor]

            # Initilize dataloader for training train dataset
            train_train_dataset = General_dataset(n_edges = edge['Train'].shape[0])
            train_dataloader = DataLoader(train_train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers, collate_fn = train_train_dataset.collate_batch, pin_memory = args.pin_memory)

            # Initilize dataloader for evaluating train/val/test datasets
            eval_pos_val_dataset, eval_pos_test_dataset = General_dataset(
                n_edges=edge['Val'].shape[0]), General_dataset(n_edges=edge['Test'].shape[0])

            # print(neg_edge)
            eval_neg_train_dataset, eval_neg_val_dataset, eval_neg_test_dataset = General_dataset(n_edges=neg_edge['Train'].shape[0]), General_dataset(
                n_edges=neg_edge['Val'].shape[0]), General_dataset(n_edges=neg_edge['Test'].shape[0])

            eval_pos_dataloaders = {'Train': DataLoader(train_train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=train_train_dataset.collate_batch, pin_memory=args.pin_memory),
                                    'Val': DataLoader(eval_pos_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=eval_pos_val_dataset.collate_batch, pin_memory=args.pin_memory),
                                    'Test': DataLoader(eval_pos_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=eval_pos_test_dataset.collate_batch, pin_memory=args.pin_memory)}
            eval_neg_dataloaders = {'Train': DataLoader(eval_neg_train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=eval_neg_train_dataset.collate_batch, pin_memory=args.pin_memory),
                                    'Val': DataLoader(eval_neg_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=eval_neg_val_dataset.collate_batch, pin_memory=args.pin_memory),
                                    'Test': DataLoader(eval_neg_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=eval_neg_test_dataset.collate_batch, pin_memory=args.pin_memory)}

            train_adj_t = SparseTensor(
                row=data.train_edge_index[0], col=data.train_edge_index[1], value=data.train_edge_weight, is_sorted=False)
            train_adj_t = train_adj_t.to(args.device)
            x = data.x.to(args.device)

            if args.train:
                best_val_hit100 = -math.inf

                for epoch in range(1, 1 + args.epochs):
                    loss = train_gcn(encoder, predictor, opts, train_adj_t, x, train_dataloader,
                                     edge['Train'], adj_list_dict['Train_val'], args)

                    if epoch % args.eval_steps == 0:
                        ress = eval_gcn(encoder, predictor, train_adj_t, x, eval_pos_dataloaders,
                                        eval_neg_dataloaders, evaluator, edge, neg_edge, test_node, adj_list_dict, args)

                        if ress['Val']['Hit@K'][4] > best_val_hit100:
                            best_val_hit = ress['Val']['Hit@K']
                            best_val_hit100 = ress['Val']['Hit@K'][4]
                            test_hit = ress['Test']['Hit@K']

                            # print(epoch, test_hit[-1], ress['Val']['Hit@K'][4])

                            if args.save:
                                torch.save(encoder.state_dict(), os.getcwd() + '/model/' + args.dataset + '/' + args.model_name + '/' + args.encoder_name + '_' + str(run) + '.pkl')
                                torch.save(predictor.state_dict(), os.getcwd() + '/model/' + args.dataset + '/' + args.model_name + '/' + args.predictor_name + '_' + str(run) + '.pkl')

                val_hits.append(best_val_hit)
                test_hits.append(test_hit)



            if not args.train and args.save:
                # # Final evaluation
                encoder.load_state_dict(torch.load(
                    os.getcwd() + '/model/' + args.dataset + '/' + args.model_name + '/' + args.encoder_name + '_' + str(run) + '.pkl'))
                predictor.load_state_dict(torch.load(
                    os.getcwd() + '/model/' + args.dataset + '/' + args.model_name + '/' + args.predictor_name + '_' + str(run) + '.pkl'))


                ress = eval_gcn_comprehensive(encoder, predictor, train_adj_t, x, eval_pos_dataloaders,
                                              eval_neg_dataloaders, evaluator, edge, neg_edge, test_node, adj_list_dict, args, run)

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
        recalls, ndcgs, hits, f1s, precisions, mrrs = [], [], [], [], [], []

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

        print('========Node-level evaluation=========')
        for key, metric in zip(keys, metrics):
            metric = np.stack(metric, axis=0)  # run * num_node * K
            metric = metric.mean(axis=1)

            print(key, metric.mean(axis=0), metric.std(axis=0))

        print('========Link-level evaluation========')
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
    data, edge, adj_list_dict, neg_edge, test_node, eval_node, args.deg, args.tc, args.density, args.n_nodes, args.n_edges = load_data(args)

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
    if args.model == 'GCN':
        encoder = GCN(data.x.shape[1], args.n_hidden, args.n_hidden,
                      args.n_layers, args.dropout).to(args.device)
        predictor = LinkPredictor(
            args.n_hidden, args.n_hidden, 1, args.n_layers, args.dropout).to(args.device)

    train_edge_index = edge['Train']
    val_edge_index = edge['Val']
    test_edge_index = edge['Test']

    data.train_edge_index, data.train_edge_weight = normalize_edge_gcn(
        train_edge_index, args.n_nodes)
    data.val_edge_index, data.val_edge_weight = normalize_edge_gcn(
        val_edge_index, args.n_nodes)
    data.test_edge_index, data.test_edge_weight = normalize_edge_gcn(
        test_edge_index, args.n_nodes)

    run(encoder, predictor, edge, adj_list_dict, neg_edge, data, test_node, args)
