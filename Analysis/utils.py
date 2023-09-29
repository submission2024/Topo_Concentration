import numpy as np
import random
import torch
import os
np.set_printoptions(precision=4)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def path_everything(dataset, model):
    if not os.path.exists('./data/' + dataset):
        os.mkdir('./data/' + dataset)

    if not os.path.exists('./res/' + dataset):
        os.mkdir('./res/' + dataset)
    if not os.path.exists('./res/' + dataset + '/' + model):
        os.mkdir('./res/' + dataset + '/' + model)

    if not os.path.exists('./model/' + dataset):
        os.mkdir('./model/' + dataset)
    if not os.path.exists('./model/' + dataset + '/' + model):
        os.mkdir('./model/' + dataset + '/' + model)

def batch_to_gpu(batch, device):
    for c in batch:
        batch[c] = batch[c].to(device)

    return batch


def getLabel(test_data, pred_data):
    r = []

    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)

    return np.array(r).astype('float')


def Hit_at_k(r, k):
    right_pred = r[:, :k].sum(axis=1)

    return 1. * (right_pred > 0)


def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    # print(right_pred, 2213123213)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = right_pred / recall_n
    recall[np.isnan(recall)] = 0
    precis = right_pred / precis_n
    return {'Recall': recall, 'Precision': precis}


def NDCGatK_r(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """

    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix

    # print(max_r[0], pred_data[0])
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = np.sum(pred_data * (1. / np.log2(np.arange(2, k + 2))), axis=1)

    idcg[idcg == 0.] = 1.  # it is OK since when idcg == 0, dcg == 0
    ndcg = dcg / idcg
    # ndcg[np.isnan(ndcg)] = 0.

    return ndcg


def test_one_batch(X, topks):
    sorted_items = X[0].numpy()
    groundTrue = X[1]

    r = getLabel(groundTrue, sorted_items)

    pre, recall, ndcg, hit_ratio, F1 = [], [], [], [], []
    for k in topks:
        ret = RecallPrecision_ATk(groundTrue, r, k)
        ndcgs = NDCGatK_r(groundTrue, r, k)
        hit_ratios = Hit_at_k(r, k)

        hit_ratio.append(sum(hit_ratios))
        pre.append(sum(ret['Precision']))
        recall.append(sum(ret['Recall']))
        ndcg.append(sum(ndcgs))

        temp = ret['Precision'] + ret['Recall']
        temp[temp == 0] = float('inf')
        F1s = 2 * ret['Precision'] * ret['Recall'] / temp
        # F1s[np.isnan(F1s)] = 0

        F1.append(sum(F1s))

    return {'Recall': np.array(recall),
            'Precision': np.array(pre),
            'NDCG': np.array(ndcg),
            'F1': np.array(F1),
            'Hit_ratio': np.array(hit_ratio)}


def test_one_batch_group(X, topks):
    sorted_items = X[0].numpy()
    groundTrue = X[1]

    r = getLabel(groundTrue, sorted_items)

    pres, recalls, ndcgs, hit_ratios, F1s = [], [], [], [], []
    for k in topks:
        ret = RecallPrecision_ATk(groundTrue, r, k)
        ndcgs.append(NDCGatK_r(groundTrue, r, k))
        hit_ratios.append(Hit_at_k(r, k))
        recalls.append(ret['Recall'])
        pres.append(ret['Precision'])

        temp = ret['Precision'] + ret['Recall']
        temp[temp == 0] = float('inf')
        F1s.append(2 * ret['Precision'] * ret['Recall'] / temp)
        # F1s[np.isnan(F1s)] = 0

    return np.stack(recalls).transpose(1, 0), np.stack(ndcgs).transpose(1, 0), np.stack(hit_ratios).transpose(1, 0), np.stack(pres).transpose(1, 0), np.stack(F1s).transpose(1, 0)


def cal_bpr_loss(user_embs, pos_item_embs, neg_item_embs):
    pos_scores = torch.sum(
        torch.mul(user_embs, pos_item_embs), axis=1)

    neg_scores = torch.sum(torch.mul(user_embs.unsqueeze(
        dim=1), neg_item_embs), axis=-1)

    bpr_loss = (torch.log(
        1 + torch.exp((neg_scores - pos_scores.unsqueeze(dim=1))).sum(dim=1))).mean()

    return torch.mean(bpr_loss)


def cal_l2_loss(user_embs, pos_item_embs, neg_item_embs, batch_size):
    return 0.5 * (user_embs.norm(2).pow(2) + pos_item_embs.norm(2).pow(2) + neg_item_embs.norm(2).pow(2)) / batch_size


def eval_group_loss_dist(rating_batch, batch_emb, interact_mask):
    rating_batch = rating_batch / batch_emb.norm(dim=1).view(-1, 1)

    pos_mask = (interact_mask == 1) * 1.
    neg_mask = (interact_mask == 0) * 1.

    score = (rating_batch * pos_mask).sum(dim=1) / pos_mask.sum(dim=1) - \
        (rating_batch * neg_mask).sum(dim=1) / neg_mask.sum(dim=1)

    return score


def eval_group_loss_dist2(rating_batch, interact_mask):
    pos_mask = (interact_mask == 1) * 1.
    neg_mask = (interact_mask == 0) * 1.

    score_pos_mean = (rating_batch * pos_mask).sum(dim=1) / pos_mask.sum(dim=1)
    score_neg_mean = (rating_batch * neg_mask).sum(dim=1) / neg_mask.sum(dim=1)

    return torch.log(1 + torch.exp((score_neg_mean - score_pos_mean)))
