import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument("--dataset", nargs="?", default="Cora")

    # model
    parser.add_argument("--model", type=str, default="LightGCN")
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--n_hidden", type=int, default=64)

    # training
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=16 * 1024)
    parser.add_argument('--test_batch_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--pin_memory', type=int, default=0)
    parser.add_argument('--encoder_lr', type=float, default=5e-2)
    parser.add_argument('--dropout', type=float, default=0.0)

    # specific for LP
    parser.add_argument("--n_neg", type=int, default=1,
                        help="number of negative in K-pair loss")
    parser.add_argument('--topks', default=[5, 10, 20, 50, 100])

    # experiments
    parser.add_argument("--seed", type=int, default=1028,
                        help="seed to run the experiment")
    parser.add_argument("--encoder_name", type=str, default='encoder')
    parser.add_argument("--eval_steps", type=int, default=1)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument('--normalize_features', type=bool, default=True)
    parser.add_argument("--aggr", type=str, default='mean')
    parser.add_argument("--l2_coeff", type=float, default=0.05)
    parser.add_argument('--model_name', type=str, default='LightGCN')
    parser.add_argument('--eval', type=int, default=0)
    parser.add_argument("--save", action = 'store_true')
    parser.add_argument("--train", action = 'store_true')
    parser.add_argument("--load", action = 'store_true')
    parser.add_argument('--tc', type=str, default='tc')
    parser.add_argument('--tc_layer', type=int, default=1)
    parser.add_argument('--eval_node_type', type=str, default='Test')

    return parser.parse_args()
