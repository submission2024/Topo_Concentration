echo '======================COLLAB======================'
echo '*********************Train*********************'
python main.py --dataset='ogbl-collab' --n_layers=3 --n_hidden=256 --encoder_lr=0.001 --predictor_lr=0.001 --train --save --run=5 --runs=5
python main_lightgcn.py --dataset='ogbl-collab' --n_layers=1 --n_hidden=256 --encoder_lr=0.001 --train --run=1 --epoch=50

echo '*********************Eval*********************'
python main.py --dataset='ogbl-collab' --n_layers=3 --n_hidden=256 --encoder_lr=0.001 --predictor_lr=0.001 --eval_node_type='Train' --save --test_batch_size=24 --runs=5
python main.py --dataset='ogbl-collab' --n_layers=3 --n_hidden=256 --encoder_lr=0.001 --predictor_lr=0.001 --eval_node_type='Val' --save --test_batch_size=24 --runs=5
python main.py --dataset='ogbl-collab' --n_layers=3 --n_hidden=256 --encoder_lr=0.001 --predictor_lr=0.001 --eval_node_type='Test' --save --test_batch_size=24 --runs=5
python main_lightgcn.py --dataset='ogbl-collab' --n_layers=1 --n_hidden=256 --encoder_lr=0.001 --eval_node_type='Train' --save --test_batch_size=1024 --runs=5
python main_lightgcn.py --dataset='ogbl-collab' --n_layers=1 --n_hidden=256 --encoder_lr=0.001 --eval_node_type='Val' --save --test_batch_size=1024 --runs=5
python main_lightgcn.py --dataset='ogbl-collab' --n_layers=1 --n_hidden=256 --encoder_lr=0.001 --eval_node_type='Test' --save --test_batch_size=1024 --runs=5

echo '=======GCN-Train======='
python main.py --dataset='ogbl-collab' --n_layers=3 --n_hidden=256 --encoder_lr=0.001 --predictor_lr=0.001 --load --eval_node_type='Train' --runs=5
echo '=======GCN-Val======='
python main.py --dataset='ogbl-collab' --n_layers=3 --n_hidden=256 --encoder_lr=0.001 --predictor_lr=0.001 --load --eval_node_type='Val' --runs=5
echo '=======GCN-Test======='
python main.py --dataset='ogbl-collab' --n_layers=3 --n_hidden=256 --encoder_lr=0.001 --predictor_lr=0.001 --load --eval_node_type='Test' --runs=5
echo '=======LightGCN-Train======='
python main_lightgcn.py --dataset='ogbl-collab' --n_layers=1 --n_hidden=256 --encoder_lr=0.001 --load --eval_node_type='Train' --runs=5
echo '=======LightGCN-Val======='
python main_lightgcn.py --dataset='ogbl-collab' --n_layers=1 --n_hidden=256 --encoder_lr=0.001 --load --eval_node_type='Val' --runs=5
echo '=======LightGCN-Test======='
python main_lightgcn.py --dataset='ogbl-collab' --n_layers=1 --n_hidden=256 --encoder_lr=0.001 --load --eval_node_type='Test' --runs=5

