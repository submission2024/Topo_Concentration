echo '======================CORA======================'

echo '*********************Train*********************'
python main.py --dataset='Cora' --n_layers=1 --n_hidden=256 --encoder_lr=0.001 --predictor_lr=0.01 --train --save
python main_lightgcn.py --dataset='Cora' --n_layers=2 --n_hidden=128 --encoder_lr=0.05 --l2_coeff=0.05 --train --save

echo '*********************Eval*********************'
python main.py --dataset='Cora' --n_layers=1 --n_hidden=256 --encoder_lr=0.001 --predictor_lr=0.01 --eval_node_type='Train' --save
python main.py --dataset='Cora' --n_layers=1 --n_hidden=256 --encoder_lr=0.001 --predictor_lr=0.01 --eval_node_type='Val' --save
python main.py --dataset='Cora' --n_layers=1 --n_hidden=256 --encoder_lr=0.001 --predictor_lr=0.01 --eval_node_type='Test' --save
python main_lightgcn.py --dataset='Cora' --n_layers=2 --n_hidden=128 --encoder_lr=0.05 --l2_coeff=0.05 --eval_node_type='Train' --save
python main_lightgcn.py --dataset='Cora' --n_layers=2 --n_hidden=128 --encoder_lr=0.05 --l2_coeff=0.05 --eval_node_type='Val' --save
python main_lightgcn.py --dataset='Cora' --n_layers=2 --n_hidden=128 --encoder_lr=0.05 --l2_coeff=0.05 --eval_node_type='Test' --save

echo '=======GCN-Train======='
python main.py --dataset='Cora' --n_layers=1 --n_hidden=256 --encoder_lr=0.001 --predictor_lr=0.01 --load --eval_node_type='Train'
echo '=======GCN-Val======='
python main.py --dataset='Cora' --n_layers=1 --n_hidden=256 --encoder_lr=0.001 --predictor_lr=0.01 --load --eval_node_type='Val'
echo '=======GCN-Test======='
python main.py --dataset='Cora' --n_layers=1 --n_hidden=256 --encoder_lr=0.001 --predictor_lr=0.01 --load --eval_node_type='Test'
echo '=======LightGCN-Train======='
python main_lightgcn.py --dataset='Cora' --n_layers=2 --n_hidden=128 --encoder_lr=0.05 --l2_coeff=0.05 --load --eval_node_type='Train'
echo '=======LightGCN-Val======='
python main_lightgcn.py --dataset='Cora' --n_layers=2 --n_hidden=128 --encoder_lr=0.05 --l2_coeff=0.05 --load --eval_node_type='Val'
echo '=======LightGCN-Test======='
python main_lightgcn.py --dataset='Cora' --n_layers=2 --n_hidden=128 --encoder_lr=0.05 --l2_coeff=0.05 --load --eval_node_type='Test'




echo '======================Citeseer======================'
echo '*********************Train*********************'
python main.py --dataset='Citeseer' --n_layers=1 --n_hidden=256 --encoder_lr=0.01 --predictor_lr=0.01 --train --save
python main_lightgcn.py --dataset='Citeseer' --n_layers=2 --n_hidden=256 --encoder_lr=0.05 --l2_coeff=0.05 --train --save

echo '*********************Eval*********************'
python main.py --dataset='Citeseer' --n_layers=1 --n_hidden=256 --encoder_lr=0.01 --predictor_lr=0.01 --eval_node_type='Train' --save
python main.py --dataset='Citeseer' --n_layers=1 --n_hidden=256 --encoder_lr=0.01 --predictor_lr=0.01 --eval_node_type='Val' --save
python main.py --dataset='Citeseer' --n_layers=1 --n_hidden=256 --encoder_lr=0.01 --predictor_lr=0.01 --eval_node_type='Test' --save
python main_lightgcn.py --dataset='Citeseer' --n_layers=2 --n_hidden=256 --encoder_lr=0.05 --l2_coeff=0.05 --eval_node_type='Train' --save
python main_lightgcn.py --dataset='Citeseer' --n_layers=2 --n_hidden=256 --encoder_lr=0.05 --l2_coeff=0.05 --eval_node_type='Val' --save
python main_lightgcn.py --dataset='Citeseer' --n_layers=2 --n_hidden=256 --encoder_lr=0.05 --l2_coeff=0.05 --eval_node_type='Test' --save

echo '=======GCN-Train======='
python main.py --dataset='Citeseer' --n_layers=1 --n_hidden=256 --encoder_lr=0.01 --predictor_lr=0.01 --load --eval_node_type='Train'
echo '=======GCN-Val======='
python main.py --dataset='Citeseer' --n_layers=1 --n_hidden=256 --encoder_lr=0.01 --predictor_lr=0.01 --load --eval_node_type='Val'
echo '=======GCN-Test======='
python main.py --dataset='Citeseer' --n_layers=1 --n_hidden=256 --encoder_lr=0.01 --predictor_lr=0.01 --load --eval_node_type='Test'
echo '=======LightGCN-Train======='
python main_lightgcn.py --dataset='Citeseer' --n_layers=2 --n_hidden=256 --encoder_lr=0.05 --l2_coeff=0.05 --load --eval_node_type='Train'
echo '=======LightGCN-Val======='
python main_lightgcn.py --dataset='Citeseer' --n_layers=2 --n_hidden=256 --encoder_lr=0.05 --l2_coeff=0.05 --load --eval_node_type='Val'
echo '=======LightGCN-Test======='
python main_lightgcn.py --dataset='Citeseer' --n_layers=2 --n_hidden=256 --encoder_lr=0.05 --l2_coeff=0.05 --load --eval_node_type='Test'









echo '======================Pubmed======================'
echo '*********************Train*********************'
python main.py --dataset='Pubmed' --n_layers=2 --n_hidden=256 --encoder_lr=0.001 --predictor_lr=0.01 --train --save
python main_lightgcn.py --dataset='Pubmed' --n_layers=1 --n_hidden=64 --encoder_lr=0.01 --l2_coeff=0.01 --train --save

echo '*********************Eval*********************'
python main.py --dataset='Pubmed' --n_layers=2 --n_hidden=256 --encoder_lr=0.001 --predictor_lr=0.01 --eval_node_type='Train' --save --test_batch_size=24
python main.py --dataset='Pubmed' --n_layers=2 --n_hidden=256 --encoder_lr=0.001 --predictor_lr=0.01 --eval_node_type='Val' --save --test_batch_size=24
python main.py --dataset='Pubmed' --n_layers=2 --n_hidden=256 --encoder_lr=0.001 --predictor_lr=0.01 --eval_node_type='Test' --save --test_batch_size=24
python main_lightgcn.py --dataset='Pubmed' --n_layers=1 --n_hidden=64 --encoder_lr=0.01 --l2_coeff=0.01 --eval_node_type='Train' --save
python main_lightgcn.py --dataset='Pubmed' --n_layers=1 --n_hidden=64 --encoder_lr=0.01 --l2_coeff=0.01 --eval_node_type='Val' --save
python main_lightgcn.py --dataset='Pubmed' --n_layers=1 --n_hidden=64 --encoder_lr=0.01 --l2_coeff=0.01 --eval_node_type='Test' --save

echo '=======GCN-Train======='
python main.py --dataset='Pubmed' --n_layers=2 --n_hidden=256 --encoder_lr=0.001 --predictor_lr=0.01 --load --eval_node_type='Train'
echo '=======GCN-Val======='
python main.py --dataset='Pubmed' --n_layers=2 --n_hidden=256 --encoder_lr=0.001 --predictor_lr=0.01 --load --eval_node_type='Val'
echo '=======GCN-Test======='
python main.py --dataset='Pubmed' --n_layers=2 --n_hidden=256 --encoder_lr=0.001 --predictor_lr=0.01 --load --eval_node_type='Test'
echo '=======LightGCN-Train======='
python main_lightgcn.py --dataset='Pubmed' --n_layers=1 --n_hidden=64 --encoder_lr=0.01 --l2_coeff=0.01 --load --eval_node_type='Train'
echo '=======LightGCN-Val======='
python main_lightgcn.py --dataset='Pubmed' --n_layers=1 --n_hidden=64 --encoder_lr=0.01 --l2_coeff=0.01 --load --eval_node_type='Val'
echo '=======LightGCN-Test======='
python main_lightgcn.py --dataset='Pubmed' --n_layers=1 --n_hidden=64 --encoder_lr=0.01 --l2_coeff=0.01 --load --eval_node_type='Test'









echo '======================Wiki_co_read======================'
echo '*********************Train*********************'
python main.py --dataset='Wiki_co_read' --n_layers=3 --n_hidden=64 --encoder_lr=0.001 --predictor_lr=0.001 --train --save --runs=5
python main_lightgcn.py --dataset='Wiki_co_read' --n_layers=1 --n_hidden=64 --encoder_lr=0.001 --train --save --runs=5

echo '*********************Eval*********************'
python main.py --dataset='Wiki_co_read' --n_layers=3 --n_hidden=64 --encoder_lr=0.001 --predictor_lr=0.001 --eval_node_type='Train' --save --test_batch_size=24
python main.py --dataset='Wiki_co_read' --n_layers=3 --n_hidden=64 --encoder_lr=0.001 --predictor_lr=0.001 --eval_node_type='Val' --save --test_batch_size=24
python main.py --dataset='Wiki_co_read' --n_layers=3 --n_hidden=64 --encoder_lr=0.001 --predictor_lr=0.001 --eval_node_type='Test' --save --test_batch_size=24
python main_lightgcn.py --dataset='Wiki_co_read' --n_layers=1 --n_hidden=64 --encoder_lr=0.001 --eval_node_type='Train' --save
python main_lightgcn.py --dataset='Wiki_co_read' --n_layers=1 --n_hidden=64 --encoder_lr=0.001 --eval_node_type='Val' --save
python main_lightgcn.py --dataset='Wiki_co_read' --n_layers=1 --n_hidden=64 --encoder_lr=0.001 --eval_node_type='Test' --save

echo '=======GCN-Train======='
python main.py --dataset='Wiki_co_read' --n_layers=3 --n_hidden=64 --encoder_lr=0.001 --predictor_lr=0.001 --load --eval_node_type='Train'
echo '=======GCN-Val======='
python main.py --dataset='Wiki_co_read' --n_layers=3 --n_hidden=64 --encoder_lr=0.001 --predictor_lr=0.001 --load --eval_node_type='Val'
echo '=======GCN-Test======='
python main.py --dataset='Wiki_co_read' --n_layers=3 --n_hidden=64 --encoder_lr=0.001 --predictor_lr=0.001 --load --eval_node_type='Test'
echo '=======LightGCN-Train======='
python main_lightgcn.py --dataset='Wiki_co_read' --n_layers=1 --n_hidden=64 --encoder_lr=0.001 --load --eval_node_type='Train'
echo '=======LightGCN-Val======='
python main_lightgcn.py --dataset='Wiki_co_read' --n_layers=1 --n_hidden=64 --encoder_lr=0.001 --load --eval_node_type='Val'
echo '=======LightGCN-Test======='
python main_lightgcn.py --dataset='Wiki_co_read' --n_layers=1 --n_hidden=64 --encoder_lr=0.001 --load --eval_node_type='Test'








echo '======================Reptile======================'
echo '*********************Train*********************'
python main_lightgcn.py --dataset='Reptile' --n_layers=1 --n_hidden=64 --encoder_lr=0.01 --l2_coeff=0.05 --train --save

echo '*********************Eval*********************'
python main_lightgcn.py --dataset='Reptile' --n_layers=1 --n_hidden=64 --encoder_lr=0.01 --l2_coeff=0.05 --eval_node_type='Train' --save
python main_lightgcn.py --dataset='Reptile' --n_layers=1 --n_hidden=64 --encoder_lr=0.01 --l2_coeff=0.05 --eval_node_type='Val' --save
python main_lightgcn.py --dataset='Reptile' --n_layers=1 --n_hidden=64 --encoder_lr=0.01 --l2_coeff=0.05 --eval_node_type='Test' --save

echo '=======LightGCN-Train======='
python main_lightgcn.py --dataset='Reptile' --n_layers=1 --n_hidden=64 --encoder_lr=0.01 --l2_coeff=0.05 --load --eval_node_type='Train'
echo '=======LightGCN-Val======='
python main_lightgcn.py --dataset='Reptile' --n_layers=1 --n_hidden=64 --encoder_lr=0.01 --l2_coeff=0.05 --load --eval_node_type='Val'
echo '=======LightGCN-Test======='
python main_lightgcn.py --dataset='Reptile' --n_layers=1 --n_hidden=64 --encoder_lr=0.01 --l2_coeff=0.05 --load --eval_node_type='Test'





echo '======================Vole======================'
echo '*********************Train*********************'
python main_lightgcn.py --dataset='Vole' --n_layers=1 --n_hidden=64 --encoder_lr=0.01 --l2_coeff=0.05 --train --save

echo '*********************Eval*********************'
python main_lightgcn.py --dataset='Vole' --n_layers=1 --n_hidden=64 --encoder_lr=0.01 --l2_coeff=0.05 --eval_node_type='Train' --save
python main_lightgcn.py --dataset='Vole' --n_layers=1 --n_hidden=64 --encoder_lr=0.01 --l2_coeff=0.05 --eval_node_type='Val' --save
python main_lightgcn.py --dataset='Vole' --n_layers=1 --n_hidden=64 --encoder_lr=0.01 --l2_coeff=0.05 --eval_node_type='Test' --save

echo '=======LightGCN-Train======='
python main_lightgcn.py --dataset='Vole' --n_layers=1 --n_hidden=64 --encoder_lr=0.01 --l2_coeff=0.05 --load --eval_node_type='Train'
echo '=======LightGCN-Val======='
python main_lightgcn.py --dataset='Vole' --n_layers=1 --n_hidden=64 --encoder_lr=0.01 --l2_coeff=0.05 --load --eval_node_type='Val'
echo '=======LightGCN-Test======='
python main_lightgcn.py --dataset='Vole' --n_layers=1 --n_hidden=64 --encoder_lr=0.01 --l2_coeff=0.05 --load --eval_node_type='Test'







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

