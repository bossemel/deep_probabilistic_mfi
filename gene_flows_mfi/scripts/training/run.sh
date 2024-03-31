set -e
cd scripts
python3 made_genes_data.py --device cuda --log_dir ../tmp/run/made_bigger --batch_size 128 --use_wandb True --n_epochs 100 --weight_decay 0.000001 --hidden_dims 8000 --num_layers 2 --n_masks 2 --lr 0.01 --step_size 30
