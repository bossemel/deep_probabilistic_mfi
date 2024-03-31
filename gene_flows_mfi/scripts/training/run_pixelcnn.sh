et -e
cd scripts
python3 pixel_cnn_genes_data.py --device cuda --log_dir ../tmp/run/pixelcnn  --batch_size 128 --use_wandb True --n_epochs 100 --weight_decay 0.000001 --device cuda


