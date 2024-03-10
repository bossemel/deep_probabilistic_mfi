
source ~/miniconda3/bin/activate discrete_flows
python3 mfi_load_dict.py --sequence_length 1000 --batch_size 100000 --num_factors 2 --device mps --save_every 1000