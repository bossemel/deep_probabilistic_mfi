

## Installation 


```
conda create -n discrete_flows python=3.11
python3 -m pip install -r requirements_linux.txt
```


## Scripts

### split_data.py

This scripts splits the data into train and test, using 40% of the "dev_astros_subset1_1000genes.csv" dataset. It defaults to random seed 4. Call using:

```
python3 scripts/split_data.py --subset_1_path data/raw/dev_astros_subset1_1000genes.csv --subset_2_path data/raw/dev_astros_subset2_1000genes.csv --output_directory data/processed/
```

The resulting hashes should be: 
```
sha1sum data/processed/train.csv  
95ddfa909702cc4c960925c7fc764d959baef2fa  data/processed/train.csv
```

```
sha1sum data/processed/test.csv  
8597612ee57d4f74b8ff04da1639131f6843dcce  data/processed/test.csv
```


## MFI calculation

Navigate to `scripts/mfi` and run

```
python3 mfi_value_pred.py --sequence_length 1000 --batch_size 100000 --num_factors 2 --device cuda --save_every 1000 --value_to_save 0 --dtype float16 --compile
```

for each value (in this case, 0, 1 and 2), and then run 

```
python3 mfi_assemble.py --sequence_length 1000 --batch_size 100000 --num_factors 2 --device cpu --save_every 1000
```
