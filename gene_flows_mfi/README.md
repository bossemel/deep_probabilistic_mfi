

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

Run

```
python3 mfi_one_value.py --sequence_length 1000 --batch_size 100000 --num_factors 2 --device mps --save_every 1000 --value_to_save 0 > test_log.txt
```

for each value, and then run 

```
python3 mfi_load_dict.py --sequence_length 1000 --batch_size 100000 --num_factors 2 --device mps --save_every 1000
```