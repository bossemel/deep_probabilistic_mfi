

## Installation 


```
conda create -n discrete_flows python=3.11
python3 -m pip install -r requirements_linux.txt
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