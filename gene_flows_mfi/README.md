

## Installation 


```
conda create -n discrete_flows python=3.11
python3 -m pip install -r requirements_linux.txt
```


## MFI calculation

Navigate to working directory `scripts/mfi/` and then run

```
python3 mfi_value_pred.py --sequence_length 1000 --batch_size 100000 --num_factors 2 --device cuda --save_every 1000 --value_to_save 0 --dtype float16 --compile
```

for each value (in this case, 0, 1 and 2), and then run 

```
python3 mfi_load_dict.py --sequence_length 1000 --batch_size 100000 --num_factors 2 --device cpu --save_every 1000
```