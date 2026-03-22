import datasets
import json

ds = datasets.load_dataset('parquet', data_files='/arf/scratch/hvural/DeepEyes/caption_dataset_rl_v6_env_clean/train-00000-of-00008.parquet')['train']
with open('/arf/scratch/hvural/DeepEyes/debug_ds.json', 'w') as f:
    json.dump(ds[0], f, indent=4, default=str)
