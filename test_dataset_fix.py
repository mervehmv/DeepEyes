import datasets
import os
import sys

# add verl to sys path if running from repo root
sys.path.append('/arf/scratch/hvural/DeepEyes')

from verl.utils.dataset.rl_dataset import RLHFDataset
from transformers import AutoProcessor, AutoTokenizer

def test():
    print("Loading tokenizer and processor...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    print("Initializing RLHFDataset...")
    ds = RLHFDataset(
        parquet_files=['/arf/scratch/hvural/DeepEyes/caption_dataset_rl_v6_env_clean/train-00000-of-00008.parquet'],
        tokenizer=tokenizer,
        prompt_key='problem',
        max_prompt_length=10240,
        filter_prompts=True,
        return_raw_chat=True,
        truncation='error',
        max_response_length=10240,
        custom_reward_fn=None,
        return_raw_input_ids=False,
        processor=processor
    )
    
    print(f"Dataset length: {len(ds)}")
    
    print("Getting first item...")
    item = ds[0]
    
    print("Success! Keys in item:", item.keys())
    print("Prompts input_ids shape:", item['input_ids'].shape)

if __name__ == "__main__":
    test()
