import json
import os
import glob
import pandas as pd
from tqdm import tqdm

def main():
    json_path = '/arf/scratch/hvural/visual_grounded_dataset_merged.json'
    src_dir = '/arf/scratch/hvural/DeepEyes/caption_dataset_rl_v6_env_clean'
    dst_dir = '/arf/scratch/hvural/DeepEyes/caption_dataset_rl_v7'
    
    os.makedirs(dst_dir, exist_ok=True)

    # 1. Load and Group QA pairs
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    print(f"Loading QA pairs from {json_path}...")
    with open(json_path, 'r') as f:
        qa_data = json.load(f)
    
    image_to_qas = {}
    for entry in qa_data:
        img = str(entry['image']) # Ensure string
        if img not in image_to_qas:
            image_to_qas[img] = []
        image_to_qas[img].append(entry)

    # 2. Process Parquets
    parquet_files = sorted(glob.glob(os.path.join(src_dir, 'train-*.parquet')))
    if not parquet_files:
        print(f"Error: No parquets found in {src_dir}")
        return
        
    print(f"Found {len(parquet_files)} training parquets.")

    # Keep track of instance count per contest across all parquets
    contest_instance_counts = {}

    for p_file in tqdm(parquet_files, desc="Augmenting Parquets"):
        df = pd.read_parquet(p_file)
        
        extra_info_list = []
        for _, row in df.iterrows():
            c_num = str(row['contest_number'])
            
            # Get the next index for this contest
            idx = contest_instance_counts.get(c_num, 0)
            contest_instance_counts[c_num] = idx + 1
            
            # Map to one of the QA pairs
            qas = image_to_qas.get(c_num, [])
            
            extra_info = None
            if qas:
                # Use modulo if instances exceed available QAs (shouldn't happen per user context)
                qa = qas[idx % len(qas)]
                extra_info = {
                    "answer": qa['answer'],
                    "index": str(idx),
                    "question": qa['question'],
                    "split": "train",
                    "contest_number": c_num
                }
            
            extra_info_list.append(extra_info)
        
        df['extra_info'] = extra_info_list
        
        # Save to new location
        out_path = os.path.join(dst_dir, os.path.basename(p_file))
        df.to_parquet(out_path)

    print(f"Done! Augmented parquets saved to {dst_dir}")

if __name__ == "__main__":
    main()
