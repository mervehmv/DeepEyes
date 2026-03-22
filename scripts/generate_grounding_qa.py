import json
import os
import glob
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
import re

# Load environment variables (API Key)
# Explicitly pointing to the identified .env file
load_dotenv('/arf/scratch/hvural/research/.env')

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

if not DEEPSEEK_API_KEY:
    print("Error: DEEPSEEK_API_KEY not found in environment or .env file.")
    exit(1)

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

def generate_qa_pairs(visual_info):
    """
    Prompts DeepSeek to generate 6 QA pairs based on visual info.
    """
    prompt = f"""You are an expert AI assistant specialized in visual grounding and reasoning.
Based on the following detailed visual information about a cartoon/image, generate 6 diverse and specific question-answer pairs.

Visual Information:
{visual_info}

Task: Generate 6 NEW and DIFFERENT question-answer pairs.
The questions should focus on different elements of the scene, such as characters, actions, objects, and spatial relationships.
Ensure the answers are concise and directly supported by the visual information.

Output format: Return ONLY a JSON list of objects, each with "question" and "answer" keys.
Example:
[
  {{"question": "What is the character on the left doing?", "answer": "The character is reading a book."}},
  ...
]
"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates data in JSON format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        content = response.choices[0].message.content.strip()
        
        # Clean potential markdown code blocks
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        elif content.startswith("```"):
            content = content.replace("```", "").strip()
            
        data = json.loads(content)
        # DeepSeek might return a dict with a list, but we want the list
        if isinstance(data, dict):
            for val in data.values():
                if isinstance(val, list) and len(val) > 0 and 'question' in val[0]:
                    return val
        return data
    except Exception as e:
        print(f"Error generating QA pairs: {e}")
        return []

def main():
    json_path = '/arf/scratch/hvural/visual_grounded_dataset.json'
    output_path = '/arf/scratch/hvural/visual_grounded_dataset_merged.json'
    parquet_pattern = '/arf/scratch/hvural/DeepEyes/caption_dataset_rl_v6_env_clean/train-*-of-*.parquet'

    # 1. Load existing JSON
    print(f"Loading existing dataset from {json_path}...")
    with open(json_path, 'r') as f:
        grounding_data = json.load(f)

    # Group existing questions by image
    image_to_existing = {}
    for d in grounding_data:
        img = d['image']
        if img not in image_to_existing:
            image_to_existing[img] = []
        image_to_existing[img].append(d)

    print(f"Dataset has {len(grounding_data)} entries across {len(image_to_existing)} images.")

    # 2. Identify Training Set Contests
    train_parquets = glob.glob(parquet_pattern)
    print(f"Found {len(train_parquets)} training parquets.")

    training_contests = {}
    for p in train_parquets:
        df = pd.read_parquet(p)
        for _, row in df.iterrows():
            c_num = str(row['contest_number'])
            training_contests[c_num] = row.get('visual_info', '')

    print(f"Total distinct training contests found: {len(training_contests)}")

    # 3. Generation Loop
    new_qa_pairs = []
    
    # Process only contests that are NOT in the existing dataset
    missing_contests = {k: v for k, v in training_contests.items() if k not in image_to_existing}
    print(f"Number of contests missing from grounding dataset: {len(missing_contests)}")
    
    LIMIT = None  # FOR TRIAL RUN. Change to None for full run.
    count = 0
    for contest_num, visual_info in tqdm(missing_contests.items(), desc="Generating QA for missing contests"):
        if not visual_info:
            continue
            
        # Generate 6 new pairs
        generated = generate_qa_pairs(visual_info)
        
        if generated:
            for qa in generated[:6]:
                qa['image'] = contest_num
                new_qa_pairs.append(qa)
        
        count += 1
        if LIMIT and count >= LIMIT:
            break

    # 4. Merge and Final Save
    all_data = grounding_data + new_qa_pairs
    print(f"\nMerging complete.")
    print(f"New pairs added: {len(new_qa_pairs)}")
    print(f"Total entries: {len(all_data)}")
    
    with open(output_path, 'w') as f:
        json.dump(all_data, f, indent=2)
    print(f"Saved merged dataset to {output_path}")

if __name__ == "__main__":
    main()
