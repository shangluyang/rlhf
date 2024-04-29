from transformers import AutoTokenizer
import torch
from trl.core import LengthSampler
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset
import wandb


def build_dataset(args):
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token 

    ds = load_dataset("imdb", split='train') 
    ds = ds.rename_columns({'text': 'review'})
    
    # Filter reviews to be between 500 and 1000 characters
    ds = ds.filter(lambda x: 500 < len(x["review"]) <= 1000, batched=False)
    input_size = LengthSampler(2, 8)
    
    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[:input_size()] 
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type='torch', columns=["input_ids", "label"], output_all_columns=True)
    return ds


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])