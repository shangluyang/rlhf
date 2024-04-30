import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer
from dpo_dataset import load_train_dataset
from dpo_config import get_training_config

def main():
    parser = argparse.ArgumentParser(description="DPO Training Script")
    parser.add_argument('--file_path', type=str, required=True, help="File path for training data")
    parser.add_argument('--model_name_or_path', type=str, default='lvwerra/gpt2-imdb', help="Model name or path")
    parser.add_argument('--output_dir', type=str, default='dpo_anthropic_hh', help="Output directory")
    args = parser.parse_args()

    # Load datasets
    train_dataset = load_train_dataset(args.file_path)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.float32, use_cache=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get training config
    config = get_training_config(args.output_dir, 4, 1e-3, 1, 500, 10, 150)

    # Trainer setup
    trainer = DPOTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        tokenizer=tokenizer
    )

    # Training
    trainer.train()

    # Save the model
    trainer.save_model(args.output_dir)

if __name__ == '__main__':
    main()
