from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
from datasets import load_dataset
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from transformers import BitsAndBytesConfig
from dataset import build_dataset, collator
import wandb
from argparse import ArgumentParser

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    

def init_model_and_tokenizer(args):
    
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_module, 
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    pretrained_model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto")
    pretrained_model = get_peft_model(pretrained_model, lora_config)
    
    model = AutoModelForCausalLMWithValueHead.from_pretrained(pretrained_model)
    model.gradient_checkpointing_disable = model.pretrained_model.gradient_checkpointing_disable
    model.gradient_checkpointing_enable = model.pretrained_model.gradient_checkpointing_enable
    print_trainable_parameters(model)
    
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model_name)
    return model, ref_model, tokenizer


def train(args, model, ref_model, tokenizer, dataset, sentiment_pipe, sent_kwargs):
    
    ppo_config = PPOConfig(
        model_name=args.model_name, 
        learning_rate=args.learning_rate, 
        log_with="wandb"
    )
    
    wandb.login(key='a019fbecab1016fdc282c08a275a99adb885b41e')
    wandb.init(project="gpt_imdb")
    
    output_length_sampler = LengthSampler(4, 16)
    ppo_trainer = PPOTrainer(ppo_config, model, ref_model=ref_model, tokenizer=tokenizer, dataset=dataset, data_collator=collator)
    print('Training Start')
    for epoch in range(args.num_epochs):
        for batch in tqdm(ppo_trainer.dataloader):
            query_tensors = batch['input_ids']
            model.gradient_checkpointing_disable()
            model.pretrained_model.config.use_cache = True

            response_tensors = []
            for query in query_tensors:
                gen_len = output_length_sampler()
                generation_kwargs["max_new_tokens"] = gen_len
                response = ppo_trainer.generate(query, **generation_kwargs)
                response_tensors.append(response.squeeze()[-gen_len:])
            batch['response'] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

            #### Compute sentiment score
            texts = [q + r for q,r in zip(batch['query'], batch['response'])]
            pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
            rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]

            # Run PPO step
            model.gradient_checkpointing_enable()
            model.pretrained_model.config.use_cache = False

            #### Run PPO step 
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)

        
if __name__ == "__main__":
    parser = ArgumentParser(description="Train a model using Proximal Policy Optimization with optional LoRA.")
    parser.add_argument("--model_name", type=str, default="lvwerra/gpt2-imdb")
    parser.add_argument("--reward_model_path", type=str, default="distilbert-imdb")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--use_peft", type=bool, default=True)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_module", type=str, default="all-linear")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1.5e-5)
    parser.add_argument("--output_dir", type=str, default="gpt2-rlhf")
    parser.add_argument("--do_train", type=bool, default=True)
    args = parser.parse_args()
    
    sent_kwargs = {
        "return_all_scores": True, 
        "function_to_apply": "none", 
        "batch_size": args.batch_size 
    }
    
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    model, ref_model, tokenizer = init_model_and_tokenizer(args)
    dataset = build_dataset(args)
    sentiment_pipe = pipeline("sentiment-analysis", model=args.reward_model_path)
    if args.do_train:
        train(args, model, ref_model, tokenizer, dataset, sentiment_pipe, sent_kwargs)
        model.save_pretrained(args.output_dir, push_to_hub=False)