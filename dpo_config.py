from trl import DPOConfig

def get_training_config(output_dir, per_device_train_batch_size, learning_rate, gradient_accumulation_steps, eval_steps, logging_steps, warmup_steps):
    config = DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        warmup_steps=warmup_steps,
        per_device_eval_batch_size=16,
        num_train_epochs=4,
        do_train=True,
        do_eval=True,
        report_to='wandb'  # Adjust this as per your logging and tracking setup
    )
    return config
