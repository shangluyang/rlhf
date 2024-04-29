import numpy as np
from datasets import load_dataset, load_metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding

def tokenize(examples):
    outputs = tokenizer(examples['text'], truncation=True, max_length=512)
    return outputs

def compute_metrics(eval_preds):
    metric = load_metric("accuracy")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


if __name__ == "__main__":
    ds = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
    tokenized_ds = ds.map(tokenize, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer)



    training_args = TrainingArguments(num_train_epochs=1,
                                      output_dir="distilbert-imdb",
                                      per_device_train_batch_size=16,
                                      per_device_eval_batch_size=16,
                                      evaluation_strategy="epoch",
                                      push_to_hub=False)



    trainer = Trainer(model=model, tokenizer=tokenizer,
                      data_collator=data_collator,
                      args=training_args,
                      train_dataset=tokenized_ds["train"],
                      eval_dataset=tokenized_ds["test"], 
                      compute_metrics=compute_metrics)
    trainer.train()
    trainer.save_model("distilbert-imdb")