import datasets
from transformers import RobertaTokenizerFast, DataCollatorWithPadding
import torch 
from torch.utils.data import DataLoader


GLUE_TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None)
}


def get_glue_dataloaders(task_name: str, 
                            model_name: str = "roberta-base", 
                            max_seq_length: int = 128, 
                            batch_size: int = 8):
    
    # Ensure task name is a valid GLUE (classification) task
    assert task_name in list(GLUE_TASK_TO_KEYS.keys()), "Please choose valid GLUE task"

    # Load in raw dataset and tokenizer 
    raw_dataset = datasets.load_dataset("glue", task_name)
    tokenizer = RobertaTokenizerFast.from_pretrained(model_name)

    # Define pre-processing / tokenization function
    def preprocess(batch):
        key1, key2 = GLUE_TASK_TO_KEYS[task_name]
        texts = (batch[key1],) if key2 is None else (batch[key1], batch[key2])

        return tokenizer(
            *texts,
            padding="max_length", 
            truncation=True, 
            max_length=max_seq_length
        )   
    
    # Tokenize dataset 
    dataset = raw_dataset.map(preprocess, batched=True)
    dataset = dataset.rename_column('label', 'labels')
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


    # Create dataloaders
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(
        dataset["train"],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator
    )

    val_loader = DataLoader(
        dataset["validation"],
        shuffle=True, 
        batch_size=batch_size,
        collate_fn=data_collator
    )

    return train_loader, val_loader