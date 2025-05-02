import torch
from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("daily_dialog", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("t5-small")

def build_input_response_pairs(dataset):
    input_response_pairs = []
    for dialog in dataset["dialog"]:
        for i in range(len(dialog) - 1):
            input_utterance = dialog[i].strip()
            response_utterance = dialog[i + 1].strip()
            input_response_pairs.append((input_utterance, response_utterance))
    return input_response_pairs

train_pairs = build_input_response_pairs(dataset["train"])
val_pairs = build_input_response_pairs(dataset["validation"])
test_pairs = build_input_response_pairs(dataset["test"])

class DailyDialogPairs(torch.utils.data.Dataset):
    def __init__(self, pairs, tokenizer, max_length=128):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, index):
        input, response = self.pairs[index]
        input_encoding = self.tokenizer(
            input,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        response_encoding = self.tokenizer(
            response,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "input_padding_mask": input_encoding["attention_mask"].squeeze().bool(),
            "response_ids": response_encoding["input_ids"].squeeze(),
            "response_padding_mask": response_encoding["attention_mask"].squeeze().bool(),
        }

train_dataset = DailyDialogPairs(train_pairs, tokenizer)
val_dataset = DailyDialogPairs(val_pairs, tokenizer)
test_dataset = DailyDialogPairs(test_pairs, tokenizer)