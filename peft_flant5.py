'''
!pip install -q transformers
!pip install -q pytorch_lightning
!pip install -q sentencepiece
!mkdir -p t5_swag
# !pip install -q jsonlines
!pip install "peft==0.2.0"
!pip install "transformers==4.27.1" "datasets==2.9.0" "accelerate==0.17.1" "evaluate==0.4.0" "bitsandbytes==0.37.1" loralib --upgrade --quiet
!pip install rouge-score tensorboard py7zr 
'''
import jsonlines
import argparse, random
import os
import logging
import random
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
from tqdm.auto import tqdm
from sklearn import metrics
#import nltk

#nltk.download('punkt')
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from tqdm import tqdm

@dataclass(frozen=True)
class InputExample:
    input_text: str
    output_text: Optional[str]


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class DataProcessor:

    def get_train_examples(self, data_dir):

        f = open("dataset/train.jsonl", mode='r', encoding="utf8")
        reader = jsonlines.Reader(f)
        examples = []
        for line in reader:
            examples.append(InputExample(
                input_text=line['input_text'],
                label=line['output_text'],
            ))
        return examples

    def get_dev_examples(self, data_dir):

        f = open("dataset/dev.jsonl", mode='r', encoding="utf8")
        reader = jsonlines.Reader(f)
        examples = []
        for line in reader:
            examples.append(InputExample(
                input_text=line['input_text'],
                label=line['output_text'],
            ))
        return examples

    def get_test_examples(self, data_dir):
        f = open("dataset/test.jsonl", mode='r', encoding="utf8")
        reader = jsonlines.Reader(f)
        examples = []
        for line in reader:
            examples.append(InputExample(
                input_text=line['input_text'],
            ))
        return examples

    def get_labels(self, is_negative_augmented=False):
        if is_negative_augmented:
            return ["A", "B", "C", "D", "E", "F"]
        
        return ["A", "B", "C", "D", "E"]


class MCQDataset(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, max_len=512):
        self.data_dir = data_dir
        self.type_path = type_path
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []
        self.reordering_limit=10

        self.proc = DataProcessor()

        self._build()

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()
        target_mask = self.targets[index]["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

    def __len__(self):
        return len(self.inputs)

    def _build(self):
        if self.type_path == 'train':
            examples = self.proc.get_train_examples(self.data_dir)
        elif self.type_path == 'val':
            examples = self.proc.get_dev_examples(self.data_dir)
        elif self.type_path == "test":
            examples = self.proc.get_test_examples(self.data_dir, test_mode=True)
        elif self.type_path == 'reorder_val':
            examples = self.proc.get_dev_examples(self.data_dir)

        for example in examples:
            self._create_features(example, test_mode=self.type_path == "test", is_reorder=self.type_path == 'reorder_val')

    def _create_features(self, example, test_mode=False, is_reorder=False):

        input_ = example.input_text
        target_label, tokenized_targets = None, None
        if not test_mode:
            target_label = "%s </s>" % (example.output_text)

        if is_reorder:
            options = input_.split(" ")[-5:]
            text = input_.split(" ")[:-5]
            label = ord(target_label[0]) - ord('A')
            permute_options = [(txt, i) for i, txt in enumerate(options)]
            inputs, labels = [], []
            
            for it in self.reordering_limit:
                random.shuffle(permute_options)
                modified_input = " ".join(text + [opt[0] for opt in permute_options])
                modified_label = None
                for i, opt in enumerate(permute_options):
                    if opt[1] == label:
                        modified_label = i
                        break
                modified_label = "%s </s>" % (str(modified_label))

                tokenized_inputs = self.tokenizer.batch_encode_plus([modified_input], max_length=self.max_len, truncation=True, pad_to_max_length=True, return_tensors="pt")
                tokenized_targets = self.tokenizer.batch_encode_plus([modified_label], max_length=2, truncation=True, pad_to_max_length=True, return_tensors="pt")
                inputs.append(tokenized_inputs)
                labels.append(tokenized_targets)
            self.inputs += inputs
            self.targets += labels
        else:
            tokenized_inputs = self.tokenizer.batch_encode_plus([input_], max_length=self.max_len, truncation=True, pad_to_max_length=True, return_tensors="pt")
            
            if not test_mode:
                tokenized_targets = self.tokenizer.batch_encode_plus([target_label], max_length=2, truncation=True, pad_to_max_length=True, return_tensors="pt")
            
            self.inputs.append(tokenized_inputs)
            if not test_mode:
                self.targets.append(tokenized_targets)

def get_dataset(tokenizer, type_path, args):
    return MCQDataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path, max_len=args.max_seq_length)

model_id = "google/flan-t5-xxl"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model_id = "philschmid/flan-t5-xxl-sharded-fp16"
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=True, device_map="auto")
lora_config = LoraConfig(
 r=16, 
 lora_alpha=32,
 target_modules=["q", "v"],
 lora_dropout=0.05,
 bias="none",
 task_type=TaskType.SEQ_2_SEQ_LM
)

model = prepare_model_for_int8_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

label_pad_token_id = -100

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)

output_dir = "lora-flan-t5-xxl"
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
        auto_find_batch_size=True,
    learning_rate=1e-3,
    num_train_epochs=5,
    logging_dir=f"{output_dir}/logs",
    logging_strategy="steps",
    logging_steps=500,
    save_strategy="no",
    report_to="tensorboard",
)

val_dataset = MCQDataset(tokenizer, data_dir='./dataset', type_path='val')
train_dataset = MCQDataset(tokenizer, data_dir='./dataset', type_path='train')
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

model.config.use_cache = False
trainer.train()

peft_model_id = "anlp4_results"
trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)

def evaluate_peft_model(sample,max_target_length=50):
    outputs = model.generate(input_ids=sample["input_ids"].unsqueeze(0).cuda(), do_sample=True, top_p=0.9, max_new_tokens=max_target_length)    
    prediction = tokenizer.decode(outputs[0].detach().cpu().numpy(), skip_special_tokens=True)
    labels = np.where(sample['labels'] != -100, sample['labels'], tokenizer.pad_token_id)
    labels = tokenizer.decode(labels, skip_special_tokens=True)
    return prediction, labels

import sys

if len(sys.argv) > 1 and sys.argv[1] == "reorder":
    test_dataset = MCQDataset(tokenizer, data_dir='./dataset', type_path='reorder_val')
else:
    test_dataset = MCQDataset(tokenizer, data_dir='./dataset', type_path='val')

predictions, references = [] , []
for sample in tqdm(test_dataset):
    p,l = evaluate_peft_model(sample)
    predictions.append(p)
    references.append(l)

predictions = [o[-1] for o in predictions]
references = [t[0] for t in references]
print(metrics.accuracy_score(predictions, references))
