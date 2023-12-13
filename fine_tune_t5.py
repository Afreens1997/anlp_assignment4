'''
!pip install -q sentencepiece transformers datasets evaluate
!pip install -q transformers
!pip install -q pytorch_lightning
!pip install -q sentencepiece
!mkdir -p t5_swag
# !pip install -q jsonlines
'''
import jsonlines
import argparse
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super(T5FineTuner, self).__init__()
        self.save_hyperparameters(hparams)
        self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.tokenizer_name_or_path)
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def is_logger(self):
        return self.trainer.global_rank <= 0

    def forward(
            self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        tensorboard_logs = {"train_loss": loss}
        self.training_step_outputs.append(loss)
        self.log("train_loss", loss, prog_bar=True)
        return {"loss": loss, "log": tensorboard_logs}

    def on_train_epoch_end(self):
        outputs = self.training_step_outputs
        avg_train_loss = torch.stack([x for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        self.log("avg_train_loss", avg_train_loss, prog_bar=True)
        return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.validation_step_outputs.append(loss)
        self.log("val_loss", loss, prog_bar=True)
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        avg_loss = torch.stack([x for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        self.log("avg_val_loss", avg_loss, prog_bar=True)
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        super(T5FineTuner, self).optimizer_step(epoch, batch_idx, optimizer, optimizer_closure=optimizer_idx)
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}
        return tqdm_dict

    def train_dataloader(self):
        train_dataset = get_dataset(tokenizer=self.tokenizer, type_path="train", args=self.hparams)
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, drop_last=True, shuffle=True,
                                num_workers=4)
        t_total = (
                (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
                // self.hparams.gradient_accumulation_steps
                * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = get_dataset(tokenizer=self.tokenizer, type_path="val", args=self.hparams)
        return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4)


logger = logging.getLogger(__name__)


class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
            with open(output_test_results_file, "w") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        logger.info("{} = {}\n".format(key, str(metrics[key])))
                        writer.write("{} = {}\n".format(key, str(metrics[key])))


args_dict = dict(
    data_dir=".",
    output_dir=".",
    model_name_or_path='google/flan-t5-base',
    tokenizer_name_or_path='google/flan-t5-base',
    max_seq_length=256,
    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=2,
    eval_batch_size=2,
    num_train_epochs=3,
    gradient_accumulation_steps=16,
    n_gpu=1,
    early_stop_callback=False,
    fp_16=False,
    opt_level='O1',
    max_grad_norm=1.0,
    seed=42,
)


@dataclass(frozen=True)
class InputExample:
    example_id: str
    context: str
    question: str
    endings: List[str]
    label: Optional[int]


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class DataProcessor:
    def get_train_examples(self, data_dir):
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        raise NotImplementedError()

    def get_labels(self):
        raise NotImplementedError()


class SwagProcessor(DataProcessor):

    def get_train_examples(self, data_dir):

        f = open("dataset/train.jsonl", mode='r', encoding="utf8")
        reader = jsonlines.Reader(f)
        examples = []
        id = 0
        for line in reader:
            id += 1
            choice_0 = line['option_0']
            choice_1 = line['option_1']
            choice_2 = line['option_2']
            choice_3 = line['option_3']
            choice_4 = line['option_4']
            examples.append(InputExample(
                example_id=str(id),
                context=line['article'],
                question=line['question'],
                endings=[choice_0, choice_1, choice_2, choice_3, choice_4],
                label=int(line['label']),
            ))
        return examples

    def get_dev_examples(self, data_dir):
        logger.info("LOOKING AT {} dev".format(data_dir))
        f = open("dataset/dev.jsonl", mode='r', encoding="utf8")
        reader = jsonlines.Reader(f)
        examples = []
        id = 0
        for line in reader:
            id += 1
            choice_0 = line['option_0']
            choice_1 = line['option_1']
            choice_2 = line['option_2']
            choice_3 = line['option_3']
            choice_4 = line['option_4']
            examples.append(InputExample(
                example_id=str(id),
                context=line['article'],
                question=line['question'],
                endings=[choice_0, choice_1, choice_2, choice_3, choice_4],
                label=int(line['label']),
            ))
        return examples

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} dev".format(data_dir))
        raise ValueError(
            "For swag testing, the input file does not contain a label column. It can not be tested in current code"
            "setting!"
        )

    def get_labels(self):
        return ["0", "1", "2", "3", "4"]


class SwagDataset(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, max_len=512):
        self.data_dir = data_dir
        self.type_path = type_path
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        self.proc = SwagProcessor()

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
        else:
            examples = self.proc.get_dev_examples(self.data_dir)

        for example in examples:
            self._create_features(example)

    def _create_features(self, example):
        sample_article = 'Notices of intended prosecution claimed 991 drivers had broken a 40mph (64km/h) speed limit in Conwy tunnel last October. But the limit had been imposed for night maintenance only and not lifted in the morning as it should have been. Within days the drivers got an apology in the post. North Wales Police released the figures in a Freedom of Information reply. The force said: \"The issue was caused by an administration error surrounding the enforcement period. \"North Wales police do not record the cost of cancelling notices.\"'
        sample_summary = "Nearly 1,000 drivers were wrongly sent speeding notices after a @placeholder limit on a north Wales road was not lifted , figures have shown ."
        sample_placeholder = "temporary"
        example1 = f'Article: {sample_article} \nSummary: {sample_summary} \n@placeholder replacement: {sample_placeholder}'

        sample_article = '12 June 2017 Last updated at 12:52 BST Previously code-named Project Scorpio - take a look at the new Xbox One X console. Phil Spencer, head of Xbox said it was the: \"most powerful console ever made\". The console was revealed at this year\'s E3 conference - one of the world\'s biggest gaming and technology shows. It runs from 13th to 15th June in Los Angeles, America.'
        sample_summary = "Microsoft have revealed their brand - new top - @placeholder console at a big game show in America ."
        sample_placeholder = "secret"
        example2 = f'Article: {sample_article} \nSummary: {sample_summary} \n@placeholder replacement: {sample_placeholder}'

        # prompt_basic = f"The task is to fill in the @placeholder in the summary. Based on the article provided, fill in the @placeholder in the summary one of the best suited option word."
        prompt_with_task_description = f"The task is to fill in the @placeholder in the summary. Based on the article provided, fill in the @placeholder in the summary one of the best suited option word. Each option is an abstract word. Abstract words refer to intangible qualities, ideas, and concepts."
        # prompt_1_shot = f"The task is to fill in the @placeholder in the summary. Based on the article provided, fill in the @placeholder in the summary one of the best suited option word. Each option is an abstract word. Abstract words refer to intangible qualities, ideas, and concepts. For example: \n{example1}"
        # prompt_2_shot = f"The task is to fill in the @placeholder in the summary. Based on the article provided, fill in the @placeholder in the summary one of the best suited option word. Each option is an abstract word. Abstract words refer to intangible qualities, ideas, and concepts. For example: \n{example1} \n{example2}\n"

        input_ = example.context
        question = example.question
        options = ['%s: %s' % (i, option) for i, option in zip('12345', example.endings)]
        options = " ".join(options)
        input_ = "%s article: %s question: %s options: %s </s>" % (prompt_with_task_description, input_, question, options)
        target_label = "%s </s>" % str(int(example.label) + 1)
        tokenized_inputs = self.tokenizer.batch_encode_plus([input_], max_length=self.max_len, truncation=True, pad_to_max_length=True,
                                                            return_tensors="pt")
        tokenized_targets = self.tokenizer.batch_encode_plus([target_label], max_length=2, truncation=True, pad_to_max_length=True,
                                                             return_tensors="pt")
        self.inputs.append(tokenized_inputs)
        self.targets.append(tokenized_targets)


tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-base')

args_dict.update({'data_dir': 'swag_data', 'output_dir': 't5_swag', 'num_train_epochs': 3})
args = argparse.Namespace(**args_dict)
print(args_dict)

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath=args.output_dir, monitor="val_loss", mode="min", save_top_k=5
)

train_params = dict(
    accumulate_grad_batches=args.gradient_accumulation_steps,
    accelerator="auto",
    max_epochs=3,
    # early_stop_callback=False,
    precision=32,
    # opt_level=args.opt_level,
    gradient_clip_val=args.max_grad_norm,
    callbacks=[LoggingCallback(), checkpoint_callback],
)


def get_dataset(tokenizer, type_path, args):
    return SwagDataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path, max_len=args.max_seq_length)


model = T5FineTuner(args)
trainer = pl.Trainer(**train_params)
trainer.fit(model)

dataset = SwagDataset(tokenizer, data_dir='swag_data', type_path='val')
loader = DataLoader(dataset, batch_size=2, num_workers=4)

model.model.eval()
outputs = []
targets = []
for batch in tqdm(loader):
    outs = model.model.to("cuda").generate(input_ids=batch['source_ids'].to("cuda"),
                                           attention_mask=batch['source_mask'].to("cuda"),
                                           max_length=2)

    dec = [tokenizer.decode(ids) for ids in outs]
    target = [tokenizer.decode(ids) for ids in batch["target_ids"]]

    outputs.extend(dec)
    targets.extend(target)

#for i, out in enumerate(outputs):
 #   if out not in "12345":
  #      print(i, 'detected invalid prediction')

outputs_1 = [o[-1] for o in outputs]
targets_1 = [t[0] for t in targets]
print(metrics.accuracy_score(targets_1, outputs_1))
