import os
import argparse
import random
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from transformers.optimization import get_linear_schedule_with_warmup, Adafactor
from datasets import load_dataset, load_metric, concatenate_datasets
from datasets.dataset_dict import DatasetDict

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

class ClickbaitDataset(Dataset):
  def __init__(self, dataset, tokenizer, max_input_len):
    self.dataset = dataset
    self.tokenizer = tokenizer
    self.max_input_len = max_input_len

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    input_ids = self.tokenizer.encode(self.dataset[idx]['headline'], truncation=True, max_length=self.max_input_len, return_tensors='pt')[0]
    label = self.dataset[idx]['label']
    return input_ids, torch.tensor(label).type_as(input_ids)

  @staticmethod
  def collate_fn(batch):
    pad_token_id = 1
    input_ids, labels = list(zip(*batch))
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    return input_ids, labels

class ClickbaitDetector(pl.LightningModule):
  def __init__(self, params):
    super().__init__()
    self.args = params
    self.tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer)
    self.model = AutoModel.from_pretrained(self.args.model_path)
    self.train_dataloader_object = self.val_dataloader_object = self.test_dataloader_object = None

    self.model_drop = nn.Dropout(0.25)
    self.out = nn.Linear(768, 1)

    self.val_pred = []
    self.val_ref = []
    self.metrics = {
      'accuracy': load_metric("accuracy"),
      'precision': load_metric("precision"),
      'recall': load_metric("recall"),
      'f1': load_metric("f1"),
    }

  def _prepare_input(self, input_ids):
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
    attention_mask[input_ids == self.tokenizer.pad_token_id] = 0

    return input_ids, attention_mask

  def forward(self, input_ids):
    input_ids, attention_mask = self._prepare_input(input_ids)
    last_hidden_state = self.model(input_ids, attention_mask=attention_mask)
    last_hidden_state = last_hidden_state[0]
    mean_last_hidden_state = torch.mean(last_hidden_state, dim=1)
    model_out = self.model_drop(mean_last_hidden_state)
    outputs = self.out(model_out)

    # loss = nn.BCEWithLogitsLoss()(outputs, torch.tensor(labels, device=self.device, dtype=float).view(-1, 1))
    return outputs

  def training_step(self, batch, batch_nb):
    input_ids, labels = batch
    outputs = self.forward(input_ids)
    
    loss = nn.BCEWithLogitsLoss()(outputs, torch.tensor(labels, device=self.device, dtype=float).view(-1, 1))

    lr = loss.new_zeros(1) + self.trainer.optimizers[0].param_groups[0]['lr']
    tensorboard_logs = {'train_loss': loss, 'lr': lr,
                        'input_size': batch[0].numel(),
                        'mem': torch.cuda.memory_allocated(loss.device) / 1024 ** 3 if torch.cuda.is_available() else 0}
    return {'loss': loss, 'log': tensorboard_logs}

  def validation_step(self, batch, batch_nb):
    for p in self.model.parameters():
        p.requires_grad = False
    
    outputs, vloss = self.forward(*batch)
    
    val_pred = (torch.sigmoid(outputs.flatten()) >= 0.5) * 1
    val_ref = torch.tensor(batch[1]).type_as(vloss)

    self.log('vloss', vloss)
    
    return vloss, val_pred, val_ref

  def validation_epoch_end(self, outputs):
    for p in self.model.parameters():
        p.requires_grad = True

    vloss, val_pred, val_ref = list(zip(*outputs))

    vloss = torch.stack(vloss).mean()
    torch.distributed.all_reduce(vloss, op=torch.distributed.ReduceOp.SUM)
    vloss /= self.trainer.world_size

    val_pred = torch.hstack(val_pred)
    val_ref = torch.hstack(val_ref)

    logs = {
        'vloss': vloss,
        'accuracy': self.metrics['accuracy'].compute(predictions=val_pred, references=val_ref),
        'precision': self.metrics['precision'].compute(predictions=val_pred, references=val_ref),
        'recall': self.metrics['recall'].compute(predictions=val_pred, references=val_ref),
        'f1': self.metrics['f1'].compute(predictions=val_pred, references=val_ref),
    }

    print(logs)

    return {'avg_val_loss': logs['vloss'], 'log': logs, 'progress_bar': logs}

  def test_step(self, batch, batch_nb):
    return self.validation_step(batch, batch_nb)

  def test_epoch_end(self, outputs):
    result = self.validation_epoch_end(outputs)
    print(result)

  def configure_optimizers(self):
    if self.args.adafactor:
        optimizer = Adafactor(self.model.parameters(), lr=self.args.lr, scale_parameter=False, relative_step=False)
    else:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
    if self.args.debug:
        return optimizer  # const LR
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    num_steps = self.args.dataset_size * self.args.epochs / num_gpus / self.args.grad_accum / self.args.batch_size
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=self.args.warmup, num_training_steps=num_steps
    )
    return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

  def _get_dataloader(self, current_dataloader, split_name, is_train):
    if current_dataloader is not None:
        return current_dataloader
    dataset = ClickbaitDataset(dataset=self.dataset[split_name], tokenizer=self.tokenizer,
                                   max_input_len=self.args.max_input_len)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=is_train)
    return DataLoader(dataset, batch_size=self.args.batch_size, shuffle=(sampler is None),
                      num_workers=self.args.num_workers, sampler=sampler,
                      collate_fn=ClickbaitDataset.collate_fn)
    
  def train_dataloader(self):
      self.train_dataloader_object = self._get_dataloader(self.train_dataloader_object, 'train', is_train=True)
      return self.train_dataloader_object

  def val_dataloader(self):
      self.val_dataloader_object = self._get_dataloader(self.val_dataloader_object, 'validation', is_train=False)
      return self.val_dataloader_object

  def test_dataloader(self):
      self.test_dataloader_object = self._get_dataloader(self.test_dataloader_object, 'test', is_train=False)
      return self.test_dataloader_object

  def predict(self, headline):
    headline = self.tokenizer.encode(headline, truncation=True, max_length=args.max_input_len, return_tensors='pt')
    preds = self.forward(headline)
    return torch.sigmoid(preds) >= 0.5
   
  @staticmethod
  def add_model_specific_args(parser, root_dir):
    parser.add_argument("--save_dir", type=str, default='clickbait-detection')
    parser.add_argument("--save_prefix", type=str, default='kaggle')
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--grad_accum", type=int, default=1, help="number of gradient accumulation steps")
    parser.add_argument("--gpus", type=int, default=-1,
                        help="Number of gpus. 0 for CPU")
    parser.add_argument("--warmup", type=int, default=1000, help="Number of warmup steps")
    parser.add_argument("--lr", type=float, default=0.00003, help="Maximum learning rate")
    parser.add_argument("--val_every", type=float, default=1.0, help="Number of training steps between validations")
    parser.add_argument("--limit_val_batches", default=1.00, type=float, help='Percent of validation data used')
    parser.add_argument("--num_workers", type=int, default=0, help="Number of data loader workers")
    parser.add_argument("--seed", type=int, default=1234, help="Seed")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--disable_checkpointing", action='store_true', help="No logging or checkpointing")
    parser.add_argument("--max_input_len", type=int, default=512,
                        help="maximum num of wordpieces/summary. Used for training and testing")
    parser.add_argument("--test", action='store_true', help="Test only, no training")
    parser.add_argument("--predict", action='store_true', help="Predict output for input.")
    parser.add_argument("--server", action='store_true', help="Run server for prediction")
    parser.add_argument("--model_path", type=str, default='roberta-base',
                        help="Path to the checkpoint directory or model name")
    parser.add_argument("--tokenizer", type=str, default='roberta-base')
    parser.add_argument("--progress_bar_refresh_rate", type=int, default=None, help="0 for no progress bar.")
    parser.add_argument("--precision", type=int, default='16', help="default is 16 for fp16. Use 32 to switch to fp32")
    parser.add_argument("--debug", action='store_true', help="debug run")
    parser.add_argument("--resume_ckpt", type=str, help="Path of a checkpoint to resume from")
    parser.add_argument("--from_pretrained", type=str, default=None,
                        help="Path to a checkpoint to load model weights but not training state")
    parser.add_argument("--hparams", type=str, default=None,
                        help="Path to a saved hparams yaml file")
    parser.add_argument('--grad_ckpt', action='store_true', help='Enable gradient checkpointing to save memory')
    parser.add_argument("--attention_dropout", type=float, default=0.1, help="attention dropout")
    parser.add_argument("--label_smoothing", type=float, default=0.0, required=False)
    parser.add_argument("--adafactor", action='store_true', help="Use adafactor optimizer")
    
    return parser

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    if args.from_pretrained is not None:
        model = ClickbaitDetector.load_from_checkpoint(args.from_pretrained, args)
    else:
        model = ClickbaitDetector(args)

    train1 = load_dataset('csv', data_files={'train': 'train1.csv'})
    train2 = load_dataset('csv', data_files={'train': 'train2.csv'})
    train1 = train1.rename_column('clickbait', 'label')
    train2 = train2.rename_column('title', 'headline')
    train2 = train2.map(lambda example: {
        'label': 0 if example['label'] == 'news' else 1
    })
    cb_dataset = concatenate_datasets([train1['train'], train2['train']])

    # cb_dataset = cb_dataset.train_test_split(train_size=0.01, seed=20).pop('train')

    cb_dataset = cb_dataset.train_test_split(train_size=0.8, seed=20)
    cb_dataset_val_test = cb_dataset['test'].train_test_split(train_size=0.5, seed=20)
    cb_dataset['validation'] = cb_dataset_val_test.pop('train')
    cb_dataset['test'] = cb_dataset_val_test.pop('test')

    del train1
    del train2
    
    model.dataset = cb_dataset

    logger = TensorBoardLogger(
        save_dir=args.save_dir,
        name=args.save_prefix,
        version=0  # always use version=0
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.save_dir, args.save_prefix, "checkpoints"),
        save_top_k=2,
        verbose=True,
        monitor='avg_val_loss',
        mode='min',
        every_n_epochs=1
    )

    print(args)

    args.dataset_size = model.dataset['train'].num_rows  # hardcode dataset size. Needed to compute number of steps for the lr scheduler

    trainer = pl.Trainer(gpus=args.gpus, strategy='ddp' if torch.cuda.is_available() else None,
                         track_grad_norm=-1,
                         max_epochs=args.epochs if not args.debug else 100,
                         max_steps=None if not args.debug else 1,
                         replace_sampler_ddp=False,
                         accumulate_grad_batches=args.grad_accum,
                         val_check_interval=args.val_every if not args.debug else 1,
                         num_sanity_val_steps=2 if not args.debug else 0,
                         check_val_every_n_epoch=1 if not args.debug else 1,
                         limit_val_batches=args.limit_val_batches,
                         limit_test_batches=args.limit_val_batches,
                         logger=logger,
                         checkpoint_callback=checkpoint_callback if not args.disable_checkpointing else False,
                         progress_bar_refresh_rate=args.progress_bar_refresh_rate,
                         precision=args.precision, amp_backend='native',
                         resume_from_checkpoint=args.resume_ckpt,
                         )
    if not args.test:
        trainer.fit(model)
    trainer.test(model)


def load_model(args):
  model = ClickbaitDetector.load_from_checkpoint(checkpoint_path=args.from_pretrained, params=args)
  model.freeze()
  return model

def predict(model, headline):
  preds = model.predict(headline)
  return 'Clickbait' if preds else 'News'

def console(args):
  import sys
  import signal
  model = load_model(args)
  while True:
    try:
      headline = input('Enter a headline (press Ctrl+c to terminate): ')
      print(predict(model, headline))
      signal.signal(signal.SIGINT, signal.default_int_handler)
    except KeyboardInterrupt:
      print('\nBye!!!')
      sys.exit()


if __name__ == "__main__":
    main_arg_parser = argparse.ArgumentParser(description="clickbait-detector")
    parser = ClickbaitDetector.add_model_specific_args(main_arg_parser, os.getcwd())
    args = parser.parse_args()
    print(args)
    if args.predict:
      if args.server:
        from flask import Flask
        app = Flask(__name__)
        print(__name__)
      else:
        console(args)
    # else:
    #   main(args)