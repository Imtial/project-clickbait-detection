import os
import sys
import signal
import argparse

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

import pytorch_lightning as pl

TOKENIZER_PATH = 'roberta-base'
MODEL_PATH = 'roberta-base'
PRETRAINED_MODEL_PATH = 'clickbait-detection/kaggle/version_0/checkpoints/epoch=4-step=53029.ckpt'

class ClickbaitDetector(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    self.model = AutoModel.from_pretrained(MODEL_PATH)

    self.model_drop = nn.Dropout(0.25)
    self.out = nn.Linear(768, 1)

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

    return outputs

  def predict(self, headline):
    headline = self.tokenizer.encode(headline, truncation=True, max_length=512, return_tensors='pt')
    preds = self.forward(headline)
    return torch.sigmoid(preds) >= 0.5

def predict(model, headline):
  preds = model.predict(headline)
  return 'Clickbait' if preds else 'News'

def main():
  model = ClickbaitDetector.load_from_checkpoint(checkpoint_path=PRETRAINED_MODEL_PATH)
  print(model)
  model.freeze()
  model.eval()

  while True:
    try:
      headline = input('Enter a headline (press Ctrl+c to terminate): ')
      print(predict(model, headline))
      signal.signal(signal.SIGINT, signal.default_int_handler)
    except KeyboardInterrupt:
      print('\nBye!!!')
      sys.exit()

if __name__ == "__main__":
    main()