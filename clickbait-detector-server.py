import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

import pytorch_lightning as pl

from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
app = Flask(__name__)
CORS(app)

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



model = ClickbaitDetector.load_from_checkpoint(checkpoint_path=PRETRAINED_MODEL_PATH)
# print(model)
model.freeze()
model.eval()

import requests
from bs4 import BeautifulSoup

@app.route('/api/<headline>', methods=['GET'])
@cross_origin(origin='*',headers=['Content-Type'])
def predictFromHeadline(headline):
  if request.method == 'GET':
    preds = model.predict(headline)
    return jsonify({'className': 'Clickbait' if preds else 'News', 'headline': headline})

@app.route('/api/url=<path:url>', methods=['GET'])
@cross_origin(origin='*',headers=['Content-Type'])
def predictFromUrl(url):
  if request.method == 'GET':
    try:
      reqs = requests.get(url)
      soup = BeautifulSoup(reqs.text, 'html.parser')
      headline = soup.find('h1').get_text()
      print(headline)
    except:
      print('BAD_URL REQUEST', url)
      return 'Bad request', 400
    
    preds = model.predict(headline)
    return jsonify({'className': 'Clickbait' if preds else 'News', 'headline': headline})

@app.route('/api/', methods=['POST'])
@cross_origin(origin='*',headers=['Content-Type'])
def saveGuess():
  feedback = request.get_json()
  print(feedback)
  if request.method == 'POST':
    headline = feedback['headline']
    label = int(feedback['className'].lower() == 'clickbait')
    
    with open('feedback.csv', 'a') as f:
      f.write(f"{headline}, {label}\n")

    return jsonify({'className': feedback['className'].capitalize(), 'headline': headline})