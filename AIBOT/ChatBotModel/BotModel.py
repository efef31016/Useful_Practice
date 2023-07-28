import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
import json
import numpy as np

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

def transform(txtlist, word2int, index):

  BOS = 0
  EOS = 1
  UNK = 2

  cn = [BOS]

  for word in txtlist[index]:
    cn.append(word2int.get(word, UNK))  # 若 word 不存在就用 UNK 取代
  cn.append(EOS)

  cn = np.asarray(cn)
  cn = torch.LongTensor(cn)

  return cn

PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token


# Data processing
class Batchdata:
  def __init__(self, word2int, trainData, batch_size = 1, n_iteration = 1, labelData = False, method = "no_onehot"):
    self.word2int = word2int
    self.trainData = trainData
    self.batch_size = batch_size
    self.n_iteration = n_iteration
    if not labelData:
      self.labelData = labelData
    else:
      self.labelData = labelData
    
    self.method = method

  def rd(self,):
    numbers_list = list(range(len(self.trainData)))
    rds = random.sample(numbers_list, self.batch_size)
    return rds


  def padding(self, x, padlen, train=True):

    mask = []

    for k in range(len(x)):

      mk = []
      if not train:
        for i in range(padlen):
          if i < len(x[k]):
            mk.append(True)
          else:
            mk.append(False)

      mask.append(mk)

      x[k] = np.pad(x[k],
      (0, (padlen - len(x[k]))), mode='constant', constant_values = 0)

    return x, mask

  def Getdata(self,):

    batchTrain = []

    for i in range(self.n_iteration):

      trainBatchi = []
      labelBatchi = []
      rdidx = self.rd()

      if self.method == "no_onehot":
        trainBatchi.append([transform(self.trainData, word2int, j) for j in rdidx])
      else:
        trainBatchi.append([self.trainData[j] for j in rdidx])

      if not self.labelData:
        print("evaluate")
      else:
        if self.method == "no_onehot":
          labelBatchi.append([transform(labelData, word2int, j) for j in rdidx])
        else:
          labelBatchi.append([self.labelData[j] for j in rdidx])

        # print(rdidx)
        # print(labelBatchi)

        labelBatchi = sorted(labelBatchi[0], key=lambda x: x.shape[0], reverse=True)
        labelPadlen = max(labelBatchi, key=len).shape[0]
        labelBatchi = self.padding(labelBatchi, labelPadlen, False)
        mask = np.asarray(labelBatchi[1])
        mask = torch.LongTensor(mask)
        mask = mask.bool()
        mask = torch.BoolTensor(mask)
        mask = torch.transpose(mask, 0, 1)
        labelBatchi = np.asarray(labelBatchi[0])
        labelBatchi = torch.LongTensor(labelBatchi)
        labelBatchi = torch.transpose(labelBatchi, 0, 1)


      trainBatchi = sorted(trainBatchi[0], key=lambda x: x.shape[0], reverse=True)
      length = torch.tensor([len(tensor) for tensor in trainBatchi]).view(-1)
      trainPadlen = max(trainBatchi, key=len).shape[0]
      trainBatchi = self.padding(trainBatchi, trainPadlen)
      trainBatchi = np.asarray(trainBatchi[0])
      trainBatchi = torch.LongTensor(trainBatchi)
      trainBatchi = torch.transpose(trainBatchi, 0, 1)

      try:
        batchTrain.append([trainBatchi, length, labelBatchi, mask, mask.shape[0]])
      except:
        batchTrain.append([trainBatchi, length])

    return batchTrain

def getInput(userInput, voc_dict):

  lenchar = len(next(iter(voc_dict.values())))
  trainData = []
  trainData.append([voc_dict.get(token, [0.5 for _ in range(lenchar)]) for token in userInput])
  trainData = torch.cat(trainData[0], dim=0)
  return Batchdata("pass", [trainData], method="onehot").Getdata()

def converse(idxint, voc_dict):
  word = next((key for key, value in voc_dict.items() if torch.equal(value, idxint)), " ")
  return word

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


# Model
class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size parameters are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden

# Luong attention layer
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden

def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()

# predict
class GreedySearchDecoder(nn.Module):
  def __init__(self, encoder, decoder):
      super(GreedySearchDecoder, self).__init__()
      self.encoder = encoder
      self.decoder = decoder

  def forward(self, input_seq, input_length, max_length):
      # Forward input through encoder model
      encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
      # Prepare encoder's final hidden layer to be first hidden input to the decoder
      decoder_hidden = encoder_hidden[:self.decoder.n_layers]
      # Initialize decoder input with SOS_token
      decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
      # Initialize tensors to append decoded words to
      all_tokens = torch.zeros([0], device=device, dtype=torch.long)
      all_scores = torch.zeros([0], device=device)
      # Iteratively decode one word token at a time
      for _ in range(max_length):
          # Forward pass through decoder
          decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
          # Obtain most likely word token and its softmax score
          decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
          # Record token and score
          all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
          all_scores = torch.cat((all_scores, decoder_scores), dim=0)
          # Prepare current token to be next decoder input (add a dimension)
          decoder_input = torch.unsqueeze(decoder_input, 0)
      # Return collections of word tokens and scores
      return all_tokens, all_scores

def AiBot(strinput, encoder, decoder, voc_dict):

  # Set dropout layers to ``eval`` mode
  encoder.eval()
  decoder.eval()

  # Initialize search module
  searcher = GreedySearchDecoder(encoder, decoder)

  input_var, lengths = getInput(strinput, voc_dict)[0] 

  wordnum = len(next(iter(voc_dict.values())))
  MAXLENGTH = 20 * wordnum
  result = searcher.forward(input_var, lengths, MAXLENGTH)
  word_res = [result[0][i:i+wordnum] for i in range(0, MAXLENGTH, wordnum)]
  res = ""
  for i in word_res:
    res += converse(i, voc_dict)
  
  return res