import inspect
from pandas.io import clipboard
import numpy as np


def func(name):
    clipboard.copy(dict_[name])

llm = None

def red_button(text, or_token):
    try:
        from langchain.schema import SystemMessage, HumanMessage, AIMessage
        from langchain.prompts import ChatPromptTemplate, PromptTemplate
        # from langchain.schema.runnable import RunnableLambda, RunnableSequence, RunnablePassthrough, RunnableParallel, \
        #     RunnableBranch
        # from langchain.schema.output_parser import StrOutputParser
        # from langchain.output_parsers.structured import StructuredOutputParser, ResponseSchema

        from langchain_openai import ChatOpenAI

        global llm

        if not llm:
            llm = ChatOpenAI(
                base_url="https://openrouter.ai/api/v1/",
                model="deepseek/deepseek-chat-v3-0324:free",
                api_key=or_token
            )

        system_prompt = "Ты — эксперт и хороший программист " \
                        "В ответе на запросы просто пиши код без комментариев и дополнительных пояснений (если не сказано обратного)"

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=text)]

        response = llm.invoke(messages).content
        clipboard.copy(response)
    except Exception as e:
        clipboard.copy(f'!failed: {e}')
        llm = None


dict_ = {
'test_func': '''def test_func():
    print('success!')''',

'imports': '''import os
import re
from copy import deepcopy
from functools import reduce
from collections import defaultdict
from random import choice
from random import randint
from time import time, strftime, gmtime

import numpy as np
import pandas as pd
from scipy import stats

import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F

# %pip install torcheval
# from torcheval.metrics import MulticlassAccuracy, BinaryAccuracy, MulticlassF1Score, BinaryF1Score

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from tokenizers import Tokenizer, Regex
from tokenizers.models import WordLevel, WordPiece
from tokenizers.trainers import WordLevelTrainer, WordPieceTrainer, UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace, CharDelimiterSplit, Split, PreTokenizer, WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from tokenizers.normalizers import Lowercase, Replace, Sequence
from tokenizers.decoders import ByteLevel
from tokenizers import InputSequence, PreTokenizedInputSequence
from urllib.request import urlopen
from types import ModuleType''',

'tokenizer': '''tokenizer = Tokenizer(WordLevel(unk_token="<UNK>"))
trainer = WordLevelTrainer(vocab_size=100, special_tokens=["<PAD>", "<SOS>", "<EOS>", "<UNK>"])
normalizer = Sequence([Lowercase()])
tokenizer.normalizer = normalizer
tokenizer.pre_tokenizer = Split(pattern='', behavior='removed') # Split(' ', behavior='removed') # WhitespaceSplit()
tokenizer.decoder = ByteLevel()

tokenizer.enable_padding(pad_token="<PAD>", length=10)
tokenizer.enable_truncation(max_length=10)

tokenizer.post_processor = TemplateProcessing(
            single="<SOS> $A <EOS>",
            special_tokens=[("<SOS>", 2), ("<EOS>", 3)],
        )

tokenizer.train_from_iterator(dataset.name.to_list(), trainer=trainer)''',

'translation_enc_dec': '''
class Encoder(nn.Module):
    def __init__(self, n_words, emb_dim, hidden_dim):
        super().__init__()
        self.emb = nn.Embedding(n_words, emb_dim, padding_idx=0)
        self.rnn = nn.GRU(emb_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        x = self.emb(x)
        out, h = self.rnn(x)
        return h
class Decoder(nn.Module):
    def __init__(self, n_words, emb_dim, hidden_size, seq_len, sos_idx=2, device='cpu'):
        super().__init__()

        self.emb = nn.Embedding(n_words, emb_dim, padding_idx=0)
        self.gru = nn.GRUCell(emb_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, n_words)
        self.seq_len = seq_len
        self.sos_idx = sos_idx
        self.device = device

    def forward(self, encoding, labels: torch.Tensor | None = None):
        batch_size = encoding.size(0)
        decoder_input = torch.full(size=(batch_size,), fill_value=self.sos_idx).long().to(self.device)
        decoder_h = encoding
        outputs = []
        for i in range(self.seq_len):
            emb = self.emb(decoder_input)
            decoder_h = self.gru(emb, decoder_h)
            token_predictions = self.fc(decoder_h)
            if (labels is not None) and (i < labels.size(1)):

                decoder_input = labels[:, i]
            else:
                decoder_input = token_predictions.argmax(dim=-1).detach()

            outputs.append(token_predictions)

        outputs = torch.stack(outputs, dim=1)
        return outputs

class EncoderDecoder(nn.Module):
    def __init__(self, n_ru_words, n_en_words, emb_dim, hidden_size, seq_len_ru, seq_len_en, sos_idx=2, device='cpu'):
        super().__init__()
        self.encoder = Encoder(n_ru_words, emb_dim, hidden_size)
        self.n_en_words = n_en_words
        self.decoder = Decoder(self.n_en_words, emb_dim, hidden_size, seq_len_en, sos_idx, device=device)

    def forward(self, x, labels=None):
        bs = x.size(0)
        x = x.reshape(bs, -1)
        x = self.encoder(x).reshape(bs, -1)
        # print('---')
        if labels is not None:
            labels = labels.reshape(bs, -1)
        x = self.decoder(x, labels)
        return x.reshape(-1, self.n_en_words)
    

def pad_after_eos(tokenized_sent, pad_idx=0, eos_idx=3):
    try:
        ind = torch.where(tokenized_sent == eos_idx)[0][0]
        tokenized_sent[ind+1:] = pad_idx
    except IndexError:
        pass
    return tokenized_sent

def translate(ru_sent, model, ru_tokenizer, en_tokenizer, pre_tokenized=False, device='cpu'):
    if pre_tokenized:
        pred = model(ru_sent.to(device)).argmax(1).to('cpu')
    else:
        pred = model(torch.LongTensor([ru_tokenizer.encode(ru_sent).ids]).to(device)).argmax(1).to('cpu')
    pred = pad_after_eos(pred) 
    return en_tokenizer.decode(list(pred))
    
def meaningful_part(tokens, en_tokenizer):
    tokens = pad_after_eos(tokens)
    # print(tokens)
    return [en_tokenizer.decode([token]) for token in tokens if token not in (0, 1, 2, 3)]
''',

'learning': '''n_epoch = 10
lr = 0.001

model = Model()

loss_func = nn.CrossEntropyLoss(ignore_index=0)

opt = optim.Adam(model.parameters(), lr=lr)

epoch_time_log = []
train_loss_log = []


try:
    for epoch in range(n_epoch):
        time_start = time()

        model.train()
        tot_train_loss = []
        for batch, (x_train, y_train) in enumerate(train_dl):

            y_pred = model(x_train)
            loss = loss_func(y_pred, y_train)
            opt.zero_grad()
            loss.backward()
            opt.step()

            tot_train_loss.append(loss.item())

        train_loss_log.append(np.array(tot_train_loss).mean())
        model.eval()
        with torch.no_grad():
            pass

        epoch_time_log.append(time() - time_start)
        eta = np.array(epoch_time_log).mean() * (n_epoch - epoch - 1)

        eta = strftime('%H:%M:%S', gmtime(eta))

        print(f'epoch: {epoch:<3} | loss: {train_loss_log[-1]:.{5}f} | eta: {eta}')
except KeyboardInterrupt:
    print('Interrupted')
''',
'sym_gen': '''class PetsDataset(Dataset):
    def __init__(self, data, tokenizer, transform_label=lambda x: x, seq_len=None):
        self.data = data.name
        self.lang = data.lang
        self.transform_label = transform_label
        self.tokenizer = tokenizer
        self.seq_len = seq_len


        self.tokenizer.post_processor = TemplateProcessing(
            single="<SOS> $A <EOS>",
            special_tokens=[("<SOS>", 1), ("<EOS>", 2)],
        )

        if self.seq_len:
            self.tokenizer.enable_padding(pad_token="<PAD>", length=self.seq_len)
            self.tokenizer.enable_truncation(max_length=self.seq_len)


    def __getitem__(self, idx):
        lang_id = tokenizer.get_vocab()[self.lang.iloc[idx]]
        ids = [lang_id] + self.tokenizer.encode(self.data.iloc[idx]).ids
        data = torch.unsqueeze(torch.IntTensor(ids), -1)
        x = data[:-1]
        y = data[1:]
        return x.flatten(), y.flatten()

    def __len__(self):
        return len(self.data)
        
def gen_pet_name(model, tokenizer, lang_token, max_len=11):
    voc = tokenizer.get_vocab()
    sos_id = voc['<SOS>']
    eos_id = voc['<EOS>']
    land_id = voc[lang_token]
    tokens = [land_id, sos_id]
    for i in range(max_len):
        o, h = model(torch.IntTensor(tokens).reshape((1, -1)))
        o = F.softmax(o, dim = 1)
        new_token = torch.multinomial(o[-1], num_samples=1)
        tokens.append(int(new_token[0]))
        if tokens[-1] == eos_id:
            break
    return tokenizer.decode(tokens).capitalize()''',

'sym_gen_cycle': '''n_epoch = 500
lr = 0.001


model = PetNamesModel(emb_dim=40, hidden_size=100, vocab_size=tokenizer.get_vocab_size())
model.to(device)

loss_func = nn.CrossEntropyLoss(ignore_index=0)

opt = optim.Adam(model.parameters(), lr=lr)

epoch_time_log = []
train_loss_log = []

print_every = 30

try:
    for epoch in range(n_epoch):
        time_start = time()

        model.train()
        tot_train_loss = []
        for batch, (x_train, y_train) in enumerate(train_dl):
            x_train = x_train.to(device)
            y_train = y_train.to(device)

            out, h = model(x_train)
            loss = loss_func(out, y_train.flatten().reshape(-1).long())
            opt.zero_grad()
            loss.backward()
            opt.step()

            tot_train_loss.append(loss.item())

        train_loss_log.append(np.array(tot_train_loss).mean())
        model.eval()
        with torch.no_grad():
            if epoch % print_every == 0 or (epoch == n_epoch - 1):
                generated_names_en = []
                generated_names_ru = []
                for _ in range(5):
                     generated_names_en.append(gen_pet_name(model, tokenizer, '<EN>'))
                     generated_names_ru.append(gen_pet_name(model, tokenizer, '<RU>'))
                print(f'Generated (EN): {", ".join(generated_names_en)}')
                print(f'Generated (RU): {", ".join(generated_names_ru)}')

        epoch_time_log.append(time() - time_start)
        eta = np.array(epoch_time_log).mean() * (n_epoch - epoch - 1)

        eta = strftime('%H:%M:%S', gmtime(eta))

        if epoch % print_every == 0 or (epoch == n_epoch - 1):
            epoch_time = strftime('%H:%M:%S', gmtime(sum(epoch_time_log[-print_every:])))
            print(f'epoch: {epoch:<3} | train_loss: {train_loss_log[-1]:.{5}f} | eta: {eta} ({epoch_time})')

except KeyboardInterrupt:
    print('Interrupted')''',

'text_clsf': '''class NewsDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.corpus = data.text
        self.labels = data.topic.astype('category').cat.codes
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        return torch.tensor(self.tokenizer.encode(self.corpus.iloc[idx]).ids), torch.tensor(int(self.labels.iloc[idx]))

    def __len__(self):
        return len(self.labels)
def forward(self, x):
    x = self.emb(x)
    x = x.mean(dim=1)
    x = self.fc(x)
    return x
    
''',
'stacked_gru': '''class StackedGRU2Layers(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.0, batch_first=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.batch_first = batch_first
        
        self.gru_cell1 = nn.GRUCell(input_size, hidden_size)
        self.gru_cell2 = nn.GRUCell(hidden_size, hidden_size)
        
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x, hx=None):
        if not self.batch_first:
            x = x.transpose(0, 1)
        
        batch_size, seq_len, _ = x.size()
        
        if hx is None:
            hx = torch.zeros(2, batch_size, self.hidden_size, device=x.device)
        elif hx.dim() == 2:
            hx = hx.unsqueeze(0).expand(2, -1, -1)
        
        h1, h2 = hx[0], hx[1]
        
        out = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            h1 = self.gru_cell1(x_t, h1)
            
            h1_drop = self.dropout_layer(h1) if self.dropout > 0 else h1
            
            h2 = self.gru_cell2(h1_drop, h2)
            
            out.append(h2)
        
        out = torch.stack(out, dim=1)
        hx = torch.stack([h1, h2])
        
        if not self.batch_first:
            out = out.transpose(0, 1)  # (seq_len, batch, hidden_size)
        
        return out, hx''',
'rnncell': '''class MyRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.ih = nn.Linear(input_size, hidden_size, bias=False)
        self.hh = nn.Linear(hidden_size, hidden_size)

    #     # Инициализация весов
    #     self._init_weights()

    # def _init_weights(self):
    #     nn.init.normal_(self.ih.weight, mean=0, std=0.01)
    #     nn.init.normal_(self.hh.weight, mean=0, std=0.01)
    #     nn.init.zeros_(self.hh.bias)

    def forward(self, x, h_prev):
        h_next = torch.tanh(
            self.ih(x) +  
            self.hh(h_prev)
        )
        return h_next
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=False):
    super(RNN, self).__init__()
    
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.seq_dim = int(batch_first)
    self.batch_dim = int(not batch_first)
    self.rnn = MyRNNCell(self.input_size, self.hidden_size)
    
    def forward(self, x, h=None):
        batch_size = x.size(self.batch_dim)
    
        if h is None:
            h = torch.zeros((batch_size, self.hidden_size))
        
        output = torch.zeros((*x.shape[:2], self.hidden_size))
        idx = [slice(None)] * output.ndim
        
        for i, s_t in enumerate(torch.unbind(x, dim=self.seq_dim)):
            h = self.rnn(s_t, h)
            idx[self.seq_dim] = i
            output[tuple(idx)] = h
        
        return output, torch.unsqueeze(h, 0)
''',

'sents': '''class ModelA(nn.Module):
    def __init__(self, voc_size, emb_dim):
        super().__init__()
        self.emb = nn.Embedding(voc_size, emb_dim)
        self.A = nn.Parameter(torch.randn((emb_dim, emb_dim)))
        self.fc = nn.Sequential(

        )

    def forward(self, sents1, sents2):
        e1 = self.emb(sents1).mean(1)
        e2 = self.emb(sents2).mean(1)
        return ((e1 @ self.A) * e2).sum(1)
        



with open('sents/sents_pairs_itos.json') as f:
    itos = json.load(f)
voc_size = len(itos)
voc_size

model = ModelA(voc_size, 100)

first_sents = sents_pairs[:, 0, :]
second_sents = sents_pairs[:, 1, :]

res = model(first_sents, second_sents)
print(res.shape)''',

'my_embeddings': '''import torch as th
import torch
import torch.nn as nn

class MyEmbedding(nn.Module):
    def __init__(self, 
                 num_embeddings,
                 embedding_dim, 
                 max_norm=1, 
                 norm_type=2, 
                padding_idx=0):
        super().__init__()
        
        self.emb = nn.Linear(num_embeddings, 
                             embedding_dim, 
                             bias=False)
        self.emb.weight.data.T[padding_idx] = 0
        self.max_norm = max_norm
        self.norm_type = norm_type
        
    
    def forward(self, X):
        # X -> [batch_size, tokens]
        # emb -> [num_emb, emb_dim]
        
        current_norm = torch.norm(self.emb.weight.data.T, 
                                  dim=1, 
                                  p=self.norm_type)
        for indx, one_norm in enumerate(current_norm):
            if one_norm > self.max_norm:
                self.emb.weight.data.T[indx] /= one_norm
                self.emb.weight.data.T[indx] *= self.max_norm

        X_emb = self.emb.weight.data.T[X]
        
        return X_emb

model = MyEmbedding(5, 3)
model.eval()

X = torch.randint(0, 5, (1, 5))
X

model.emb.weight.data.T

model(X)
''',
'my_embedding2': '''class MyEmbedding(nn.Module):
    def __init__(self, 
                 num_embeddings,
                 embedding_dim, 
                 max_norm=1, 
                 norm_type=2, 
                padding_idx=0):
        super().__init__()
        
#         self.emb = nn.Linear(num_embeddings, 
#                              embedding_dim, 
#                              bias=False)
#         self.emb.weight.data.T[padding_idx] = 0
        self.emb = torch.nn.Parameter(
            torch.randn(size=(num_embeddings, embedding_dim))
        )
        self.max_norm = max_norm
        self.norm_type = norm_type
        
    
    def forward(self, X):
        # X -> [batch_size, tokens]
        # emb -> [num_emb, emb_dim]
        
        weights = self.emb.data
        current_norm = torch.norm(weights, 
                                  dim=1, 
                                  p=self.norm_type)
        for indx, one_norm in enumerate(current_norm):
            if one_norm > self.max_norm:
                weights[indx] /= one_norm
                weights[indx] *= self.max_norm

        X_emb = weights[X]
        
        return X_emb
''',
'pos_tag': '''class POSTaggingDataset(Dataset):
    def __init__(self, X, y, X_tokenizer, y_tokenizer, seq_len=None):
        super().__init__()
        self.X = X
        self.y = y
        self.X_tokenizer = X_tokenizer
        self.y_tokenizer = y_tokenizer

        self.seq_len = seq_len
        if self.seq_len:
            self.X_tokenizer.enable_padding(pad_token="[PAD]", length=self.seq_len)
            self.y_tokenizer.enable_padding(pad_token="[PAD]", length=self.seq_len)

    def __getitem__(self, idx):
        x = self.X_tokenizer.encode(self.X[idx]).ids
        y = self.y_tokenizer.encode(self.y[idx]).ids
        return torch.LongTensor(x), torch.LongTensor(y)

    def __len__(self):
        return len(self.X)
        
class PetNamesModel(nn.Module):
    def __init__(self, emb_dim, hidden_size, seq_len, words_voc_size, pos_voc_size):
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.words_voc_size = words_voc_size
        self.pos_voc_size = pos_voc_size

        self.emb = nn.Embedding(self.words_voc_size, self.emb_dim)
        self.rnn = nn.RNN(self.emb_dim, self.hidden_size, batch_first=True)
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=self.pos_voc_size)

    def forward(self, x):
        x = self.emb(x)
        out, h = self.rnn(x)
        bs, sl, hs = out.shape
        x = out.reshape((bs*sl, hs))
        x = self.linear(x)
        return x.squeeze()
''',

'pos_tag_train': '''
n_epoch = 500
lr = 0.001


model = PetNamesModel(emb_dim=40, hidden_size=100, seq_len=max_sent_len, words_voc_size=X_tokenizer.get_vocab_size(), pos_voc_size=y_tokenizer.get_vocab_size())
model.to(device)

loss_func = nn.CrossEntropyLoss(ignore_index=0)

opt = optim.Adam(model.parameters(), lr=lr)

epoch_time_log = []
train_loss_log = []

print_every = 30

try:
    for epoch in range(n_epoch):
        time_start = time()

        model.train()
        tot_train_loss = []
        acc_train_metric = MulticlassAccuracy()
        for batch, (x_train, y_train) in enumerate(train_dl):

            x_train = x_train.to(device)
            y_train = y_train.to(device).flatten()
            y_pred = model(x_train)
            # print(y_pred.shape)
            # print(y_train.shape)
            loss = loss_func(y_pred, y_train)
            opt.zero_grad()
            loss.backward()
            opt.step()

            tot_train_loss.append(loss.item())
            acc_train_metric.update(y_pred.argmax(dim=1), y_train)

        train_loss_log.append(np.array(tot_train_loss).mean())
        train_acc_log.append(acc_train_metric.compute())
        model.eval()
        with torch.no_grad():
            tot_test_loss = []
            acc_test_metric = MulticlassAccuracy()
            for batch, (x_test, y_test) in enumerate(test_dl):
                x_test = x_test.to(device)
                y_test = y_test.to(device)
                y_pred = torch.squeeze(model(x_test))
                loss = loss_func(y_pred, y_test)
                tot_test_loss.append(loss.item())
                acc_test_metric.update(y_pred.argmax(dim=1), y_test)

            test_loss_log.append(np.array(tot_test_loss).mean())
            test_acc_log.append(acc_test_metric.compute())

        epoch_time_log.append(time() - time_start)
        eta = np.array(epoch_time_log).mean() * (n_epoch - epoch - 1)

        eta = strftime('%H:%M:%S', gmtime(eta))
        epoch_time = strftime('%H:%M:%S', gmtime(epoch_time_log[-1]))
        if epoch % 10 == 0 or epoch == n_epoch - 1:
            print(f'epoch: {epoch:<3} | train_loss: {train_loss_log[-1]:.{5}f} | train_acc: {train_acc_log[-1]:.{5}f} | test_loss: {test_loss_log[-1]:.{5}f} | test_acc: {test_acc_log[-1]:.{5}f} | eta: {eta} ({epoch_time})')

except KeyboardInterrupt:
    print('Interrupted')'''
}
