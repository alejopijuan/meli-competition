#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import unicodedata
from fastai.text import * 
from fastai.callbacks import *
from sklearn.model_selection import train_test_split
from pathlib import Path, PurePosixPath
import pickle as pkl

'''from pynvml import *
#learn.destroy()
torch.cuda.empty_cache()

nvmlInit()

handle = nvmlDeviceGetHandleByIndex(0)
info = nvmlDeviceGetMemoryInfo(handle)

print("Total memory:", info.total/1000000000)
print("Free memory:", info.free/1000000000)
print("Used memory:", info.used/1000000000)'''

PATH = Path('.')

## Training data

data_lm = TextLMDataBunch.from_csv(
    PATH, 'train10%.csv', 
    text_cols='title', 
    label_cols='category', 
    valid_pct=0.05, 
    max_vocab=100000, bs=256
    )


#data_lm.save('data_lm_exportmin.pkl')


data_lm.save('data_lm10%.pkl')

data_lm = load_data(PATH, 'data_lm10%.pkl', bs=64)

data_lm.vocab

len(data_lm.vocab.itos)

# LM Train

'''learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.05, pretrained=True)

print('Finding learning rate...')
learn.lr_find()
learn.recorder.plot(skip_end=20, skip_start=40, suggestion=True)

#If finetuning an existing model, we must first train the head
learn.freeze()
learn.fit_one_cycle(1, 2e-2, moms=(0.8,0.7))

#If finetuning an existing model, we must then unfreeze the model and train it completely
learn.unfreeze()

learn.fit_one_cycle(
    2, 3e-3, callbacks=[SaveModelCallback(learn, every='epoch', monitor='accuracy', name='lm')]
)'''

# Classification

data_clas = TextClasDataBunch.from_csv(
        PATH, 'train10%.csv', 
        bs=1024, 
        vocab = data_lm.vocab,
        text_cols='title', 
        label_cols='category'
        )

data_clas.save('data_clas10%.pkl')

data_clas = load_data(PATH,'data_clas10%.pkl', bs=1792)

data_clas.batch_size

len(data_clas.vocab.itos)

#data_clas.vocab = data_lm.vocab

#learn.destroy()

data_clas.show_batch(rows=10)

## Classifier Training

learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.05, bptt=12, path=PATH)

learn.load_encoder('lm_enc')
learn.freeze()

learn.lr_find()
learn.recorder.plot(skip_end=10, skip_start=15, suggestion=True)

learn.fit_one_cycle(1, 1e-1, moms=(0.8, 0.7))

learn.save('clas-head-10%')

learn = learn.load('clas-head-10%')

learn.data.batch_size

learn.lr_find(start_lr=1e-6, end_lr=2e-1)
learn.recorder.plot(suggestion=True)

learn.freeze_to(-2)

learn.fit_one_cycle(2, slice(7e-3/(2.6**4), 7e-3), moms=(0.8, 0.7))

learn.save('clas-head2-10%')

learn = learn.load('clas-head2-10%')

learn.unfreeze()

learn.data.batch_size

learn.lr_find(start_lr=1e-6, end_lr=2e-1)
learn.recorder.plot(suggestion=True)

 learn.fit_one_cycle(
    2, slice(7e-4/(2.6**4), 7e-4), moms=(0.8, 0.7)
    , callbacks=[SaveModelCallback(learn, every='epoch', monitor='accuracy', name='class_fit_full2')]
)

learn.save('clas-full2')