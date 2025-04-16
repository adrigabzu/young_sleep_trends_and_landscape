#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@Author  :   Adrian G. Zucco
@Contact :   adrigabzu@sund.ku.dk
Decription: 
    This script generates Word2vec embeddings from synthetic longitudinal data using gensim 
    under different subsets and seeds.
'''

# %%
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import pandas as pd
import numpy as np
import scipy.sparse as sp
import multiprocessing
import time


# %%

class losscb_old(CallbackAny2Vec):
    
    def __init__(self):
        self.epoch = 0
        
    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 0:
            print('Loss after epoch {}: {}'.format(self.epoch, loss))
        else:
            print('Loss after epoch {}: {}'.format(self.epoch, loss - self.loss_previous_step))
        self.epoch += 1
        self.loss_previous_step = loss


class losscb(CallbackAny2Vec):
    
    def __init__(self):
        self.epoch = 1
        self.losses = []
        self.previous_epoch_time = time.time()
        
    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        
        now = time.time()
        epoch_seconds = now - self.previous_epoch_time 
        self.previous_epoch_time = now
        
        if self.epoch == 1:
            print(f'Epoch {self.epoch} \t loss {loss} ' +\
              f"\t time {round(epoch_seconds/60,2)} min")
        else:
            try:
                diff = float(loss) - self.loss_previous_step
                result = self.loss_previous_step - diff
                print(f'Epoch {self.epoch} \t loss {result} ' +\
                  f"\t time {round(epoch_seconds/60,2)} min")
            except:
                print(f'Epoch {self.epoch} \t loss {loss} ' +\
                  f"\t time {round(epoch_seconds/60,2)} min")

        self.epoch += 1
        self.loss_previous_step = float(loss)
        self.losses.append(float(loss))
        
        model.running_training_loss = 0.0


#%% OUTPUT FILENAMES
input_data = "../data/synthetic_data/mock_longdata.csv"
model_name = "w2v_embeddings"


#%%
print("#### Loading data...")
start_time = time.time()
data = pd.read_csv(input_data)
load_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
print("### Data loaded in {}".format(load_time))

# %% Create subsets
unique_ids = data['PNR'].unique()

# %%
np.random.seed(2023)
num_subsets = 5
np.random.shuffle(unique_ids)
id_subsets = np.array_split(unique_ids, num_subsets)
random_seeds = np.random.randint(0,10000, num_subsets)

# %%
for i in range(num_subsets):
    subset_size = len(id_subsets[i])
    data_sub = data[data['PNR'].isin(id_subsets[i])]
    seed = random_seeds[i]
    print(f' #### Subset {i} of size = {subset_size} and seed {seed}')

    print("## Generating sentences")
    start_time = time.time()
    print(f"##    Starting at {start_time}")
    # data_grouped = data.groupby(['PNR','year'])['code'].agg(list)
    
    # Each individual is one sentence
    data_grouped = data_sub.groupby(['PNR'])['code'].agg(list)
    sentences = data_grouped.values.tolist()
    
    proc_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    
    print(f"##    Sentences Ready in {proc_time}")
    print("## ", len(sentences))
    
    start_time = time.time()
    print(f"## Training model {start_time}")
    
    callbacker = losscb()
    model = Word2Vec(sentences=sentences, sorted_vocab=1, 
                     vector_size=200, window=100,
                     # Already filtered
                     min_count=1, 
                     compute_loss=True, workers=5, 
                     epochs=25,
                     sg=1, hs=1,
                     # When using hierarchical softmax, deactivate negative sampling,
                     negative=0,
                     alpha=0.05,
                     # min_alpha = 0.01,
                     seed=seed,
                     callbacks=[callbacker])
    
    outpath = "../results/{}_subset{}.model".format(model_name, str(i))
    model.save(outpath)
    
    outpathtxt = "../results/{}_subset{}_embeddings.txt".format(model_name, str(i))
    model.wv.save_word2vec_format(outpathtxt, binary=False)
    
    training_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print("# Training time =", training_time)
    print("#### Model bin saved at ", outpath)
    print("#### Model txt saved at ", outpathtxt)

