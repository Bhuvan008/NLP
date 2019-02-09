#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 18:50:30 2019
@Bhavani Bhasutkar
"""
import json
import os
import pandas as pd
import sklearn
from sklearn.metrics import matthews_corrcoef
import numpy as np
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM,Dense,Embedding
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import os
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Dense, Flatten
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')
from keras.preprocessing.text import Tokenizer
from keras import Sequential
from keras.models import load_model
from sklearn.metrics import matthews_corrcoef,classification_report

#%%
path_to_jsonfiles_pos= '/home/fractaluser/Desktop/Personal/Study/Hackathon/reviews_hackathon_data_Train_test_problem_statement/train/eng_positive'
path_to_jsonfiles_neg= '/home/fractaluser/Desktop/Personal/Study/Hackathon/reviews_hackathon_data_Train_test_problem_statement/train/eng_negative'
path_to_jsonfiles_test= '/home/fractaluser/Desktop/Personal/Study/Hackathon/reviews_hackathon_data_Train_test_problem_statement/test/'

#%%

#tran
#%%
alldicts = []
#for file in os.listdir(path_to_jsonfiles):
#    full_filename = "%s/%s" % (path_to_jsonfiles, file)
#    print (full_filename)
json_files_pos = [pos_json for pos_json in os.listdir(path_to_jsonfiles_pos) if pos_json.endswith('.json')]
json_files_neg = [pos_json for pos_json in os.listdir(path_to_jsonfiles_neg) if pos_json.endswith('.json')]

os.chdir(path_to_jsonfiles_pos)
for file in json_files_pos:
    with open(file,'r') as fi:
        dict = json.load(fi)
        alldicts.append(dict)

os.chdir(path_to_jsonfiles_neg)
for file in json_files_neg:
    with open(file,'r') as fi:
        dict = json.load(fi)
        alldicts.append(dict)
        
os.chdir(path_to_jsonfiles_neg)
for file in json_files_neg:
    with open(file,'r') as fi:
        dict = json.load(fi)
        alldicts.append(dict)


 #%%
df= pd.DataFrame(alldicts)
df['Sentiment']= 0
df.iloc[0:len(json_files_pos),-1]=1

#%% Checking sentiment using Text Blob package Not used in the final code
df['txtblob_sentiment']= df['text'].apply(lambda x: TextBlob(x).sentiment[0])
df['txtblob_sentiment']=df['txtblob_sentiment'].apply(lambda x: 0 if x<0 else 1)
#%%
#Acc_checks:
matthews_corrcoef(df['Sentiment'], df['txtblob_sentiment'])
#%%

#%%
#gensm library
#300 vecs
#1st layer- embedding-pretrained/own embeddings
#2nd layer- LSTM
#
#fasttext- for rare words
#word2vec- freeze embeddings

#%%
#Sample example
docs = ['Well done!',
		'Good work',
		'Great effort',
		'nice work',
		'Excellent!',
		'Weak',
		'Poor effort!',
		'not good',
		'poor work',
		'Could have done better.']
# define class labels
labels = np.array([1,1,1,1,1,0,0,0,0,0])
vocab_size = 40
encoded_docs = [one_hot(d, vocab_size) for d in docs]
print(encoded_docs)

max_length = 4
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)

#Initial LSTM model code with Embedding Layer
model = Sequential()
model.add(Embedding(vocab_size, 8, input_length=max_length))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
print(model.summary())
#%%
REPLACE_BY_SPACE_RE = re.compile('[\/(){}\[\]\|@,;"#]')
BAD_SYMBOLS_RE = re.compile('[0-9#+_.:?!]')
LINKS = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)')

def text_prepare(text):
    text = re.sub(r'https:\/\/t\.co\/\w+', "", text)
    text = re.sub(LINKS, "", text)
    text = re.sub('<[^<]*?/?>', ' ', text)
    text = re.sub(r'["\'\"!|@$—%^&*(){};:,./<>?\|=_+-]', r' ', text)
    text = re.sub(REPLACE_BY_SPACE_RE, " ", text) 
    text = re.sub(BAD_SYMBOLS_RE, "", text) 
    text = re.sub(r'\\n', "", text)
    text = re.sub(r'\n', "", text)
    text = re.sub(r'gt', "", text) 
    text = re.sub(r'amp', "", text)
    text = re.sub(' +', ' ',text)
    text = text.lower() 
    return text
#%%

df = df[['text','Sentiment']].dropna()
#df.index = np.arange(0,df.shape[0])
#df.columns = ['text','Sentiment']
#df['category'] = df['category'].apply(lambda x : x.lower())
df['text_prepared'] =  [text_prepare(df['text'][i]) for i in range(len(df['text']))]
#%%
df= pd.read_csv('/home/fractaluser/Desktop/Personal/Study/Hackathon/final_dataframe.csv', index_col=0)
df['text_prepared']=df['text_prepared'].astype('str')
#stop = stopwords.words('english')

#df['text_stp'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
#%%
#from textblob import Word
#
#!python -m textblob.downlo1779ad_corpora
#df['text_final'] = df['text_stp'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
#df[['text_stp','text_final']].head(25)
#%%
df['wc']= df['text_prepared'].apply(lambda x: str(x).split(" "))
df['wordcount']= df['text_prepared'].apply(lambda x:len( str(x).split(" ")))

ls= df['wc'].values
flat_list = [item for sublist in ls for item in sublist]
ls1=set(flat_list)
len(ls1)
s = set()
for filename in flat_list:
    s.add(filename)
#%%
maxlen=250  #Cut off mail after 100 words
max_words=30000
embedding_dim = 50


# In[11]:


tokenize=Tokenizer(num_words=max_words)
tokenize.fit_on_texts(df.text_prepared)


# In[13]:


sequences = tokenize.texts_to_sequences(df.text_prepared)
data = pad_sequences(sequences, maxlen=maxlen)

#x_train, x_test, y_train, y_test = train_test_split(data,df['Sentiment'] ,test_size=0.25, random_state=42)
x_train, y_train= data, df['Sentiment']
y_train= y_train.values
y_train=y_train.reshape(y_train.shape[0],1)
train= np.column_stack((x_train,y_train))
np.random.shuffle(train)
x_train_df= train[:,0:-1]
y_train_df= train[:,-1]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
# In[15]:
glove_dir="/home/fractaluser/Desktop/Personal/Study/Hackathon/"
#/home/fractaluser/Desktop/Personal/Study
embedding_index={}
f=open(os.path.join(glove_dir,'glove.6B.50d.txt'))
for line in f:                                                                                                                                                                                              
    values=line.split()
    word=values[0]
    coefs=np.asarray(values[1:], dtype='float32')
    embedding_index[word]=coefs
f.close()
#%%


word_index = tokenize.word_index


# In[41]:


embedding_matrix=np.zeros((max_words,embedding_dim))
for word,i in word_index.items():
    if i < max_words:
        embedding_vector=embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i]=embedding_vector
#%%
model=Sequential()
model.add(Embedding(max_words,embedding_dim,input_length=maxlen))
model.add(LSTM(32))
model.add(Dense(1,activation='sigmoid'))

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False
#%%

model.summary()

#%%
def matthews_correlation(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())

# In[108]:
filepath= '/home/fractaluser/Desktop/Personal/Study/Hackathon/model_retrain'
checkpoint = ModelCheckpoint(filepath+'.h5', monitor='val_matthews_correlation', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=[matthews_correlation])
history = model.fit(x_train_df, y_train_df,epochs=50,batch_size=255,validation_split=0.2,shuffle= True,callbacks=callbacks_list)
#%%
y_pred= model.predict(x_test, batch_size=1, verbose=1)
y_pred=np.round(np.clip(y_pred, 0, 1))
matthews_corrcoef(y_test, y_pred)
# In[109]:


keys=history.history.keys()
acc=history.history['acc']
loss=history.history['loss']
val_loss=history.history['val_loss']
val_acc=history.history['val_acc']
epochs = range(1, len(acc) + 1)
#%%
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.legend()


# In[111]:


plt.plot(epochs, loss, 'bo', label='Training loss')
plt.legend("Traning loss")
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.legend()
#%%
alltest = []
json_files_test = [pos_json for pos_json in os.listdir(path_to_jsonfiles_test) if pos_json.endswith('.json')]
os.chdir(path_to_jsonfiles_test)
for file in json_files_test:
    with open(file,'r') as fi:
        dict = json.load(fi)
        alltest.append(dict)
df_test= pd.DataFrame(alltest)
df_test_ = df_test[['text']].dropna()
df_test_['text_prepared'] =  [text_prepare(df_test_['text'][i]) for i in range(len(df_test_['text']))]
maxlen=250  #Cut off mail after 100 words
max_words=30000
embedding_dim = 50
sequences_test = tokenize.texts_to_sequences(df_test_.text_prepared)
data_test = pad_sequences(sequences_test, maxlen=maxlen)

model=load_model('/home/fractaluser/Desktop/Personal/Study/Hackathon/model.hdf5',custom_objects={'matthews_correlation':matthews_correlation})
test_pred=model.predict(data_test)
test_pred_=np.round(test_pred)
train_pred=model.predict(x_train_df)
train_pred_=[0 if x<=0.65 else 1 for x in train_pred]
test_pred_=[0 if x<=0.65 else 1 for x in test_pred]
#train_pred_[train_pred_==0]
pred=pd.DataFrame({'Pred':test_pred_})
matthews_corrcoef(y_train_df,train_pred_)
check=classification_report(y_train_df,train_pred_)
model.save('/home/fractaluser/Desktop/Personal/Study/Hackathon/model_retrained_lt.h5')
trainmodel.predict(x_train_df)
