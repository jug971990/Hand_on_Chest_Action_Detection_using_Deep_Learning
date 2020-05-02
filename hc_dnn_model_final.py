#!/usr/bin/env python
# coding: utf-8

# ## Setting Random Seed

# In[1]:


from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


# ## GPU

# In[2]:


# ## GPU
import os
import tensorflow as tf
import keras
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#cpu-gpu configuration
#gpu_options = tf.GPUOptions(visible_device_list="5,6")
os.environ["CUDA_VISIBLE_DEVICES"]="1"

config = tf.ConfigProto(device_count = {'GPU':2, 'CPU':4}) #max no of GPUs = 1, CPUs =4
#config = tf.ConfigProto(gpu_options=gpu_options)

#config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
keras.backend.set_session(sess)


# ## Importing Libraries

# In[3]:


import numpy as np
import pandas as pd
from numpy import array
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import ast 
import joblib
import math
import time
current_t = time.time()
from pandas import DataFrame
from array import array
import xgboost 
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import MinMaxScaler
#from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_squared_error as mse
from sklearn.feature_selection import VarianceThreshold
import math
import sklearn
from pandas import DataFrame
import pickle
import scipy
from scipy import sparse
import pyodbc
import seaborn as sb
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import preprocessing
import os
from sklearn.metrics import roc_auc_score  
from scipy.sparse import csr_matrix
from scipy.stats import randint as sp_randint
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, make_scorer
import warnings
warnings.filterwarnings('ignore')
#from termcolor import colored
from sklearn.metrics import classification_report
from multiprocessing import Pool
from timeit import default_timer as timer
from math import sqrt
from collections import defaultdict
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import SelectPercentile, f_classif
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from scipy.stats import uniform as sp_rand
from sklearn import metrics   #Additional scklearn functions
from sklearn.model_selection import cross_validate
from sklearn.metrics.scorer import make_scorer
from sklearn.metrics import confusion_matrix
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg") #Needed to save figures
from sklearn.metrics import roc_auc_score
import sklearn.metrics
import json
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
import torch
import time
import numpy as np
import pandas as pd
import cv2 as cv
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize, CascadeClassifier
import glob
from tkinter import *
from PIL import Image, ImageTk
import os
import time, sys
from tkinter import font
import time
import random
from sys import argv
import sys


# ## Import keras models for Neural Network training

# In[4]:


from keras.preprocessing import image
from keras.models import Model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Input, Activation, add, Dense, Flatten, Dropout, Multiply, Embedding, Lambda
from keras.layers import Conv2D, MaxPooling2D,PReLU
from keras import backend as K
from keras.utils.vis_utils import plot_model
import theano
from keras.layers import Dense, Convolution2D, UpSampling2D, MaxPooling2D, ZeroPadding2D, Flatten, Dropout, Reshape
from keras.models import Sequential
from keras.utils import np_utils
from keras import models
from keras import layers
from keras import optimizers
from keras import regularizers
from keras import losses
from keras import metrics
from keras.utils import Sequence
from keras.utils.np_utils import to_categorical
from keras.layers import LSTM, Dense, Input, Masking, Flatten, Dropout, BatchNormalization
from keras.optimizers import RMSprop
from keras.models import load_model


# In[ ]:


path = sys.argv[1]


# ## Read training data files and append to a dataframe

# In[ ]:



# df = pd.DataFrame()
# path = r'/mnt/sde/jagadish/userdata/dl_project/hc_new_train_files_final_ff/' # use your path
# all_files = glob.glob(path + "/*.csv")

# li = []

# for filename in all_files:
#     dd = pd.read_csv(filename, index_col=None, header=0)
#     li.append(dd)

# df = pd.concat(li, axis=0, ignore_index=True)


# In[ ]:


# df = df.drop(columns=df.columns[0])


# In[ ]:


# df = df.drop(columns=df.columns[0])
# df = df.drop(columns='W')


# In[ ]:


# df.head(1)


# In[ ]:


# df.drop(list(df.filter(regex = 'pd')), axis = 1, inplace = True)


# In[ ]:


# df.head(1)


# In[ ]:


# len(df)


# ## Read test data

# In[ ]:


python OpenPose.py path


# In[ ]:


directory = path.split('/')[-1].split('.')[0]



parent_dir = "./test_videos_json_files"

path2 = os.path.join(parent_dir, directory) 


# In[ ]:


python parse_json_hc.py path2


# In[ ]:


f = path.split('/')[-1].split('.')[0]



parent_dir = "./test_data"
#parent_dir = "/mnt/sde/jagadish/userdata/dl_project/tv_test_data/"

filename = os.path.join(parent_dir + f + '.csv')


# In[231]:


test = pd.read_table(filename,sep=",")


# In[232]:


gf = test


# In[233]:


gf = gf.drop(columns=gf.columns[0])


# In[234]:


gf.drop(list(gf.filter(regex = 'pd')), axis = 1, inplace = True)


# In[235]:


gf.head(1)


# ## Get features and labels

# In[236]:


# def get_feature_label(data):
#     # remove outliers
#     #data_after = data[(data['price']<400) & (data['price']>1)]
#     #data_after = data[data['price']>1]
#     # split features and labels
#     #train_features = data.drop(['responded'],axis=1)
#     train_features = data.drop(['Y'],axis=1)
#     train_labels = data.Y
#     return train_features,train_labels


# In[237]:


# train_features,train_labels=get_feature_label(df)
# train_features=train_features
# train_labels=train_labels
# test_features,test_labels=get_feature_label(gf)
# test_features=test_features
# test_labels = test_labels


# In[ ]:


test_features = gf


# In[ ]:


# train_features.head(1)


# In[ ]:


test_features.head(1)


# In[ ]:


# X = train_features
# y = train_labels


# ## Compile and fit the model

# In[ ]:


# rmsprop = optimizers.RMSprop(lr=0.001)
# adam = optimizers.Adam(lr=0.0001)
# sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# ada =optimizers.Adadelta(lr=0.0001, rho = 0.95, epsilon = 1e-07)


# In[ ]:


#x_t, x_v, y_t, y_v = train_test_split(X, y,test_size=0.05, random_state=0, stratify=y)


# In[ ]:


# import time
# current_t = time.time()

# verbose, epochs, batch_size = 1, 35, 15
# n_samples,n_features, n_outputs = 31499,4, 1
# # define model
# model = Sequential()
# #kernel_regularizer=regularizers.l2(0.01),
# #model.add(LSTM(500, activation='relu',return_sequences=False, input_shape=(4, 1)))
# model.add(Dense(200, activation='relu',
#                 kernel_regularizer=regularizers.l2(0.001), input_shape=(n_features,)))
# #model.add(BatchNormalization())
# model.add(Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
# #model.add(Dense(100, activation='relu'))
# #model.add(Dense(512, activation='relu'))
# model.add(Dense(n_outputs, activation='sigmoid'))
# model.compile(loss='binary_crossentropy',metrics=['accuracy'], optimizer='Adam')
# model.summary()
# # fit network
# history = model.fit(X, y, epochs=epochs, batch_size=batch_size,validation_split=0.0, verbose=verbose)



# ## Save model

# In[ ]:


#model.save('hc_model_new_28_final_ffo.h5')  # creates a HDF5 file 'hc_model.h5'


# ## Load the saved model

# In[259]:


## returns a compiled model
# identical to the previous one
#model = load_model('hc_model_new_1000_u1.h5')
model = load_model('./pretrained_model/hc_model_new_60_final_fo.h5')


# ## Plotting the results

# In[260]:


# import matplotlib.pyplot as plt
# #acc = history.history['acc']
# #val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs = range(1, len(loss) + 1)

# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.savefig('Watching_TV_train_val_loss_curve.jpg')  # saves the current figure
# plt.show()


# In[261]:


# import matplotlib.pyplot as plt
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs = range(1, len(acc) + 1)

# plt.plot(epochs, acc, 'go', label='Training accuracy')
# plt.plot(epochs, val_acc, 'g', label='Validation accuracy')
# plt.title('Training and validation accuracy')
# plt.legend()
# plt.savefig('Watching_TV_train_val_accuracy_curve.jpg')  # saves the current figure
# plt.show()


# ## Testing the model

# In[262]:


y_p = model.predict(test_features, verbose=0)
results = y_p


# ## Classification Metrics

# In[263]:


results[results<=0.5]=0
results[results>0.5]=1


# In[264]:


# # Creating the Confusion Matrix
# from sklearn.metrics import confusion_matrix
# #cm = confusion_matrix(y_test, results)
# cm = confusion_matrix(test_labels, results)


# In[265]:


# cm


# In[266]:


# y_pred = results
# y_true = test_labels


# In[267]:


# accuracy = format(accuracy_score(y_true, y_pred),'.4f')


# sensitivity = format(recall_score(y_true, y_pred,pos_label=1,average='binary'),'.4f')

# specificity = format(recall_score(y_true, y_pred,pos_label=0,average='binary'),'.4f')

# print('Accuracy : ', accuracy)   
# print('Sensitivity : ', sensitivity)
# print('Specificity : ', specificity)


# In[244]:


#print ("Features_extraction complete. Time elapsed: " + str(int(time.time()-current_t )) + "s")


# ## Save JSON file with time and label information

# In[245]:


kf = results


# In[246]:


hf = pd.DataFrame(kf)


# In[247]:


#results


# In[248]:


mf = pd.DataFrame(columns=['Hand_On_Chest'])


# In[249]:


cap = cv.VideoCapture(path)   # capturing the video from the given path
fps = cap.get(cv.CAP_PROP_FPS) # Getting Franme rate of the video


# In[250]:


fps


# In[251]:


n= hf.index
l=[]
c=0
for i in n[:] :
    
    l.append(c/fps)
    l.append(hf.iloc[i][0])
    
    mf = mf.append({'Hand_On_Chest':l[:]}, ignore_index=True)
    l=[]
    c+=1


# In[252]:


mf.head()


# In[253]:


mf.to_json('timeLable.json')


# ## Plot and save "Time vs Label" graph

# In[254]:


pf = pd.DataFrame(columns=['Time', 'Label'])


# In[255]:


n= hf.index
c=0
for i in n[:] :
    

    
    pf = pf.append({'Time': c/fps, 'Label': hf.iloc[i][0]}, ignore_index=True)

    c+=1


# In[256]:


pf.head()


# In[257]:


time = pf['Time']
label1 = pf['Label']


# In[258]:


plt.figure(figsize=(20,10))
plt.plot(time, label1, 'g')
#plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.xticks(fontsize=20, fontweight='bold',rotation=90)
plt.yticks(fontsize=20, fontweight='bold')
plt.xlabel('Time (seconds)',fontsize=20, fontweight='bold')
plt.ylabel('Label',fontsize=20, fontweight='bold')
plt.title('Time vs Label', fontsize=20, fontweight='bold')
plt.tight_layout()
#plt.legend()
plt.savefig('timeLable.jpg')  # saves the current figure
plt.show()


# In[148]:


#of = pd.DataFrame(data=results,columns=['Label'])


# In[ ]:


#of.to_csv('out_8346.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




