#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[975]:


import pandas as pd
import time
import numpy as np
import math
from itertools import islice 
import json
from pandas.io.json import json_normalize
import os
from sys import argv
import sys


# In[ ]:


path = sys.argv[1]


# ## Read and parse json files for each video

# In[976]:



# this finds our json files
path_to_json = path
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]

# here I define my pandas Dataframe with the columns I want to get from the json
jsons_data = pd.DataFrame(columns=['index','person_id','po'])
df = pd.DataFrame()


# we need both the json and an index number so use enumerate()
for index, js in enumerate(json_files):
    with open(os.path.join(path_to_json, js)) as json_file:
        json_text = json.load(json_file)
        if json_text['people']!= []:
            index = (str(json_file).split('/')[-1].split('_')[2])
            person_id = json_text['people'][0]['person_id']
            po = json_text['people'][0]['pose_keypoints_2d']
            # here I push a list of data into a pandas DataFrame at row given by 'index'
            jsons_data.loc[index] = [index,person_id, po]


# now that we have the pertinent json data in our DataFrame let's look at it
print(jsons_data)


# ## Preprocessing data

# In[977]:


df = jsons_data


# In[978]:


df['index'] = df['index'].astype(int)


# In[979]:


df=df.sort_index()


# In[980]:


df.head(5)


# In[981]:


#df.to_csv('tt.csv', index=True)


# In[982]:


df.shape


# In[983]:


gf = pd.DataFrame()


# In[984]:


gf=df


# In[985]:


gf.head(1)


# In[986]:


gf = gf.join(gf['po'].apply(pd.Series).add_prefix('po'))


# In[987]:


gf = gf.drop(['po','person_id','index'], axis=1)


# In[988]:


gf.head(1)


# In[989]:


gf = gf.drop(gf.iloc[:, 2::3],axis=1)


# In[990]:


gf.head(1)


# In[991]:


mf = pd.DataFrame()


# In[992]:


n = gf.shape[1] 
i=0
j=0
while i < n:

    col = gf.columns[i][0:2] + "_" + str(j) #col name
    X = gf.columns[i]
    Y = gf.columns[i+1]
    mf[col] = gf[[X, Y]].values.tolist()
    i = i+2
    j= j+1


# In[993]:


mf.head(1)


# In[994]:


hf=pd.DataFrame()


# In[995]:


hf = mf


# In[996]:


hf.head(1)


# In[997]:


hf.shape[1]


# ## Euclidean distances between body key points

# In[998]:


def eudis5(v1, v2): # Function to calculate euclidean distance between two points
    dist = [(a - b)**2 for a, b in zip(v1, v2)]
    dist = math.sqrt(sum(dist))
    return dist


# In[999]:


hf['pd'] = ''
hf['pd'] = hf['pd'].apply(list)


# In[1000]:


n=hf.index
m = hf.shape[1]
for i in n[:] :
    
    ear =[]
    I=1
    for j in range(25) :
        
        for k in range(I,25) :

            X = hf.columns[j]
            Y = hf.columns[k]

            a = np.array(hf[X][i])
            b = np.array(hf[Y][i])
            x = eudis5(a, b)
            ear.append(x)
        I = I + 1
     
    hf.loc[i,'pd'].append(ear[:])


# In[1001]:


hf.head()


# ## Angle between upper_arm and fore_arm_left_hand

# In[1002]:


n=hf.index
for i in n[:] :
    


    X = 'po_5'
    Y = 'po_6'
    Z = 'po_7'

    a = np.array(hf[X][i])
    b = np.array(hf[Y][i])
    c = np.array(hf[Z][i])
    
    ba = a - b
    bc = c - b
    

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    
    if np.isnan(angle):
        hf.loc[i,'hl_angle'] = 0
    else:
        hf.loc[i,'hl_angle'] = np.degrees(angle)
    

    

  


# In[1003]:


hf.head()


# ## Angle between upper_arm and fore_arm_right_hand

# In[1004]:


n=hf.index
for i in n[:] :
    
    X = 'po_2'
    Y = 'po_3'
    Z = 'po_4'

    a = np.array(hf[X][i])
    b = np.array(hf[Y][i])
    c = np.array(hf[Z][i])
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    
    if np.isnan(angle):
        hf.loc[i,'hr_angle'] = 0
    else:
        hf.loc[i,'hr_angle'] = np.degrees(angle)
    


 


# In[1005]:


hf.head()


# ## Distance between left hand and neck

# In[1006]:


n=hf.index
for i in n[:] :
    
    X = 'po_1'
    Y = 'po_7'
    Z = 'po_8'

    a = np.array(hf[X][i])
    b = np.array(hf[Y][i])
    c = np.array(hf[Z][i])
    
    ab = eudis5(a,b)
    ac = eudis5(a,c)
    
    
    if ac!=0:
        dist_ratio = ab/ac
    else:
        dist_ratio = 0
    
    hf.loc[i,'hl_dist'] = dist_ratio

    


# In[1007]:


hf.head()


# ## Distance between right hand and neck

# In[1008]:


n=hf.index
for i in n[:] :
    
    X = 'po_1'
    Y = 'po_4'
    Z = 'po_8'

    a = np.array(hf[X][i])
    b = np.array(hf[Y][i])
    c = np.array(hf[Z][i])
    
    ab = eudis5(a,b)
    ac = eudis5(a,c)
    
    if ac!=0:
        dist_ratio = ab/ac
    else:
        dist_ratio = 0
    
    hf.loc[i,'hr_dist'] = dist_ratio

    


# In[1009]:


hf.head()


# ## Filtering the dataframe with desired columns

# In[1010]:


df= pd.DataFrame()


# In[1011]:


#df=hf.filter(items=['pd', 'hl_angle', 'hr_angle', 'hl_dist', 'hr_dist'])
df=hf.filter(items=['hl_angle', 'hr_angle', 'hl_dist', 'hr_dist'])


# In[1012]:


df.head(1)


# In[1013]:


#df = df.join(df['pd'].apply(pd.Series).add_prefix('p_'))


# In[1014]:


df.head(1)


# In[1015]:


#df = df.join(df['p_0'].apply(pd.Series).add_prefix('pd_'))


# In[1016]:


df.head(1)


# In[1017]:


#df = df.drop(['pd', 'p_0'], axis=1)


# In[1018]:


df.head(1)


# In[1019]:


#df.drop([col for col, val in df.sum().iteritems() if val == 0], axis=1, inplace=True)


# In[1020]:


df.head(1)


# In[ ]:


f = path.split('/')[-1].split('.')[0]
parent_dir = "./test_data"

filename = os.path.join(parent_dir + f + '.csv')


df.to_csv(filename, index=True)


# In[ ]:




