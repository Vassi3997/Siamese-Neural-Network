#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().system('pip install opencv-python')


# In[2]:


import numpy as np
import cv2
#import matplotlib.pyplot as plt
import keras
import os

#from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


from keras.utils import np_utils
from keras.models import Model, Sequential
from keras.layers import Activation,Input
from keras.layers.core import Lambda, Dense, Flatten
from keras.layers.convolutional import *
from keras.layers.pooling import *
from keras.optimizers import Adam
from keras.regularizers import l2
#import pylab as plt
import random

import warnings
warnings.filterwarnings('ignore')


# In[3]:


PATH = os.getcwd()
# Define data path
#data_path = PATH + "/people"
people=os.listdir('people')

batch_size=5
#img_rows=160
#simg_cols=160
epochs=3
x_data=[]
y_data=[]

# Define the number of classes
num_classes = 4

for x in people:
	print("Loading dataset of - {}\n".format(x[:-1]))
	for i in os.listdir('people/'+x):
		img=cv2.imread('people'+'/'+x+'/'+i,3)
		img=cv2.resize(img,(160,160))
		img=img.astype('float')/255.0
		#img=np.expand_dims(img,axis=0)
		x_data.append(img)
		y_data.append(int(x[-1]))

x_data=np.array(x_data)
y_data=np.array(y_data)
y_data=y_data.reshape(len(y_data),1)


print(np.unique(y_data,return_counts=True))

# convert class labels to on-hot encoding
#y = keras.utils.to_categorical(y_data, num_classes)


# In[11]:


print(x_data.shape[:])

# In[4]:


#Creating training dataset to feed the neural network
image_list = x_data[:]
label_list = y_data[:]

left_input = []
right_input = []
targets = []

#print(image_list.shape[0])
#Number of pairs per image

for a in range(0,int(image_list.shape[0])):
  for b in range(a+1,int(image_list.shape[0])):
    left_input.append(image_list[a])
    right_input.append(image_list[b])
    if (label_list[a]==label_list[b]):
      targets.append(1.)
    else:
      targets.append(0.)
     
    
left_input = np.array(left_input)
right_input = np.array(right_input)
targets = np.array(targets)

l_train,l_test,r_train,r_test,t_train,t_test = train_test_split(left_input,right_input,targets,test_size=0.3)

l_test = np.squeeze(l_test)
r_test = np.squeeze(r_test)
l_train = np.squeeze(l_train)
r_train = np.squeeze(r_train)

#print(l_test)
# In[12]:


print(l_test.shape[:])
print(r_test.shape[:])
print(l_train.shape[:])
print(r_train.shape[:])


# In[5]:


left_input = Input((160,160,3))
right_input = Input((160,160,3))

convnet = Sequential([
    Conv2D(128, kernel_size=(3, 3), input_shape=(160,160,3)),Activation('relu'),
    MaxPooling2D(),
    Conv2D(64,3),Activation('relu'),
    MaxPooling2D(),
    Conv2D(32,3),Activation('relu'),
    MaxPooling2D(),
    Conv2D(16,3),Activation('relu'),
    Flatten(),
    Dense(16),Activation('sigmoid')
])

encoded_l = convnet(left_input)
encoded_r = convnet(right_input)

L1_layer = Lambda(lambda tensor:K.abs(tensor[0] - tensor[1]))

L1_distance = L1_layer([encoded_l, encoded_r])

prediction = Dense(1,activation='sigmoid')(L1_distance)
siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

optimizer = Adam(0.001, decay=2.5e-4)


# In[6]:


siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=['accuracy'])


# In[7]:

print("\nSummary For Siamese Network")
siamese_net.summary()


# In[8]:


print("\nSummary For Convolutional Network")
convnet.summary()


# In[9]:


siamese_net.fit([l_train,r_train], t_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=([l_test,r_test],t_test))

siamese_net.save('siamese_net.MODEL')
# In[ ]:





