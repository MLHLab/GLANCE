#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy import loadtxt
import os
import scipy
import collections
import pathlib
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import PIL.Image
from datetime import datetime

import tensorflow as tf

import keras
#from keras.models import Sequential, Model
#from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.layers import Concatenate, Multiply, Add, Reshape, GlobalMaxPooling2D
from tensorflow.keras import backend as K

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import Activation 
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD
from tensorflow import reshape, nest, config
from tensorflow.keras import losses, metrics, optimizers

from sklearn.model_selection import train_test_split

import pickle

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

from numpy.random import seed

# Set seeds

seed(1)
tf.random.set_seed(2)
tf.compat.v1.set_random_seed(2019)


# In[2]:


now = datetime.now()
current_time = now.strftime("%H:%M:%S.%f")
print("Time when code is starting =", current_time)
print('***************************** Iteration 1 Starts ***************************** ')


# # Data Viral

# In[3]:


data_dir = pathlib.Path("D:/Research/Paper_1/Data1")
image_count = len(list(data_dir.glob('*/*')))
print(image_count)


# In[4]:


#Rescaling the image
image_convert = ImageDataGenerator(rescale = 1./255)

#Flow the images from the input directory into an array, applies the pre-processing defined in image_convert function, resizes all images in same size 
#image_preprocess = image_convert.flow_from_directory(directory=data_dir, target_size=[3, 3], class_mode="binary", batch_size=1000,seed=32)
image_preprocess = image_convert.flow_from_directory(directory=data_dir, target_size=[180, 180], class_mode="binary", batch_size=4000,seed=32)
data, labels = image_preprocess.next()


# In[5]:


labels = labels. astype(int)
print(type(data))
print(data.shape)
print(type(labels))
print(labels.shape)
print(sum(labels))

print(labels)


# In[6]:


# split the data into train and test sets
# this lets us simulate how our model will perform in the future
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3,random_state=30)

print(X_train.shape)
print(y_train.shape)
print(sum(y_train))

print(X_test.shape)
print(y_test.shape)
print(sum(y_test))


# In[7]:


def CBAM(inputs, ratio=8):
    
    # ==========================================
    # 1. Channel Attention
    # ==========================================
    
    channel = inputs.shape[-1]
    
    # Average pooling
    avg_pool = GlobalAveragePooling2D()(inputs)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    
    # Max pooling
    max_pool = GlobalMaxPooling2D()(inputs)
    max_pool = Reshape((1, 1, channel))(max_pool)
    
    # Shared MLP
    shared_dense_1 = Dense(
        channel // ratio,
        activation="relu",
        use_bias=False
    )
    
    shared_dense_2 = Dense(
        channel,
        use_bias=False
    )
    
    avg_out = shared_dense_2(shared_dense_1(avg_pool))
    max_out = shared_dense_2(shared_dense_1(max_pool))
    
    # Combine average and max information
    channel_attention = Add()([avg_out, max_out])
    channel_attention = K.sigmoid(channel_attention)
    
    # Apply channel attention
    x = Multiply()([inputs, channel_attention])
    
    
    # ==========================================
    # 2. Spatial Attention
    # ==========================================
    
    # Average across channels
    avg_spatial = K.mean(x, axis=-1, keepdims=True)
    
    # Maximum across channels
    max_spatial = K.max(x, axis=-1, keepdims=True)
    
    # Combine
    spatial = Concatenate(axis=-1)([
        avg_spatial,
        max_spatial
    ])
    
    # Generate spatial attention map
    spatial_attention = Conv2D(
        filters=1,
        kernel_size=7,
        padding="same",
        activation="sigmoid",
        use_bias=False
    )(spatial)
    
    # Apply spatial attention
    output = Multiply()([x, spatial_attention])
    
    return output


# In[8]:


input_img = Input(shape=(180, 180, 3))

x = Conv2D(16, (3, 3), activation='relu')(input_img)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(32, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

x = CBAM(x)                                   #Squeeze and Excitation Attention Layer

x = Flatten()(x)

x = Dense(550, activation='relu')(x)
x = Dropout(0.1,seed=2019)(x)

x = Dense(400, activation='relu')(x)
x = Dropout(0.3,seed=2019)(x)

x = Dense(200, activation='relu')(x)
x = Dropout(0.2,seed=2019)(x)

output = Dense(3, activation='softmax')(x)                  # Add Output layer

model = Model(inputs=input_img, outputs=output)


# In[9]:


# Print the model summary
model.summary()


# In[10]:


model.compile(optimizer='Adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model

r = model.fit(X_train,y_train, epochs=40, batch_size=1000, shuffle=True)


# In[11]:


# Print final model performance

print("Train score:", model.evaluate(X_train, y_train))
print("Test score:", model.evaluate(X_test, y_test))


# In[12]:


# predict probabilities for test set
yhat_probs = model.predict(X_test, verbose=0)
# predict crisp classes for test set
#yhat_classes = model.predict_classes(X_test, verbose=0)


yhat_classes_1 =model.predict(X_test, verbose=0)

yhat_classes = np.argmax(yhat_probs, axis=1)

# reduce to 1d array
yhat_probs = yhat_probs[:, 0]

print("Performance on Test Set")

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_test, yhat_classes,average='macro')
print('Precision: %f' % precision)

# recall: tp / (tp + fn)
recall = recall_score(y_test, yhat_classes,average='macro')
print('Recall/ Senseitivity: %f' % recall)

# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, yhat_classes,average='macro')
print('F1 score: %f' % f1)


# In[14]:


# Save model weights

pickle.dump(model.get_weights(), open('Client1.pkl', 'wb'))


now = datetime.now()
current_time = now.strftime("%H:%M:%S.%f")
print("Time when code is ending =", current_time)
print('***************************** Iteration 1 Ends ***************************** ')


# In[ ]:





# In[ ]:




