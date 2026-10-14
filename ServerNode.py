#!/usr/bin/env python
# coding: utf-8

# In[1]:


import socket
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime


# In[2]:


print('***************************** Server Execution Starts ***************************** ')
now = datetime.now()
current_time = now.strftime("%H:%M:%S.%f")
print("Time when Server Execution is starting =", current_time)


# In[3]:


# Read model weight from Clinet1, Clinet2 and Clinet3 representing USA, China and India

Client1_weight = pickle.load(open('Client1.pkl', 'rb'))

Client2_weight = pickle.load(open('Client2.pkl', 'rb'))

Client3_weight = pickle.load(open('Client3.pkl', 'rb'))


# In[4]:


# Derive model weight from the weights received from the clinets
agg_weight_arr = np.array(Client1_weight) + np.array(Client2_weight) + np.array(Client3_weight)
Central_weight=(1/3)*np.array(agg_weight_arr)


# In[5]:


# Save model weights
pickle.dump(Central_weight, open('Central_Server.pkl', 'wb'))


now = datetime.now()
current_time = now.strftime("%H:%M:%S.%f")
print("Time when Server Execution is ending =", current_time)
print('***************************** Server Execution Ends ***************************** ')


# In[ ]:





# In[ ]:




