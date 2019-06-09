#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install opencv-contrib-python')


# In[18]:


import cv2


# In[31]:


import pandas as pd
import numpy as np


# In[38]:


cap=cv2.VideoCapture(0)


# In[39]:


while cap.isOpened():
    status,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #gray scale
    back=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR) # gray 2 BGR
    dual=np.hstack((frame,back)) #stacking horizaontally
    cv2.imshow('d',dual)
    #cv2.imshow('1',frame)
    #cv2.imshow('2',gray)
    if cv2.waitKey(0) & 0xff==ord('q'):
        break
        
cv2.destroyAllWindows()
cap.release()
    


# In[23]:





# In[24]:





# In[29]:





# In[ ]:




