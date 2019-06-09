#!/usr/bin/env python
# coding: utf-8

# In[2]:


#get_ipython().system('pip install opencv-contrib-python')


# In[21]:


import cv2 
import pandas as pd
import numpy as np


# In[22]:


cap=cv2.VideoCapture(0)


# In[23]:


while cap.isOpened():
    status,img=cap.read()
    (h,w)=img.shape[:2] #getting values of height and width
    centre=(h/2,w/2) # getting center 
    rot=cv2.getRotationMatrix2D(centre,90,1) # rotating with 90 as angle and 1 as scaling factor
    final=cv2.warpAffine(img,rot,(h,w)) #using affine transformation
    
    cv2.imshow('1',img)
    cv2.imshow('2',final)
    if cv2.waitKey(0) & 0xff==ord('q'):
        break
cv2.destroyAllWindows()
cap.release()


# In[15]:





# In[16]:





# In[17]:





# In[19]:





# In[ ]:




