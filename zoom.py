#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import pandas as pd
import numpy as np


# In[3]:


cap=cv2.VideoCapture(0)


# In[5]:


while cap.isOpened():
    status,img=cap.read()
    zin=cv2.resize(img,None,fx=2,fy=2) # None is used to specify that we want to use scaling factors
    zout=cv2.resize(img,None,fx=0.5,fy=0.5)
    cv2.imshow('1',img)
    cv2.imshow('2',zin)
    cv2.imshow('3',zout)
    if cv2.waitKey(0) & 0xff==ord('q'):
        break
cv2.destroyAllWindows()
cap.release()


# In[ ]:




