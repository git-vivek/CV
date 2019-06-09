#!/usr/bin/env python
# coding: utf-8

# In[12]:


import cv2


# In[ ]:





# In[ ]:





# In[ ]:





# In[14]:


cap=cv2.VideoCapture(0)
video_plugin=cv2.VideoWriter_fourcc(*'XVID')
op=cv2.VideoWriter('new.avi',video_plugin,60,(640,480))
while cap.isOpened():
    status,frame=cap.read()
    cv2.imshow('1',frame)
    op.write(frame)
    if cv2.waitKey(10) & 0xff==ord('q'):
        break
cv2.destroyAllWindows()
cap.release()
    


# In[ ]:




