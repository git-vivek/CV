#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2


# In[2]:


fd=cv2.CascadeClassifier("C:\\Users\\vivek\\Downloads\\frontal.xml")


# In[3]:


cap=cv2.VideoCapture(0)


# In[4]:


while cap.isOpened():
    status,img=cap.read()
    grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face=fd.detectMultiScale(grey,1.15,5)
    for x,y,w,h in face:
        cv2.rectangle(grey,(x,y),(x+w,y+h),(0,0,255),2)
    if len(face)==0:
        print("No faces")
    else:
        print(face.shape)
        print("Number of faces: " + str(face.shape[0]))
        if face.shape[0]==2:
            stat,frame=cap.read()
            cv2.imwrite('harsh.jpg',frame)
        
    #print(face.shape)
         #   print('hello')
    #print(type(face))
    cv2.imshow('grey',grey)
    if cv2.waitKey(25) & 0xff==ord('q'):
        break
cv2.destroyAllWindows()
cap.release()


# In[ ]:




