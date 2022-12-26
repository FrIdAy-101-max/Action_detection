#!/usr/bin/env python
# coding: utf-8

# In[9]:


import cv2
from tensorflow.keras import datasets,layers,models
import os
from matplotlib import pyplot
import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten,Dropout
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.optimizers import Adam,SGD
import keras
from keras.layers import LeakyReLU


# In[10]:


train = ImageDataGenerator(rescale = 1/256)
train_dataset = train.flow_from_directory("C:\\Users\\nithe\\Downloads\\data-20220906T142508Z-001\\data\\train" ,target_size = (256,256),batch_size = 64, class_mode ='binary')
validation=ImageDataGenerator(rescale = 1/256)
validation_dataset=validation.flow_from_directory("C:\\Users\\nithe\\Downloads\\data-20220906T142508Z-001\\data\\val" ,target_size = (256,256),batch_size = 64, class_mode ='binary')


# In[ ]:





# In[ ]:





# In[ ]:





# In[11]:



import keras.layers as layers
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3), padding="same", input_shape=(256, 256, 3)))
model.add(LeakyReLU(0.1))
model.add(MaxPool2D(pool_size=4))
model.add(Conv2D(filters=32, kernel_size=(3,3),padding="same"))
model.add(LeakyReLU(0.1))
model.add(MaxPool2D(pool_size=4))
model.add(Conv2D(filters=64, kernel_size=(3,3),padding="same"))
model.add(LeakyReLU(0.1))
model.add(MaxPool2D(pool_size=4))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid'))
model.summary()


# In[12]:


from keras.layers import LeakyReLU


# In[15]:


opt=keras.losses.BinaryCrossentropy(from_logits=True)
model.compile(loss='binary_crossentropy', 
              optimizer='SGD', 
              metrics=['accuracy'])


# In[16]:


model.fit(train_dataset,batch_size=30,epochs=5,validation_data=validation_dataset,verbose=1, shuffle=True)


# In[17]:


test = ImageDataGenerator(rescale = 1/256)
test_dataset = test.flow_from_directory("C:\\Users\\nithe\\Downloads\\inference" ,target_size = (256,256),batch_size = 64, class_mode ='binary')
result = model.evaluate(test_dataset)


# In[18]:


model.save("C:\\Users\\nithe\\PycharmProjects\\pstreamlit\\model")


# In[3]:


from keras.models import load_model

reconstructed_model = load_model("C:\\Users\\nithe\\PycharmProjects\\pstreamlit\\model")


# In[25]:


from tensorflow.keras.preprocessing import image
img=img = image.load_img("C:\\Users\\nithe\\PycharmProjects\\deep vision\\signed\\i2.png")
img_array = image.img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)
a=reconstructed_model.predict(img_batch)
print(a)


# In[26]:


import matplotlib.pyplot as plt
plt.imshow(img)
plt.show()


# In[4]:


reconstructed_model.save('final.h5')


# In[ ]:




