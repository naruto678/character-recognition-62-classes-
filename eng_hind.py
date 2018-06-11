#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 23:34:04 2018

@author: arnab
"""
import numpy as np
#
def make_data():
    import cv2
    import numpy as np
    import os
    org_dir='Sample'
    data=np.zeros((6500,128,128,3),dtype='float64')
    labels=np.zeros((6500,1))
    
    count=0
    for i in range(1,63):
        if i<10:
            sample='00'+str(i)+'/'
            
            
            result_dir=org_dir+sample
        else:
            sample='0'+str(i)+'/'
            result_dir=org_dir+sample
        
        file_list=os.listdir(result_dir)[:100]
        for image in file_list:
            path=result_dir+image
            y=cv2.imread(path)
            y=y.astype('float64')
            
            
            y=y.astype('float64')
            data[count,:,:,:]=y/255
            labels[count]=i-1
    
             
                
            count=count+1
            
            
            
            
            
            
            
            
    return data,labels

eng_data,eng_labels_0=make_data()


eng_labels=eng_labels_0
eng_labels=eng_labels.astype('float32')        
from sklearn.model_selection import train_test_split
partial_X_train,val_X_train,partial_Y_train,val_Y_train=train_test_split(eng_data,eng_labels,test_size=0.30,random_state=42)            
#partial_X_train=partial_X_train.reshape((8695,,48,3))
#val_X_train=val_X_train.reshape((512,48,48,3))          
            

def vectorize_sequences(sequences, dimension=62):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        
        results[i,sequence] = 1
    return results        
partial_Y_train=vectorize_sequences(partial_Y_train.astype('int')).astype('float32')
val_Y_train=vectorize_sequences(val_Y_train.astype('int')).astype('float32')

from keras.applications import VGG16
conv_base = VGG16(weights='imagenet',
include_top=False,
input_shape=(128, 128, 3))

VGG_pred=conv_base.predict(partial_X_train)

validation_pred=conv_base.predict(val_X_train)
VGG_pred=VGG_pred.reshape((2898,512))
validation_pred=validation_pred.reshape((512,512))
np.save('VGG_pred.npy',VGG_pred)
np.save('validation_pred.npy',validation_pred)
np.save('partial_Y_train.npy',partial_Y_train)
np.save('val_Y_train.npy',val_Y_train)
np.save('new_VGG_pred.npy',VGG_pred)
np.save('new_validation_pred.npy',validation_pred)

VGG_pred=np.load('new_VGG_pred.npy')
validation_pred=np.load('new_validation_pred.npy')
partial_Y_train=np.load('partial_Y_train.npy')
val_Y_train=np.load('val_Y_train.npy')
VGG_pred=VGG_pred.reshape((4550,4*4*512))
validation_pred=validation_pred.reshape((1950,4*4*512))

def make_model():
    from keras import layers
    from keras import models
    model=models.Sequential()
    
    model.add(layers.Dense(600,activation='relu',input_shape=(8192,)))
    model.add(layers.Dense(1000,activation='relu'))
    model.add(layers.Dense(250,activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(128,activation='relu'))
    
    model.add(layers.Dense(62,activation='softmax'))
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
    return model
speech_model=make_model()
#from keras.utils import to_categorical
#val_Y_train=to_categorical(val_Y_train,num_classes=62)
#partial_Y_train=to_categorical(partial_Y_train,num_classes=62)
speech_model.fit(VGG_pred,partial_Y_train,epochs=10,batch_size=50,validation_data=(validation_pred,val_Y_train))



#from keras.preprocessing.image import ImageDataGenerator
#train_datagen = ImageDataGenerator(
#rescale=1./255,
#rotation_range=40,
#width_shift_range=0.2,
#height_shift_range=0.2,
#shear_range=0.2,
#zoom_range=0.2,
#horizontal_flip=True,
#fill_mode='nearest')
#
#test_datagen=ImageDataGenerator(rescale=1./255)
#
#train_data=train_datagen.flow_from_directory('train_dir',target_size=(48,48,3))
#
#
#
#
n1=np.zeros((3,128,128,3))
y=y.astype('float64')
y1=y1.astype('float64')
y2=y2.astype('float64')
n1[0,:,:,:]=y/255
n1[1,:,:,:]=y1/255
n1[2,:,:,:]=y2/255

n1=conv_base.predict(n1)

n1=n1.reshape((3,8192))
print((np.argmax(speech_model.predict(n1),axis=1)))

cv.imshow('image1',y1)
cv.waitKey(0)
cv.destroyAllWindows()

