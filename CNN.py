import tensorflow as tf
from tensorflow.keras import losses, Model, metrics
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Flatten, Reshape, Masking
import skimage as sk
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

mask_rad=np.array(np.load('/Users/cin/Documents/pyfile/mask.npz')['arr_0'])
mask_rad=np.delete(mask_rad,-1,axis=1)
mask_rad=np.delete(mask_rad,-1,axis=0)
print(np.max(mask_rad),np.min(mask_rad))

input_train=[]
for f in np.array(np.load('/Users/cin/Documents/pyfile/CNN/input_train.npz')['arr_0']):
    input_train.append(tf.convert_to_tensor(f))
input_train=np.array(input_train)
output_train=[]
for f in np.array(np.load('/Users/cin/Documents/pyfile/CNN/output_train.npz')['arr_0']):
    output_train.append(tf.convert_to_tensor(f,dtype=tf.bool))
output_train=np.array(output_train)


print(np.max(input_train),np.min(input_train),np.max(output_train),np.min(output_train))

#Build Model
#model=models.Sequential()
inp=Input(shape=(1000,1000,3))
mask=Masking(mask_value=0)(inp)
conv1=Conv2D(16,(3,3),activation='softmax',padding='same')(mask)
conv2=Conv2D(16,(3,3),activation='softmax',padding='same')(conv1)
pool1=MaxPooling2D((2,2),strides=2)(conv2)
conv3=Conv2D(32,(3,3),activation='softmax',padding='same')(pool1)
conv4=Conv2D(32,(3,3),activation='softmax',padding='same')(conv3)
pool2=MaxPooling2D((2,2),strides=2)(conv4)
conv5=Conv2D(64,(3,3),activation='softmax',padding='same')(pool2)
conv6=Conv2D(64,(3,3),activation='softmax',padding='same')(conv5)
pool3=MaxPooling2D((2,2),strides=2)(conv6)
conv7=Conv2D(128,(3,3),activation='softmax',padding='same')(pool3)
conv12=Conv2D(128,(3,3),activation='softmax',padding='same')(conv7)
up2=UpSampling2D((2,2))(conv12)
con2=Concatenate()([conv6,up2])
conv13=Conv2D(64,(3,3),activation='softmax',padding='same')(con2)
conv14=Conv2D(64,(3,3),activation='softmax',padding='same')(conv13)
up3=UpSampling2D((2,2))(conv14)
con3=Concatenate()([conv4,up3])
conv15=Conv2D(32,(3,3),activation='softmax',padding='same')(con3)
conv16=Conv2D(32,(3,3),activation='softmax',padding='same')(conv15)
up4=UpSampling2D((2,2))(conv16)
con3=Concatenate()([conv2,up4])
conv17=Conv2D(16,(3,3),activation='softmax',padding='same')(con3)
conv18=Conv2D(16,(3,3),activation='softmax',padding='same')(conv17)
conv19=Conv2D(2,(3,3),activation='softmax',padding='same')(conv18)
conv20=Conv2D(1,(3,3),activation='softmax',padding='same')(conv19)
flat=Flatten()(conv20)
print('ok')
#dense=Dense(1016064,activation='relu')(flat)
out=Reshape((1000,1000))(flat)

model=Model(inputs=inp,outputs=out)


model.summary()
with open('/Users/cin/Documents/pyfile/CNN/model_summary.txt','w') as f:
    model.summary(print_fn=lambda x: f.write(x+'\n'))

model.compile(optimizer=Adam(),loss=losses.CategoricalCrossentropy(),metrics=metrics.Precision(thresholds=0.8,name='POD'))
#hist=model.fit(x=input_train,y=output_train,batch_size=3,epochs=10,validation_data=(input_test,output_test))
hist=model.fit(x=input_train,y=output_train,batch_size=32,epochs=10,steps_per_epoch=28)
model.save('/Users/cin/Documents/pyfile/CNN/CNN_longsor.h5')
model.save_weights('/Users/cin/Documents/pyfile/CNN')

plt.plot(hist.history['POD'],label='POD')
plt.xlabel('Epoch')
plt.ylabel('POD')
#plt.legend(loc='lower right')
plt.show()

plt.plot(hist.history['loss'],label='loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
#plt.legend(loc='lower right')
plt.show()


input_test=[]
for f in np.array(np.load('/Users/cin/Documents/pyfile/CNN/input_test.npz')['arr_0']):
    input_test.append(tf.convert_to_tensor(f))
input_test=np.array(input_test)
output_test=[]
for f in np.array(np.load('/Users/cin/Documents/pyfile/CNN/output_test.npz')['arr_0']):
    output_test.append(tf.convert_to_tensor(f))
output_test=np.array(output_test)
print(np.shape(input_train))
print(output_train[0])

test_loss,test_acc=model.evaluate(input_test,output_test)
print(test_acc)
print(test_loss)





