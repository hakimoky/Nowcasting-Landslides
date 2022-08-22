import tensorflow as tf
from tensorflow.keras import losses,  metrics, models, utils
from tensorflow.keras.layers import Input, Reshape, experimental ,Conv2D, MaxPooling2D, UpSampling2D, Bidirectional, LSTM, Dense, Conv2DTranspose, Reshape, Dropout, ZeroPadding2D
import os
import numpy as np
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

input_train=np.load('/Users/cin/Documents/pyfile/CNN_BiLSTM/input_train_JJA.npz')['arr_0']
output_train=np.load('/Users/cin/Documents/pyfile/CNN_BiLSTM/output_train_JJA.npz')['arr_0']

#input_train=np.load('/Users/cin/Documents/pyfile/CNN_BiLSTM/input_train.npz')['arr_0']+31.5/(31.5+95.5)
#input_train=tf.data.Dataset.from_tensor_slices([np.load('/Users/cin/Documents/pyfile/CNN_BiLSTM/input_train.npz')['arr_0']])
#input_train=np.array_split(np.load('/Users/cin/Documents/pyfile/CNN_BiLSTM/input_train.npz')['arr_0'],1701,axis=0)
#input_train=[np.squeeze(x) for x in input_train]
#print(input_train)
#output_train=np.load('/Users/cin/Documents/pyfile/CNN_BiLSTM/output_train.npz')['arr_0']+31.5/(31.5+95.5)
#output_train=tf.data.Dataset.from_tensor_slices([np.load('/Users/cin/Documents/pyfile/CNN_BiLSTM/output_train.npz')['arr_0']])
#output_train=np.array_split(np.load('/Users/cin/Documents/pyfile/CNN_BiLSTM/output_train.npz')['arr_0'],1701,axis=0)
#output_train=[np.squeeze(x) for x in output_train]
#print(output_train)

#train_data=tf.data.Dataset.from_tensor_slices((input_train,output_train))
#train_input=input_train.cache()
#train_output=output_train.cache()
#print(np.shape(input_train),np.shape(output_train),np.shape(input_test),np.shape(output_test))
#print(output_test[0])
#print(np.max(input_train),np.min(input_train),np.max(output_train),np.min(output_train))

#print(np.max(input_train),np.min(input_train),np.max(output_train),np.min(output_train))

model= models.Sequential()
model.add(Input(shape=(1001,1001,18)))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(ZeroPadding2D())
model.add(MaxPooling2D((2,2),strides=2))
model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D((2,2),strides=2))
model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling2D((2,2),strides=2))
model.add(Reshape((15625,128)))
model.add(Bidirectional(LSTM(128,return_sequences=True)))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(64,return_sequences=True)))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(32,return_sequences=True)))
model.add(Dropout(0.5))
model.add(Reshape((125,125,64)))
model.add(UpSampling2D((2,2)))
model.add(Conv2DTranspose(64,(3,3),padding='same',activation='relu'))
model.add(UpSampling2D((2,2)))
model.add(Conv2DTranspose(32,(3,3),padding='same',activation='relu'))
model.add(UpSampling2D((2,2)))
model.add(Conv2DTranspose(16,(3,3),padding='same',activation='relu'))

model.add(Conv2D(2,(3,3),activation='relu'))
model.add(experimental.preprocessing.Resizing(1001,1001))

model.add(Conv2D(1,(1,1),padding='same',activation='linear'))
#model.add(Dense(100))

#model.add(Dense(1,activation='linear'))
model.add(Reshape((1001,1001)))


model.summary()
with open('/Users/cin/Documents/pyfile/CNN_BiLSTM/CNN_BiLSTM_summary_JJA.txt','w') as f:
    model.summary(print_fn=lambda x: f.write(x+'\n'))

model.compile(optimizer='adam',loss=losses.MeanSquaredError(),metrics=metrics.RootMeanSquaredError())
#hist=model.fit(x=input_train,y=output_train,batch_size=3,epochs=10,validation_data=(input_test,output_test))
hist=model.fit(x=input_train,y=output_train,batch_size=16,epochs=10,steps_per_epoch=26)
model.save('/Users/cin/Documents/pyfile/CNN_BiLSTM/CNN_BiLSTM_longsor_JJA.h5')



train_accur=hist.history['root_mean_squared_error']
plt.plot(train_accur,label='RMSE')
plt.xlabel('Epoch')
plt.ylabel('RMSE ')
#plt.legend(loc='lower right')
plt.show()

train_loss=hist.history['loss']
plt.plot(train_loss,label='loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
#plt.legend(loc='lower right')
plt.show()

input_test=np.load('/Users/cin/Documents/pyfile/CNN_BiLSTM/input_test.npz')['arr_0']+31.5/(31.5+95.5)
#input_test=tf.data.Dataset.from_tensor_slices(np.load('/Users/cin/Documents/pyfile/CNN_BiLSTM/input_test.npz')['arr_0'])
print('ok')
output_test=np.load('/Users/cin/Documents/pyfile/CNN_BiLSTM/output_test.npz')['arr_0']+31.5/(31.5+95.5)
#output_test=tf.data.Dataset.from_tensor_slices(np.load('/Users/cin/Documents/pyfile/CNN_BiLSTM/output_test.npz')['arr_0'])
print('ok')

test_loss,test_acc=model.evaluate(input_test,output_test)
print(test_acc)
print(test_loss)

np.savetxt('stats_cnn_bilstm.csv',[train_loss,train_accur,test_loss,test_acc],delimiter=',')