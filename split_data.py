import numpy as np
from configuration import config
import os

time,weather=config()

length=int(len(os.listdir('data/'+str(time)+str(weather)))/2)-1 #-1 because merged npy is also present
frames=length*1000
trainlen=int(0.6*frames)
validlen=int(0.8*frames)
testlen=frames
print(trainlen,validlen,testlen)

x=np.load('data/'+str(time)+str(weather)+'/images_input_merged.npy')
y=np.load('data/'+str(time)+str(weather)+'/keys_output_merged.npy')

"""x0=np.load('images_input0.npy')
x1=np.load('images_input1.npy')
y0=np.load('keys_output0.npy')
y1=np.load('keys_output1.npy')

x=np.concatenate((x0,x1),axis=0)
y=np.concatenate((y0,y1),axis=0)"""

print(x.shape)
print(y.shape)

train_x=x[0:trainlen]
train_y=y[0:trainlen]
valid_x=x[trainlen:validlen]
valid_y=y[trainlen:validlen]
test_x=x[validlen:testlen]
test_y=y[validlen:testlen]

print('Train shape' + str(train_x.shape) + str(train_y.shape))
print('Valid shape' + str(valid_x.shape) + str(valid_y.shape))
print('Test shape' + str(test_x.shape) + str(test_y.shape))

os.makedirs('data/train-valid-test-split/'+str(time)+str(weather))

np.save('data/train-valid-test-split/'+str(time)+str(weather)+'/train_images.npy',train_x)
np.save('data/train-valid-test-split/'+str(time)+str(weather)+'/train_keys.npy',train_y)
np.save('data/train-valid-test-split/'+str(time)+str(weather)+'/valid_images.npy',valid_x)
np.save('data/train-valid-test-split/'+str(time)+str(weather)+'/valid_keys.npy',valid_y)
np.save('data/train-valid-test-split/'+str(time)+str(weather)+'/test_images.npy',test_x)
np.save('data/train-valid-test-split/'+str(time)+str(weather)+'/test_keys.npy',test_y)