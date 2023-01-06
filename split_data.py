import numpy as np
x0=np.load('images_input0.npy')
x1=np.load('images_input1.npy')
y0=np.load('keys_output0.npy')
y1=np.load('keys_output1.npy')

x=np.concatenate((x0,x1),axis=0)
y=np.concatenate((y0,y1),axis=0)

print(x.shape)
print(y.shape)

train_x=x[0:60000]
train_y=y[0:60000]
valid_x=x[60000:80000]
valid_y=y[60000:80000]
test_x=x[80000:100000]
test_y=y[80000:100000]

print('Train shape' + str(train_x.shape) + str(train_y.shape))
print('Valid shape' + str(valid_x.shape) + str(valid_y.shape))
print('Test shape' + str(test_x.shape) + str(test_y.shape))

np.save('train-valid-test-split/train_images.npy',train_x)
np.save('train-valid-test-split/train_keys.npy',train_y)
np.save('train-valid-test-split/valid_images.npy',valid_x)
np.save('train-valid-test-split/valid_keys.npy',valid_y)
np.save('train-valid-test-split/test_images.npy',test_x)
np.save('train-valid-test-split/test_keys.npy',test_y)