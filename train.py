import numpy as np
import keras
from keras.layers import TimeDistributed,LSTM,Dense,Input,SpatialDropout1D,Flatten
from keras import Model
from keras.layers import MaxPooling1D
from keras.layers import Conv1D
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from configuration import config

time,weather=config()

X_train=np.load('data/train-valid-test-split/'+str(time)+str(weather)+'/train_images.npy')
Y_train=np.load('data/train-valid-test-split/'+str(time)+str(weather)+'/train_keys.npy')
X_valid=np.load('data/train-valid-test-split/'+str(time)+str(weather)+'/valid_images.npy')
Y_valid=np.load('data/train-valid-test-split/'+str(time)+str(weather)+'/valid_keys.npy')

print("Data Loaded !")
print("Now creating model!")


def create_model():
	model_name='timedistributed_cnn_lstm'
	inp = Input(shape=((96,96,1)))
	x = TimeDistributed(Conv1D(filters=50, kernel_size=4, padding='same', activation='relu'))(inp)
	x = TimeDistributed(MaxPooling1D(pool_size=2))(x)
	x = TimeDistributed(SpatialDropout1D(0.2))(x)
	x = TimeDistributed(BatchNormalization())(x)
	x = TimeDistributed(Flatten())(x)
	x = LSTM(200, dropout=0.2, recurrent_dropout=0.2)(x)
	out = Dense(4,activation='softmax')(x)
	model = Model(inputs=inp, outputs=[out])
	opt = Adam(lr=0.01, decay=1e-3 / 200)
	model.compile(loss = 'categorical_crossentropy', optimizer=opt,metrics = ['accuracy'])
	return model,model_name

X_train = np.expand_dims(np.stack(X_train), axis = 3)
X_valid = np.expand_dims(np.stack(X_valid), axis = 3)
model,name=create_model()

print(X_train.shape)
print(Y_train.shape)
print(X_valid.shape)
print(Y_valid.shape)

history=model.fit(X_train,Y_train,validation_data=(X_valid,Y_valid),epochs=2)
model.save('models/'+str(time)+str(weather)+'/'+str(name)+'.h5')
