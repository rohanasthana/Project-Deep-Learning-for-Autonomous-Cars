import numpy as np
from configuration import config
import os

X_merge=[]
Y_merge=[]

time,weather=config()



print("Merging.. Please Wait")

length=int(len(os.listdir('data/'+str(time)+str(weather)))/2)

for i in range(length):
	x=np.load('data/'+str(time)+str(weather)+'/image_save'+str(i)+'.npy')
	y=np.load('data/'+str(time)+str(weather)+'/keys_save'+str(i)+'.npy')
	X_merge.append(x)
	Y_merge.append(y)
print("Merge done! Now Saving")
X_merge=np.array(X_merge)
Y_merge=np.array(Y_merge)
X_merge=X_merge.reshape(length*1000,96,96)
Y_merge=Y_merge.reshape(length*1000,4)
np.save('data/'+str(time)+str(weather)+'/images_input_merged.npy',X_merge)
np.save('data/'+str(time)+str(weather)+'/keys_output_merged.npy',Y_merge)
print("Save done!")
X=np.load('data/'+str(time)+str(weather)+'/images_input_merged.npy')
Y=np.load('data/'+str(time)+str(weather)+'/keys_output_merged.npy')

print("Input shape: "+str(X.shape))
print("Output shape: "+str(Y.shape))
