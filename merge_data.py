import numpy as np
X_merge=[]
Y_merge=[]
print("Merging.. Please Wait")
for i in range(0,17):
	x=np.load('data/image_save'+str(i)+'.npy')
	y=np.load('data/keys_save'+str(i)+'.npy')
	X_merge.append(x)
	Y_merge.append(y)
print("Merge done! Now Saving")
X_merge=np.array(X_merge)
Y_merge=np.array(Y_merge)
X_merge=X_merge.reshape(17000,96,96)
Y_merge=Y_merge.reshape(17000,4)
np.save('images_input_new.npy',X_merge)
np.save('keys_output_new.npy',Y_merge)
print("Save done!")
X=np.load('images_input_new.npy')
Y=np.load('keys_output_new.npy')

print("Input shape: "+str(X.shape))
print("Output shape: "+str(Y.shape))
