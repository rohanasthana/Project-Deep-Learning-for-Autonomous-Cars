import numpy as np
from keras.models import load_model
from game_weather import CarRacing
import cv2
from pyglet.window import key
import matplotlib.pyplot as plt
from PIL import Image
import scipy.misc
import png
import matplotlib
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.color import rgb2gray
import os
from configuration import config

# from pynput.keyboard import Key, Controller
# from pywinauto.keyboard import SendKeys
time,weather=config()
name='timedistributed_cnn_lstm'

model_keys = load_model('models/simple_cnnv_200 epochs.h5')

if not os.path.exists('models/'+str(time)+str(weather)):
    os.makedirs('models/'+str(time)+str(weather))
#model_weather = load_model('models/'+str(time)+str(weather)+'/'+str(name)+'.h5')
model_weather = load_model('models/model_snow.h5')
print(model_weather)

a = np.array([0.0, 0.0, 0.0])


def key_press(k, mod):
    global restart
    #to record keys pressed
    if k == 0xFF0D:
        restart = True
    if k == key.LEFT:
        a[0] = -1.0
    if k == key.RIGHT:
        a[0] = +1.0
    if k == key.UP:
        a[1] = +1
    if k == key.DOWN:
        a[2] = +1  # set 1.0 for wheels to block to zero rotation


def key_release(k, mod):
    if k == key.LEFT and a[0] == -1.0:
        a[0] = 0
    if k == key.RIGHT and a[0] == +1.0:
        a[0] = 0
    if k == key.UP:
        a[1] = 0
    if k == key.DOWN:
        a[2] = 0

def start_racing():
    #keyboard = Controller()
    f=0
    X_images=[]
    Y_keys = []
    rew=[]

    env = CarRacing()
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    record_video = False
    if record_video:
        from gym.wrappers.monitor import Monitor
      

        env = Monitor(env, "/tmp/video-test", force=True)
        

    isopen = True
    while isopen:

        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        f=0
        while True:
            f+=1
            steps += 1
            isopen = env.render()
            #image capture
            img=env.render(mode='rgb_array')
            
            img=np.array(img)
            if(f%10 ==0):

                matplotlib.image.imsave('original_%s_%s_%s.png'%(weather,time,f), img )
            #print(np.shape(img))
            #plt.imshow(img)
            #img_1 = np.reshape(img,(1,96,96,3))
            img = (img - 127.5) / 127.5
            img = resize(img, (96,96,3),
                       anti_aliasing=True)

            #img=rgb2gray(img) #GRAYSCALE conversion
            img=np.expand_dims(img,axis=0)

            img_gw = model_weather.predict(img)
            #print(img_gw)



            #print(img_gw)
            img_gw=np.array(img_gw)
            img_gw = (img_gw + 1) / 2.0
            print(img_gw.shape)
            img_gw = np.reshape(img_gw,(96,96,3))
            #print(img_gw)
            #if(f%10==0):
              #  scipy.misc.toimage(image_array, cmin=0.0, cmax=...).save('outfile.jpg')
                #im.save("image"+str(f)+".jpg")
                #plt.imshow(img_gw)
            #if(f%10 ==0):

            #   matplotlib.image.imsave('%s_%s_%s.png'%(weather,time,f), img_gw)
            img_gw=img_gw*255
            img_gw=prepare_image(img_gw)



            img_gw = np.reshape(img_gw,(1,96,96))
            y_predict=model_keys.predict(img_gw)
            for ind,i in enumerate(y_predict[0]):
                 if(i>0.3):
                     y_predict[0][ind]=1
                 else:
                     y_predict[0][ind]=0
            a=keys_to_action(y_predict)
            #print(y_predict)
            speed= CarRacing.render_indicators(env,96,96)
            if(speed<10):
                a=np.array(a)
                a = np.concatenate(([a[0]],[1,0]),axis=0)
                s, r, done, info = env.step(a)
            else:
                s, r, done, info = env.step(a)

            total_reward += r
            rew.append(total_reward)
            print("TOTAL REWARD IS"+str(total_reward))

            if done or restart or isopen == False:
                break

    env.close()
    print(max(rew))



def prepare_image(env):
    img=cv2.cvtColor(env,cv2.COLOR_RGB2GRAY)
    img=np.array(img)
    #print(img.shape)
    return img

def keys_from_action(a):
    frame_keys = [0, 0, 0, 0] #left, right, up, down
    if a[0] == -1.0: #left
        frame_keys[0] = 1
    if a[0] == +1.0: #right
        frame_keys[1] = 1
    if a[1] == +0.5: #up
        frame_keys[2] = 1
    if a[2] == +0.5: #down
        frame_keys[3] = 1
    return frame_keys

def keys_to_action(keys):
    x=[0,0,0]
    for ind, i in enumerate(keys[0]):
        if(i==1):
            if(ind==0):
                x[0]=-1
            if(ind==1):
                x[0]=1
            if(ind==2):
                x[1]=1
            if(ind==3):
                x[2]=1
    return x
start_racing()

