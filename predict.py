import numpy as np
import keras
from game_weather import CarRacing
import cv2
from pyglet.window import key
from pynput.keyboard import Key, Controller
from pywinauto.keyboard import SendKeys
from skimage.transform import rescale, resize, downscale_local_mean

model=keras.models.load_model('simple_cnnv_200 epochs.h5')

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
        while True:
            steps += 1
            isopen = env.render()
            #image capture
            img=env.render(mode='rgb_array')
            #img = resize(img, (96,96,3),
             #          anti_aliasing=True)
            #print(img.shape)
            img=prepare_image(img)
            img=np.array(img)
            #print(np.shape(img))
            img = np.reshape(img,(1,96,96))
            y_predict=model.predict(img)
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
                #s, r, done, info = env.step(a)
            
            s, r, done, info = env.step(a)

            total_reward += r
            print("TOTAL REWARD"+str(total_reward))
            rew.append(total_reward)
            #if steps % 200 == 0 or done:
                #print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                #print("step {} total_reward {:+0.2f}".format(steps, total_reward))


            # for ind, i in enumerate(y_predict[0]):
            #     if(i==1):
            #         if(ind==0):
            #             SendKeys('{LEFT}')
            #         #pyautogui.press('left')
            #         if(ind==1):
            #             SendKeys('{RIGHT}')
            #         if(ind==2):
            #             SendKeys('{UP}')
            #         if(ind==3):
            #             SendKeys('{DOWN}')



            #print(y_predict)
            #X_images.append(img)
            #keys capture
            #frame_keys = keys_from_action(a)
            #Y_keys.append(frame_keys)
            #if len(X_images) % 1000 == 0:
             #   print("Total length: " + str(len(X_images)))
            if done or restart or isopen == False:
                break
            #if len(X_images) == 1000: #was 1000
                #save images
             #   filename='image_save'+str(f+48)+'.npy'
              #  np.save(filename,X_images)

               # print('SAVED'+ str(filename))
                #X_images=[]

                #save keys
                #filename = 'keys_save' + str(f+48) + '.npy'
                #np.save(filename, Y_keys)

                #print('SAVED' + str(filename))
                #Y_keys = []
                #f += 1

    env.close()
    print(max(rew))

    #X_images=np.array(X_images)
    #print(X_images.shape)


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

