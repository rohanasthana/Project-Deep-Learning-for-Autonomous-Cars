# Autonomous Driving Agent in different environmental conditions
This project was done  for the Deep Learning for Autonomous Cars module at the University of Ulm. It is a result of the collective effort by Ksenia Vinogradova and Rohan Asthana.

## Introduction
The main aim of this project was to adapt an autonomous driving agent to driving in a simulated OpenAI environment that it was not trained on. The agent was initially trained in default ‘good weather daytime’ conditions using a CNN network. The environment was then modified to introduce ‘bad weather’, such as rain and snow, as well as night, by adding respective animations and manipulating colours of the grass and the road. To allow an autonomous agent to still drive in this environment, a Pix2Pix network was used to convert the images from the environment into ‘good weather daytime’ images before giving them to the agent as input. This successfully improved the performance of the agent in the unfamiliar ‘bad weather’/night environment making it comparable to the familiar ‘good weather daytime’ environment.

![Different environments](https://github.com/rohanasthana/Project-Deep-Learning-for-Autonomous-Cars/blob/master/Pictures/fig1.jpg)

## Different environments

5 different conditions were used in this project:
- Rain
- Snow
- Night
- Rainy Night
- Snowy Night

The environments differed by the colour of the grass and the road, as well as by the presense of an overlaying animation (i.e. rain or snow).

![Different environments](https://github.com/rohanasthana/Project-Deep-Learning-for-Autonomous-Cars/blob/master/Pictures/fig2.jpg)

## Pix2pix

The pix2pix model was trained by feeding it pairs of good and bad weather images. For each environment, as separate model was trained and later retrieved depending on the weather condition specified. The original Pix2pix network architecture was slightly modified to decrease the unnecessary complexity: the number of encoder-decoder blocks was changed from 7 to 3, whereas the discriminator remained unchanged. 

![Different environments](https://github.com/rohanasthana/Project-Deep-Learning-for-Autonomous-Cars/blob/master/Pictures/fig3.jpg)

## The "Good weather" agent

The agent was a model derived from a CNN network consisting of 3 convolution blocks, each comprised of 1 x Conv layer, 1 x Max Pooling layer, 1 x BatchNorm layer.

## Results

The "good weather" agent combined with the Pix2pix model operating in modified environments showed performance not significantly different from its perfoamance in the default environment.

![Results](https://github.com/rohanasthana/Project-Deep-Learning-for-Autonomous-Cars/blob/master/Pictures/fig5.jpg)

## How to run

1. Install all the requirements by running the command

```
pip install -r requirements.txt
```

Note that the library Box2D requires you to-

a) Download SWIG

b) Putting SWIG directory in the Path variables of your system

c) Download and install Microsoft Visual C++ 14.0 or greater


2. Use the configuration.py file to change the weather and time of the day

3. Prepare data by running prepare_data.py. This requires you to play the game for a while to generate meaningful training data

4. Run merge_data.py to merge individual data files into one npy file

5. Run split_data.py to split the data into training, validation and test sets

6. Train the model by running train.py

7. Use the trained model to play the game by running predict.py




