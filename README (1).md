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