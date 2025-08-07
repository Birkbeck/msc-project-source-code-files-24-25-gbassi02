import requests
import pandas as pd
import json
import matplotlib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from keras.layers import Dense
from keras import Input
from keras import Model
from keras.models import Sequential


dataset = requests.get("https://raw.githubusercontent.com/Experience-Monks/nice-color-palettes/master/1000.json") #http request to get data from api
palettes = json.loads(dataset.text) #converts it into list of palettes
palettes_select = [i for i in palettes if len(i) == 5] #selects only palettes with length = 5 colours

def hex_to_rgb(hex_value): #will convert hex values to RGB and normalise the new values between 0-1 to be used in NN
  hex_value = hex_value.lstrip("#") #removes '#' from hex values
  if len(hex_value) != 6: # Check if the hex_value has 6 characters
    return None
  try: #deals with potential errors during conversion
    return tuple(round(int(hex_value[i*2:i*2+2], 16)/255, 2) for i in range(3)) # Creates a tuple with each pair of hex converted to RGB and normalized, values rounded to 2 decimals
  except ValueError: #deals with cases wiht invalid hex characters
    return None

palettes_rgb = []

for palette in palettes_select:
  rgb_value = []
  for colour in palette:
    rgb = hex_to_rgb(colour)
    if rgb is not None: # Check if rgb is not None before appending
      rgb_value.append(rgb)
  palettes_rgb.append(rgb_value)
x = [] # initialising lists of inputs and outputs
y = []

for palette in palettes_rgb: #makes all unique pairs
  for i in range(5):
    for j in range(i+1, 5):
      input_colours = palette [i] + palette[j] #puts values in a flat list to use in the models

      output_colours = []
      for k in range(5): #makes a list with the values that were not used above to be the output
        if k != i and k != j:
          output_colours.extend(palette[k])
      x.append(input_colours)
      y.append(output_colours)

#flattening values into a vector shape to use in the models
x = np.array(x)
y = np.array (y)


# set-up of random forest model

seed = 100
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
 random_state=seed)
model = RandomForestRegressor(n_estimators=100, random_state=seed)
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
mse = mean_squared_error (y_test, y_predict)
print(mse)

def palette_visualization(input_colours, output_colours):
  full_palette = input_colours + output_colours

  fig, ax = plt.subplots(figsize = (10, 2))
 
  for i in range(len(full_palette)):
    ax.add_patch(plt.Rectangle((i,0), 1, 1, color=full_palette[i]))
  ax.set_xlim(0, len(full_palette))
  ax.set_ylim(0, 1)
  ax.set_xticks([])
  ax.set_yticks([])
  plt.show()

#creating model using the functional API
nn_input_layer = Input (shape=(6,)) #input layer takes 2 RGB colours x 3 numbers each
#building hidden layers
nn_hidden_layer_1 = Dense(64, activation='relu')(nn_input_layer)
nn_hidden_layer_2 = Dense(32, activation='relu')(nn_hidden_layer_1)
nn_output_layer = Dense(9, activation = 'sigmoid') (nn_hidden_layer_2)

nn_model = Model (inputs = nn_input_layer, outputs = nn_output_layer) #creates model
nn_model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse', 'mae']) #compiling model to prepare for training. Used 2 metrics fro comparison

