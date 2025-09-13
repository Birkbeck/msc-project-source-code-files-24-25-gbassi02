import requests
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error #used during development
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

def hex_to_rgb(hex_value): #will convert hex values to RGB and normalise the new values between 0-1 to be used in NN
  hex_value = hex_value.lstrip("#") #removes '#' from hex values
  if len(hex_value) != 6: # Check if the hex_value has 6 characters
    return None
  try: #deals with potential errors during conversion
    return tuple(round(int(hex_value[i*2:i*2+2], 16)/255, 2) for i in range(3)) # Creates a tuple with each pair of hex converted to RGB and normalised, values rounded to 2 decimals
  except ValueError: #deals with cases with invalid hex characters
    return None

def palette_visualization(input_colours, output_colours):
  full_palette = list(input_colours) + list(output_colours)

  fig, ax = plt.subplots(figsize = (10, 2))
  for i in range(len(full_palette)):
    ax.add_patch(plt.Rectangle((i,0), 1, 1, color=full_palette[i]))
  ax.set_xlim(0, len(full_palette))
  ax.set_ylim(0, 1)
  ax.set_xticks([])
  ax.set_yticks([])
  plt.show()

dataset = requests.get("https://raw.githubusercontent.com/Experience-Monks/nice-color-palettes/master/1000.json") #http request to get data from api
palettes = json.loads(dataset.text) #converts it into list of palettes
palettes_select = [i for i in palettes if len(i) == 5] #selects only palettes with length = 5 colours

palettes_rgb = []
for palette in palettes_select:
  rgb_value = []
  for colour in palette:
    rgb = hex_to_rgb(colour) #implementing function
    if rgb is not None: # Check if rgb is not None before appending
      rgb_value.append(rgb)
  palettes_rgb.append(rgb_value)

x = [] # initialising lists of inputs and outputs
y = []

dup = set() #to remove duplicates

for palette in palettes_rgb: #makes all unique pairs
  for i in range(5):
    for j in range(i+1, 5):
      input_colours = sorted([palette [i] + palette[j]]) #puts values in a flat list to use in the models
      output_colours = sorted([palette[k] for k in range(5) if k not in (i, j)])
      pairs = (tuple(input_colours), tuple(output_colours))
      if pairs not in dup:
        dup.add(pairs)
      #for k in range(5): #makes a list with the values that were not used above to be the output
       # if k != i and k != j:
        #  output_colours.extend(palette[k])
      x.append([n for colour in input_colours for n in colour])
      y.append([n for colour in output_colours for n in colour])

#flattening values into a vector shape to use in the models
x = np.array(x)
y = np.array (y)

#print(len(palettes_rgb)) #used on developing phase
#print(len(x))

#set up of random forrest model
seed = 100
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
rf_model = RandomForestRegressor(n_estimators=100, random_state=seed)
rf_model.fit(x_train, y_train)

#y_predict_rf = model.predict(x_test) #used on developing phase
#mse_rf = mean_squared_error (y_test, y_predict_rf)
#print(mse_rf)

#creating model using the functional API
nn_input_layer = Input (shape=(6,)) #input layer takes 2 RGB colours x 3 numbers each

#building hidden layers
nn_hidden_layer_1 = Dense(64, activation='relu')(nn_input_layer)
nn_hidden_layer_2 = Dense(32, activation='relu')(nn_hidden_layer_1)
nn_output_layer = Dense(9, activation = 'sigmoid') (nn_hidden_layer_2)

nn_model = Model (inputs = nn_input_layer, outputs = nn_output_layer) #creates model
nn_model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
history = nn_model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.1, callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)], verbose=1)

#plt.plot(history.history['loss'], label='Training Loss') #used on developing phase
#plt.plot(history.history['val_loss'], label='Validation Loss')
#plt.xlabel('Epoch')
#plt.ylabel('Loss')
#plt.title('Training vs. Validation Loss')
#plt.legend()
#plt.show()

#test_loss_nn = nn_model.evaluate(x_test, y_test) #used on developing phase
#print(test_loss_nn)

#y_predict_nn = nn_model.predict(x_test) #used on developing phase
#mse_nn = mean_squared_error (y_test, y_predict_nn)
#print(mse_nn)

def main ():
  while True:
    colour_input = []
    for i in range (2):
      while True:
        user_colour_input = input(f"Enter selected colour number {i+1} using RGB values (0-255) separated by comma, or type 'quit' to exit: " )
        if user_colour_input.lower() == 'quit':
          print ("Goodbye")
          exit()
        try:
          r, g ,b = map (int, user_colour_input.split(','))
          if not all (0 <= val <= 255 for val in (r,g,b)):
            print("Values must be between 0 and 255")
            continue
          colour_input.extend([r/255, g/255, b/255]) #to normalize values between 0-1
          break
        except ValueError:
          print ("Invalid input. Please enter RGB values (0-255) separated by comma, or type 'quit' to exit: ")
    user_input = np.array(colour_input).reshape (1, -1) #puts the values together
    input_colours_list = [tuple(user_input[0][:3]), tuple(user_input[0][3:6])]

    rf_prediction = rf_model.predict(user_input)[0]
    rf_output = ((min(max(rf_prediction[0],0),1), min(max(rf_prediction[1],0),1), min(max(rf_prediction[2],0),1)),
        (min(max(rf_prediction[3],0),1), min(max(rf_prediction[4],0),1), min(max(rf_prediction[5],0),1)),
        (min(max(rf_prediction[6],0),1), min(max(rf_prediction[7],0),1), min(max(rf_prediction[8],0),1)))
    print("Random Forest palette:", rf_output)
    palette_visualization(input_colours_list, rf_output)

    nn_prediction = nn_model.predict(user_input)[0]
    nn_output = ((min(max(nn_prediction[0], 0), 1), min(max(nn_prediction[1], 0), 1), min(max(nn_prediction[2], 0), 1)),
        (min(max(nn_prediction[3], 0), 1), min(max(nn_prediction[4], 0), 1), min(max(nn_prediction[5], 0), 1)),
        (min(max(nn_prediction[6], 0), 1), min(max(nn_prediction[7], 0), 1), min(max(nn_prediction[8], 0), 1)))
    print("Neural Network palette:", nn_output)
    palette_visualization(input_colours_list, nn_output)

if __name__ == "__main__":
    main()
