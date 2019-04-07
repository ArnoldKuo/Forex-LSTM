# Forex using Recurrent Neural Network

#-----------------------------------------------
# Part 1 - Data Preprocessing
#----------------------------------------------- 
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# column in dataset
cashbuy   = 1
spotbuy   = 2
cashsell  = 10
spotsell  = 11

# Configuration
epochs    = 1000
datapath  = 'data/'
year      = '2018'
month     = '01'
currency1 = 'USD'
currency2 = 'NTD'
op        = spotbuy

# Importing the training set
# read first month	
filename=currency1+currency2+"_"+year+month
dataset = pd.read_csv(datapath+filename+".csv")
training_set = dataset.iloc[:,op:(op+1)].values 

# read other months
months = ['02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
for month in months:
    filename=currency1+currency2+"_"+year+month
    dataset = pd.read_csv(datapath+filename+".csv")
    training_set1 = dataset.iloc[:,op:(op+1)].values
    training_set = np.append(training_set, training_set1, axis=0)

training_len = len(training_set)
print(training_len)

# Feature Scaling
# Will use Normalisation as the Scaling function.
# Default range for MinMaxScaler is 0 to 1, which is what we want. So no arguments in it.
 # Will fit the training set to it and get it scaled and replace the original set.
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)

# Getting the inputs and the outputs
# Restricting the input and output based on how LSTM functions.
X_train = training_set[0:(training_len-1)]
y_train = training_set[1:training_len]

# Reshaping - Adding time interval as a dimension for input.
X_train = np.reshape(X_train, ((training_len-1), 1, 1))

#-----------------------------------------------
# Part 2 - Building the RNN
#-----------------------------------------------
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Initialising the RNN
# Creating an object of Sequential class to create the RNN.
regressor = Sequential()

# Adding the input layer and the LSTM layer
# 4 memory units, sigmoid activation function and (None time interval with 1 attribute as input)
regressor.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 1)))

# Adding the output layer
# 1 neuron in the output layer for 1 dimensional output
regressor.add(Dense(units = 1))

# Compiling the RNN
# Compiling all the layers together.
# Loss helps in manipulation of weights in NN. 
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
# Number of epochs increased for better convergence.
regressor.fit(X_train, y_train, batch_size = 32, epochs = epochs)

#-------------------------------------------------------------
# Part 3 - Making the predictions and visualising the results
#-------------------------------------------------------------
# Getting the real forex price
year  ='2019'
month = '03'
filename=currency1+currency2+"_"+year+month
dataset = pd.read_csv(datapath+filename+".csv")
training_set = dataset.iloc[:,op:(op+1)].values
real_forex_price = training_set

# Getting the predicted forex price of 2019
inputs = real_forex_price
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (len(real_forex_price), 1, 1))
predicted_forex_price = regressor.predict(inputs)
predicted_forex_price = sc.inverse_transform(predicted_forex_price)

# Visualising the results
plt.plot(real_forex_price, color = 'red', label = 'Real Price')
plt.plot(predicted_forex_price, color = 'blue', label = 'Predicted Price')
plt.title(year+'/'+month+' Forex Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

#-----------------------------------------------
# Part 4 - Evaluating the RNN
#-----------------------------------------------
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_forex_price, predicted_forex_price))
print('rmse=',rmse)
