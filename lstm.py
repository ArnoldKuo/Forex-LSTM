# Forex using LSTM

#-----------------------------------------------
# Part 1 - Data Preprocessing
#----------------------------------------------- 
# Importing the libraries
import numpy as np
import pandas as pd

# column in dataset
cashbuy   = 1
spotbuy   = 2
forward_10Days_Buy = 3
forward_30Days_Buy = 4
forward_60Days_Buy = 5
forward_90Days_Buy = 6
forward_120Days_Buy = 7
forward_150Days_Buy = 8
forward_180Days_Buy = 9
cashsell  = 10
spotsell  = 11
forward_10Days_Sell = 12
forward_30Days_Sell = 13
forward_60Days_Sell = 14
forward_90Days_Sell = 15
forward_120Days_Sell = 16
forward_150Days_Sell = 17
forward_180Days_Sell = 18

# Configuration
epochs    = 5000
batch_size= 64
datapath  = 'data/'
currency1 = 'USD'
currency2 = 'NTD'
op        = spotbuy
predict_length=5

# Importing the training set
forex_prices = [[0]]

year='2018'
months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
for month in months:
	filename=currency1+currency2+"_"+year+month
	dataset = pd.read_csv(datapath+filename+".csv")
	forex_tmp = dataset.iloc[:,op:(op+1)].values
	forex_prices = np.append(forex_prices, forex_tmp, axis=0)
	
year='2019'
months = ['01', '02', '03']	
for month in months:
	filename=currency1+currency2+"_"+year+month
	dataset = pd.read_csv(datapath+filename+".csv")
	ue = pd.read_csv(r'C:\Users\akuo\Downloads\UNEMPLOY.csv')
	hpi = pd.read_csv(r'C:\Users\akuo\Downloads\USSTHPI.csv')
	fund = pd.read_csv(r'C:\Users\akuo\Downloads\FEDFUNDS.csv')
	forex_tmp = dataset.iloc[:,op:(op+1)].values
	forex_prices = np.append(forex_prices, forex_tmp, axis=0)

forex_prices = np.delete(forex_prices, [0], axis=0)

# Feature Scaling
# Will use Normalisation as the Scaling function.
# Default range for MinMaxScaler is 0 to 1, which is what we want. So no arguments in it.
 # Will fit the training set to it and get it scaled and replace the original set.
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
forex_prices = scaler.fit_transform(forex_prices)

forex_prices = np.delete(forex_prices, [0], axis=0)
# Getting the inputs and the outputs
# Restricting the input and output based on how LSTM functions.
train_size = int(len(forex_prices)*0.8)
test_size  = len(forex_prices)-train_size
train = forex_prices[:train_size]
test  = forex_prices[train_size:]
print(len(train), len(test))

# Convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], [], [], [],[]
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back),0]
		dataX.append(a)
		dataY.append(dataset[i+look_back, 0])
	return np.array(dataX), np.array(dataY)

# Reshap into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX  = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

#-----------------------------------------------
# Part 2 - Build Model
#-----------------------------------------------
# Importing the Keras libraries and packages
from subprocess import check_output
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.model_selection import train_test_split

import time
import matplotlib.pyplot as plt
from numpy import newaxis
import tensorflow as tf

# for GPU memory allocation
#gpu_options = tf.GPUOptions(allow_growth=True)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#tf.keras.backend.set_session(sess)

# Initialising the RNN
# Creating an object of Sequential class to create the RNN.
model = Sequential()

model.add(LSTM(units = 50, activation = 'sigmoid', input_shape = (None, 1), return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(units = 100, activation = 'sigmoid', input_shape = (None, 1), return_sequences=False))
model.add(Dropout(0.3))


model.add(Dense(units=1, activation='linear'))

start = time.time()

model.compile(loss='mse', optimizer='rmsprop')
print('compilation time: ', time.time()-start)

# Fitting the RNN to the Training set
# Number of epochs increased for better convergence.
model.fit(trainX, trainY, batch_size = batch_size, epochs = epochs, validation_split=0.05)

### Save model
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
	json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

#-------------------------------------------------------------
# Part 3 - Making the predictions and visualising the results
#-------------------------------------------------------------
def plot_results_multiple(predicted_data, true_data,length):
#	plt.plot(scaler.inverse_transform(true_data.reshape(-1, 1))[length:], color='blue', label='Real Prices')
#	plt.plot(scaler.inverse_transform(np.array(predicted_data).reshape(-1, 1))[length:], color='red', label='Predicted Prices')
	plt.plot(scaler.inverse_transform(true_data.reshape(-1, 1)), color='blue', label='Real Prices')
	plt.plot(scaler.inverse_transform(np.array(predicted_data).reshape(-1, 1)), color='red', label='Predicted Prices')
	plt.show()

# predict length consecutive values from a real one
def predict_sequences_multiple(model, firstValue,length):
	prediction_seqs = []
	curr_frame = firstValue
    
	for i in range(length): 
		predicted = []        
        
		print(model.predict(curr_frame[newaxis,:,:]))
		predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
        
		curr_frame = curr_frame[0:]
		curr_frame = np.insert(curr_frame[0:], i+1, predicted[-1], axis=0)
        
		prediction_seqs.append(predicted[-1])
        
	return prediction_seqs

predictions = predict_sequences_multiple(model, testX[0], predict_length)
print(len(train), len(testX))
print("predicted-price")
print(scaler.inverse_transform(np.array(predictions).reshape(-1, 1)))
print("read-price")
print(scaler.inverse_transform(testY[:predict_length].reshape(-1,1)))
plot_results_multiple(predictions, testY, predict_length)
