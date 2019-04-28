# Forex using LSTM

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
op        = 'SpotBuy'

# Importing the training set
months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
filenames = []
frames = []
for month in months:
	filenames.append(datapath+currency1+currency2+"_"+year+month+'.csv')

for f in filenames:
	df = pd.read_csv(f)
	frames.append(df)
	
training_set = pd.concat(frames, ignore_index=True, sort=False)
training_len = len(training_set)
print(training_len)

# Data Visualization
plt.plot(training_set[op], color = 'red', label = year+' Real Price')
plt.xticks(range(0,training_set.shape[0],30),training_set['Date'].loc[::30],rotation=45)
plt.title('Forex (USD-NTD)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()


