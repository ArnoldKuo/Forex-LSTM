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
epochs    = 1000
datapath  = 'data/'
year      = '2018'
month     = '01'
currency1 = 'USD'
currency2 = 'NTD'
op        = 'Spot-Buy'

# Import forex dataset
months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
filenames = []
frames = []
for month in months:
	filenames.append(datapath+currency1+currency2+"_"+year+month+'.csv')

for f in filenames:
	df = pd.read_csv(f)
	print(len(df))
	frames.append(df)
	
forex_set = pd.concat(frames, ignore_index=True, sort=False)
forex_len = len(forex_set)
print(forex_len)

# Data Visualization
plt.plot(forex_set[op], color = 'red', label = year+' Real Price')
plt.xticks(range(0,forex_set.shape[0],30),forex_set['Date'].loc[::30],rotation=45)
plt.title('Forex (USD-NTD)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()


