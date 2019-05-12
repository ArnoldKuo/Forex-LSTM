# Forex using LSTM

#-----------------------------------------------
# Part 1 - Data Preprocessing
#----------------------------------------------- 
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Configuration
datapath  = 'data/'
year      = '2018'
month     = '01'
currency1 = 'USD'
currency2 = 'NTD'
trade='-Sell' # '-Buy' or '-Sell'
ops = ['Cash','Spot','Forward-10Days','Forward-30Days','Forward-60Days','Forward-90Days','Forward-120Days','Forward-150Days','Forward-180Days']
colr= ['b','g','r','c','m','y','k','plum','aqua']


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

### Data Visualization
# Cash-Buy vs Cash-Sell
op='Cash-Buy'
plt.plot(forex_set[op], color = 'b', label = year+' '+op)
op='Cash-Sell'
plt.plot(forex_set[op], color = 'r', label = year+' '+op)
plt.xticks(range(0,forex_set.shape[0],30),forex_set['Date'].loc[::30],rotation=45)
plt.title('Forex (USD-NTD)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# all Buys
i=0
for op in ops:
	op = op+'-Buy'
	plt.plot(forex_set[op], color = colr[i], label = year+' '+op)
	i+=1

plt.xticks(range(0,forex_set.shape[0],30),forex_set['Date'].loc[::30],rotation=45)
plt.title('Forex (USD-NTD)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# all Sells
i=0
for op in ops:
	op = op+'-Sell'
	plt.plot(forex_set[op], color = colr[i], label = year+' '+op)
	i+=1

plt.xticks(range(0,forex_set.shape[0],30),forex_set['Date'].loc[::30],rotation=45)
plt.title('Forex (USD-NTD)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
