### read brent-daily_csv & forex
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Forex Configuration
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

tradingdays=[]
for f in filenames:
	df = pd.read_csv(f)
	print(len(df))
	tradingdays.append(len(df))
	frames.append(df)
	
print(tradingdays)
forex_set = pd.concat(frames, ignore_index=True, sort=False)
forex_len = len(forex_set)
print(forex_len)

# Import FEDFUNDS dataset
filename = 'FEDFUNDS.csv'
fedfunds_set = pd.read_csv(datapath+filename)
fedfunds_set = fedfunds_set.drop(fedfunds_set.index[774:], axis=0)  # drop after 2018
fedfunds_set = fedfunds_set.drop(fedfunds_set.index[0:762], axis=0) # drop before 2018
# reset index & remove columns='index'
fedfunds_set = fedfunds_set.reset_index()
fedfunds_set = fedfunds_set.drop(columns=['index'])
print(fedfunds_set)

index=0
j=0
fedfunds = pd.DataFrame(columns=['DATE', 'FEDFUNDS'])
for days in tradingdays:
	for i in range(days):
		index+=1
		fedfunds.loc[index] = fedfunds_set.loc[j]
	j+=1
print(fedfunds)

### Data Visualization
op='Cash-Buy'
plt.plot(forex_set[op], color = 'b', label = op)
plt.plot(fedfunds['FEDFUNDS'], color = 'brown', label = 'FEDFUNDS')
plt.xticks(range(0,forex_set.shape[0],30),forex_set['Date'].loc[::30],rotation=45)
plt.title(year+' Forex and Brent-Daily')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
