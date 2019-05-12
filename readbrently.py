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

### Import Brently dataset
filename = 'brent-daily'+year+'.csv'
oil_set = pd.read_csv(datapath+filename,index_col=False)

### Data Visualization
# Brent-Daily & Forex
op='Cash-Buy'
plt.plot(forex_set[op], color = 'b', label = year+' '+op)
plt.plot(oil_set['Price'], color = 'brown', label = year+' brent-daily')
plt.xticks(range(0,oil_set.shape[0],30),oil_set['Date'].loc[::30],rotation=45)
plt.title('Forex (USD-NTD)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

#
datelist=[]
for date in forex_set['Date']:
	datelist.append(date)

holidays_forex=[]
for date in oil_set['Date']:
	if date not in datelist:
		holidays_forex.append(date)
print(holidays_forex)

#
datelist=[]
for date in oil_set['Date']:
	datelist.append(date)

holidays_brently=[]
for date in forex_set['Date']:
	if date not in datelist:
		holidays_brently.append(date)
print(holidays_brently)
