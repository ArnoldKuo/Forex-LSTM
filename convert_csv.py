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
year      = '2019'
month     = '01'
currency1 = 'USD'
currency2 = 'NTD'
op        = 'Forward-30Days-Buy'

# Import forex dataset
months = ['01', '02', '03']
filenames = []
frames = []
for month in months:
	filenames.append(datapath+currency1+currency2+"_"+year+month+'.csv')

for f in filenames:
	df = pd.read_csv(f)
	dates=df['Date']
	newdates=[]
	for date in dates:
		newdate=str(date)
		newdates.append(newdate[:4]+'-'+newdate[4:6]+'-'+newdate[6:])
	df['Date']=newdates
	df.to_csv(f)

