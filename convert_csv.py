# convert forex.csv file

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

datapath  = 'data/'
in_filename = 'ExchangeRate@201905101600.csv'
out_filename = 'USDNTD_201905.csv'

df = pd.read_csv(datapath+in_filename, index_col=False)
print(df)

# df drop columns
print('----- drop columns ----------')
df = df.drop(columns = ['Currency','Rate','Rate.1'])
print(df)

# df modify item
changes_buy = ['Cash', 'Spot','Forward-10Days','Forward-30Days','Forward-60Days','Forward-90Days','Forward-120Days','Forward-150Days','Forward-180Days']

print('----- rename columns ----------')
df = df.rename(columns = {'Data Date':'Date'})

for change in changes_buy:
	df = df.rename(columns={change     : change+'-Buy'})
	df = df.rename(columns={change+'.1': change+'-Sell'})
print(df)
	
print('----- df sorting by date -------')
df = df.sort_values(by=['Date'],ascending=True)
print(df)

newdate=[]
for date in df['Date']:
	date = str(date)
	newdate.append(date[:4]+'-'+date[4:6]+'-'+date[6:])

print(newdate)
df['Date']=newdate
print(df)

# write csv
df.to_csv(datapath+out_filename, index=False)
