import glob
import pandas as pd

#extension = 'csv'
#filenames = [i for i in glob.glob('*.{}'.format(extension))]

datapath  = 'data/'
year      = '2018'
month     = '01'
currency1 = 'USD'
currency2 = 'NTD'

months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
filenames = []
frames = []
for month in months:
	filenames.append(datapath+currency1+currency2+"_"+year+month+'.csv')

#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in filenames ])
#export to csv
combined_csv.to_csv( datapath+year+"_forex_combined.csv", index=False, encoding='utf-8-sig')