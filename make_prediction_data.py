import pandas as pd
import datetime
import numpy as np
import pickle

# read file
df = pd.read_csv('data.csv')
# get the latest day
now = datetime.datetime(1999,1,1)
for i in range(df.shape[0]):
    # if Date is later than now, then update now
    tmp = datetime.datetime.strptime(df.Date[i], '%m/%d/%Y %H:%M:%S %p')
    if tmp > now:
        now = tmp
thirty_days_ago = now - datetime.timedelta(days=30)


rows,cols = 14,12
# four offences:THEFT,BATTERY,CRIMINAL DAMAGE,ASSAULT
lis = [[[[0]*4 for i in range(30)] for _ in range(cols)] for _ in range(rows)]
# get max and min for cols X coordinates and Y coordinates
x_min = df.min(axis=0)['X Coordinate']
x_max = df.max(axis=0)['X Coordinate']
y_min = df.min(axis=0)['Y Coordinate']
y_max = df.max(axis=0)['Y Coordinate']

x_gap = (x_max - x_min) / cols
y_gap = (y_max - y_min) / rows

# fill the data into list
for i in range(df.shape[0]):
    # if Date is later than thirty_days_ago, then update lis
    tmp = datetime.datetime.strptime(df.Date[i], '%m/%d/%Y %H:%M:%S %p')
    if tmp > thirty_days_ago:
        # get the index of the list
        x_cord = df.iloc[i,15]
        y_cord = df.iloc[i,16]
        if np.isnan(x_cord) or np.isnan(y_cord):
            continue
        else:
            x_index = int((x_cord - x_min) / x_gap)
            y_index = int((y_cord - y_min) / y_gap)
        # tmp和now相差几天
        days = (now - tmp).days
        day_index = (30-1) - days
        # fill the lis based on the offence
        if df.iloc[i,5] == 'THEFT':
            lis[y_index][x_index][day_index][0] += 1
        elif df.iloc[i,5] == 'BATTERY':
            lis[y_index][x_index][day_index][1] += 1
        elif df.iloc[i,5] == 'CRIMINAL DAMAGE':
            lis[y_index][x_index][day_index][2] += 1
        elif df.iloc[i,5] == 'ASSAULT':
            lis[y_index][x_index][day_index][3] += 1
# save file
a = np.array(lis)
print('file shape:',a.shape)
with open('Datasets/CHI_crime/pre.pkl', 'wb') as file:
    pickle.dump(a, file)