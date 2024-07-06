import pandas as pd
import os
import time

temp = list()
for file in os.listdir("./"):
    if file.endswith('.csv') and file != 'agg_data.csv':
        df = pd.read_csv(file, skiprows=1)
        df.drop(df.columns[0], axis=1, inplace=True)
        print(f"Now Data Date: {df['StnPresMaxTime'][0].split()[0]}", end='\r')
        time.sleep(0.5)
        temp.append(df)

df = pd.concat(temp)
print()
print(f"Total rows: {len(df)}, Total columns: {len(df.columns)}")
df.to_csv('agg_data.csv', index=False)