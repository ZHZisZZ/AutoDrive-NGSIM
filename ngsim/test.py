import pandas as pd

df = pd.read_csv('test.csv')

# print(df[df['Frame_ID']>40]['Frame_ID'].iloc[0])
print(df[df['Frame_ID']>40]['Frame_ID'].loc[0])