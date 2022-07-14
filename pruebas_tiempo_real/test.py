import numpy as np
import pandas as pd

A = np.ones(15)
B = np.ones(15)*5
C = np.ones(15)*8

dict = {'faseA': A,
        'faseB': B,
        'faseC': C }

df = pd.DataFrame(dict)
df.to_excel('file.xlsx')

df1 = pd.read_excel('file.xlsx')
dict_1 = df1.to_dict()
print(dict_1['faseA'])

# np.savetxt('file.csv', [A, B, C], delimiter=',')

# data = np.loadtxt('file.csv')
# print(data[0,:])
# print(data[1,:])
# print(data[2,:])