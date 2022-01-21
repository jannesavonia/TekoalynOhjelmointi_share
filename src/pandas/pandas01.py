import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#algebra.csv is available on shared materials/data
data=pd.read_csv('algebra.csv', skiprows=1)

print('\n--- Lahtodata ------------------\n')
print(data)

print('\n--- Lahtodatan statistiikka ----\n')
print(data.describe())

print('\n--- Kiinnostavat sarakkeet -----\n')

tehtava_tentti=data[['Tehtavapisteet', 'Tenttipisteet']]
print(tehtava_tentti)

print('\n--- Poistetaan NaN-arvot -------\n')
tehtava_tentti=tehtava_tentti.dropna()

print(tehtava_tentti)

print('\n--- Sovitetaan suora pisteisiin \n')

X=tehtava_tentti['Tehtavapisteet'].values.reshape(-1, 1)
Y=tehtava_tentti['Tenttipisteet'].values.reshape(-1, 1)

regr = LinearRegression()
regr.fit(X, Y)
Y_pred = regr.predict(X) 

print('\n--- Lineaarinen malli ----------\n')

print('Tenttipisteet = '+str(regr.coef_[0][0])+' * '+ 'Tehtavapisteet + '+str(regr.intercept_[0]))

plt.plot(X, Y, '*')
plt.plot(X, Y_pred, '--')
plt.show()
