from typing import final
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle


# data est un dataframe
data = pd.read_csv('data.csv')
# nombre d'entrées
m = len(data)

# km = la feature
X = np.array(data['km'])
# y = target 
Y = np.array(data['price'])
#on reduit les valeurs car trop grandes, on oubliera pas de les augmenter a la fin
X = X / 10000
Y = Y / 10000
theta = np.zeros((2,1))
#reshape pour avoir (24,1) plutot que (24,)
X = np.reshape(X, (X.shape[0], 1) )
Y = np.reshape(Y, (Y.shape[0], 1) )
#ajout d'une colonne de 1 pour pouvoir la multiplier avec theta 
tmpX = np.hstack((X, np.ones(X.shape)))

def get_model(tmpX, theta):
    return tmpX.dot(theta)

def get_grad(theta, m, Y, tmpX):
    return 1/m * tmpX.T.dot(get_model(tmpX, theta) - Y)

def get_cost(m, model, Y):
    
    return 1/(2*m) * np.sum((model - Y)**2)
##cost = 1/(2*m) * np.sum((model - Y)**2)
# matrice.T = transposition 
ratio = 0.01
iterations = 3000
history = np.zeros(iterations)
for i in range(0, iterations):
    theta = theta - (ratio * get_grad(theta, m, Y, tmpX))
    history[i] = get_cost(m, get_model(tmpX, theta), Y)
    

X = X * 10000
tmpX = tmpX * 10000
Y = Y * 10000

finalModel = get_model(tmpX, theta)
precision = (finalModel - Y)**2
precision = np.sqrt(precision)
precision = precision.sum()
precision = precision/m

print(precision)


with open('theta.pickle', 'wb') as f:
    pickle.dump(theta, f)

f = open("theta.txt", "w")
str = np.array2string(theta)

# plt.figure(figsize=(12, 8))
# plt.subplot(2, 2, 1)
# plt.scatter(data['km'], data['price'])
# plt.plot(tmpX[:,0], tmpX.dot(theta), c='red')

# plt.subplot(2, 2, 2)
# plt.plot(range(iterations), history)

# plt.show()
