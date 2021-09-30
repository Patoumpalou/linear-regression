
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
# from test import data, theta, plt, tmpX, np, history, iterations

data = pd.read_csv('data.csv')

try:
    with open('theta.pickle', 'rb') as f:
        theta = pickle.load(f)
except: 
    theta = np.zeros((2,1))

# print(theta)
# exit()

command = input("km: ")
if command == "EXIT":
    print("Byebye")
    exit()
# try:
km = float(command)
if (km < 0):
    km = 0
X = np.array(data['km'])
X = np.reshape(X, (X.shape[0], 1) )
X = np.hstack((X, np.ones(X.shape)))
theta[1] = theta[1] * 10000

cost = km * theta[0] + theta[1] 
# cost = train.model(X, theta)

# print(cost)
# exit()
if (cost < 0):
    cost = 0
print("esimate price: ", cost) 

# except ValueError:
    # print("Error input")

# print(X.dot(theta))
# exit()
# theta[1] = theta[1] / 10000
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.scatter(data['km'], data['price'])
plt.plot(X[:,0], X.dot(theta), c='red')

# plt.subplot(2, 2, 2)
# plt.plot(r1ange(iterations), history)

plt.show()
