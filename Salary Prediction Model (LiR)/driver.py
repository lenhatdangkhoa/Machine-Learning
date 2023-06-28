import matplotlib.pyplot as plt
import math
import numpy as np
import csv
import pandas as pd

df = pd.read_csv("Salary_data.csv")

x_train = df.iloc[:, 0].values
y_train = df.iloc[:, 1].values
predicted = np.zeros(x_train.shape[0])
cost = []
w_cost = []
b_cost = []
initial_w = 100
initial_b = 100
alpha = 0.003


def compute_cost(x, y, w, b):
    total_cost = 0
    for i in range(x_train.shape[0]):
        f_wb = w*x[i] + b
        f_wb = (f_wb - y[i])**2
        total_cost += f_wb
    total_cost /= (2*x_train.shape[0])
    return total_cost


def compute_gradient(x, y, w, b):
    dj_dw = 0
    dj_db = 0
    for i in range(x.shape[0]):
        dj_dw += ((w * x[i] + b) - y[i]) * x[i]
        dj_db += ((w*x[i] + b) - y[i])
    dj_dw /= x.shape[0]
    dj_db /= x.shape[0]
    return dj_dw, dj_db


def gradient_descent(x, y, w, b, alpha):
    cost_arr = []
    for i in range(x.shape[0]):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        w_cost.append(w)
        b_cost.append(b)
        cost = compute_cost(x, y, w, b)
        cost_arr.append(cost)
        print(f"w = {w} ; b = {b} ; cost = {cost}")
    return w, b, cost_arr


initial_w, initial_b, cost = gradient_descent(
    x_train, y_train, initial_w, initial_b, alpha)

for i in range(x_train.shape[0]):
    predicted[i] = initial_w * x_train[i] + initial_b


plt.scatter(x_train, predicted, c="b")
plt.scatter(x_train, y_train, marker="x", c="r")
plt.ylabel("Salary in $")
plt.xlabel("Years of Experience")
plt.title("Salary Prediction Model")
plt.show()

plt.figure()
ax = plt.axes(projection="3d")
ax.scatter(w_cost, b_cost, cost)
plt.show()
