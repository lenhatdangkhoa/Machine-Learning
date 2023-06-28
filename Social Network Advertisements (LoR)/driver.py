import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("Social_Network_Ads.csv")
x_train = df.iloc[:, 2].values
y_train = df.iloc[:, 3].values
colors = ["red" if purchase ==
          0 else "blue" for purchase in df["Purchased"]]
initial_w = 0.5
initial_b = 1
alpha = 0.009


def sigmoid(w, b, x):
    fw_b = w * x + b
    result = 1 / (1+np.exp(-fw_b))
    return result


def compute_cost(w, b, x, y):
    total_cost = 0
    for i in range(x.shape[0]):
        loss = (y[i] * np.log(sigmoid(w, b, x[i]))) + \
            ((1-y[i]) * np.log(1 - sigmoid(w, b, x[i])))
        total_cost += loss
    return -loss / x.shape[0]


def compute_gradient(w, b, x, y):
    dj_dw = 0
    dj_db = 0

    for i in range(x.shape[0]):
        dj_dw += (sigmoid(w, b, x[i]) - y[i]) * x[i]
        dj_db += sigmoid(w, b, x[i]) - y[i]
    dj_dw /= x.shape[0]
    dj_db /= x.shape[0]
    return dj_dw, dj_db


def gradient_descent(w, b, x, y, alpha):

    for i in range(10000):
        dj_dw, dj_db = compute_gradient(w, b, x, y)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        if (i % 10000 == 0):
            print(compute_cost(w, b, x, y))

    return w, b


"""
initial_b, initial_w = gradient_descent(
    initial_w, initial_b, x_train, y_train, alpha)

predicted = np.zeros(x_train.shape[0])


x_boundary = np.linspace(min(x_train), max(x_train), 100)
y_boundary = np.zeros(x_boundary.shape[0])
for i in range(x_boundary.shape[0]):
    y_boundary[i] = sigmoid(initial_w, initial_b, x_boundary[i])
"""

fig, ax = plt.subplots(figsize=(14, 6))

ax.scatter(x_train, y_train, color=colors)
black_patch = plt.Line2D([], [], marker='o', color='red',
                         linestyle='', markersize=8, label='No purchase')
green_patch = plt.Line2D([], [], marker='o', color='blue',
                         linestyle='', markersize=8, label='Purchase')
plt.legend(handles=[black_patch, green_patch],
           loc="upper left", bbox_to_anchor=(1.0, 1))
plt.show()
