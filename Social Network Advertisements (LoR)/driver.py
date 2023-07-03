import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("Social_Network_Ads.csv")
x_train = df[["Age", "EstimatedSalary"]].values
y_train = df.iloc[:, 4].values
x_train[:, 1] = x_train[:, 1] / 1000
initial_w = np.array([0.009, 0.009])
initial_b = -1.40741
alpha = 0.0003
m = x_train.shape[0]


def sigmoid(z):
    result = 1 / (1+np.exp(-z))
    return result


def compute_cost(w, b, x, y):
    total_cost = 0
    for i in range(x.shape[0]):
        z = np.dot(x[i], w) + b
        fw_b = sigmoid(z)
        loss = -y[i] * np.log(fw_b) - \
            (1-y[i]) * np.log(1 - fw_b)
        total_cost += loss
    return total_cost / m


def compute_gradient(w, b, x, y):
    dj_dw = 0
    dj_db = 0

    for i in range(x.shape[0]):
        z = np.dot(x[i], w) + b
        fw_b = sigmoid(z)
        dj_db += fw_b - y[i]
        for j in range(x.shape[1]):
            dj_dw += (fw_b - y[i]) * x[i][j]
    dj_dw /= x.shape[0]
    dj_db /= x.shape[0]
    return dj_dw, dj_db


def gradient_descent(w, b, x, y, alpha):

    for i in range(100000):
        dj_dw, dj_db = compute_gradient(w, b, x, y)
        w = w - alpha * dj_dw
        b -= alpha * dj_db
        if (i % 1000 == 0):
            print(f"Cost at {w[0], b}: {compute_cost(w, b, x, y)}")

    return w, b


initial_w, initial_b = gradient_descent(
    initial_w, initial_b, x_train, y_train, alpha)
"""
x_train2 = df.iloc[:, 2].values
y_train2 = df.iloc[:, 3].values
colors = ["red" if purchase ==
          0 else "blue" for purchase in df["Purchased"]]
fig, ax = plt.subplots(figsize=(14, 6))

ax.scatter(x_train2, y_train2, color=colors)
black_patch = plt.Line2D([], [], marker='o', color='red',
                         linestyle='', markersize=8, label='No purchase')
green_patch = plt.Line2D([], [], marker='o', color='blue',
                         linestyle='', markersize=8, label='Purchase')
plt.ylabel("Estimated Salary ($)")
plt.xlabel("Age")
plt.legend(handles=[black_patch, green_patch],
           loc="upper left", bbox_to_anchor=(1.0, 1))
plt.show()
"""
# Generate a grid of points
x1_min, x1_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
x2_min, x2_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.01),
                       np.arange(x2_min, x2_max, 0.01))

# Flatten the grid points and make predictions
grid_points = np.c_[xx1.ravel(), xx2.ravel()]
predicted_classes = (
    sigmoid(np.dot(grid_points, initial_w) + initial_b) >= 0.5).astype(int)

# Reshape the predicted classes to match the grid shape
predicted_classes = predicted_classes.reshape(xx1.shape)

# Plot the decision boundary and scatter plot
fig, ax = plt.subplots(figsize=(14, 6))
ax.contourf(xx1, xx2, predicted_classes, alpha=0.3, cmap=plt.cm.RdYlBu)
ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train,
           cmap=plt.cm.RdYlBu, edgecolors='k')
plt.ylabel("Estimated Salary ($)")
plt.xlabel("Age")
plt.show()
