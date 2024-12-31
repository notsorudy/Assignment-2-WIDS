import csv
import numpy as np

# Step 1: Load and Clean Data
def load_and_clean_data(filepath):
    data = []
    with open(filepath, mode='r') as file:
        csvFile = csv.reader(file)
        for row in csvFile:
            data.append(row)

    # Remove rows with 'NA'
    filtered_array = [row for row in data if 'NA' not in row]
    
    # Convert to NumPy array and separate target variable
    d = np.array(filtered_array)
    d = d[1:]  # Remove header
    
    y = np.array([float(row[15]) for row in d])
    d = np.delete(d, 15, axis=1).astype(float)  # Remove target column and convert to float
    return d, y

# Step 2: Define Sigmoid Function
def sigmoid(z):
    z = np.clip(z, -500, 500)  # Avoid overflow
    return 1 / (1 + np.exp(-z))

# Step 3: Compute Cost for Logistic Regression
def compute_cost_logistic_reg(X, y, w, b):
    m, n = X.shape
    cost = 0.
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        cost += -y[i] * np.log(f_wb_i) - (1 - y[i]) * np.log(1 - f_wb_i)
    cost = cost / m
    return cost

# Step 4: Compute Gradients
def compute_gradient_logistic_reg(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.0

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i], w) + b)
        err_i = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] += err_i * X[i, j]
        dj_db += err_i

    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_db, dj_dw

# Step 5: Gradient Descent for Logistic Regression
def gradient_descent_logistic(X, y, w_in, b_in, alpha, num_iters):
    b = b_in
    w = w_in
    for i in range(num_iters):
        dj_db, dj_dw = compute_gradient_logistic_reg(X, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
    return w, b

# Step 6: Min-Max Scaling
def min_max_scale(X):
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    return (X - X_min) / (X_max - X_min)

# Step 7: Predict Probabilities
def probability_values(X, w_model, b_model):
    m, n = X.shape
    a = []
    for i in range(m):
        z_i = np.dot(w_model, X[i]) + b_model
        f_wb = sigmoid(z_i)
        a.append(f_wb)
    return a

# Step 8: Predict Classes
def predict_classes(probabilities, threshold=0.3):
    return [1 if p > threshold else 0 for p in probabilities]

# Main Execution
filepath = 'C:\\Users\\Dell\\Downloads\\framingham.csv'
d, y = load_and_clean_data(filepath)

# Scale the features
d_scaled = min_max_scale(d)

# Initialize parameters
w_in = np.zeros((d_scaled.shape[1],))
b_in = 0
alpha = 1
num_iters = 500

# Perform gradient descent
w, b = gradient_descent_logistic(d_scaled, y, w_in, b_in, alpha, num_iters)
print("Optimized weights:", w)
print("Optimized bias:", b)

# Predict probabilities and classes
probabilities = probability_values(d_scaled, w, b)
y_model = predict_classes(probabilities)

print("Predicted classes:", y_model)
