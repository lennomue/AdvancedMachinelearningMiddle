import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(0)

# Generate data
n = 200
x = 3 * (np.random.rand(n, 3) - 0.5)
y = (2 * x[:, 0] - 1 * x[:, 1] + 0.5 + 0.5 * np.random.randn(n)) > 0
y = 2 * y.astype(int) - 1

# Add bias term (1) to the data
X = np.hstack([np.ones((n, 1)), x])
d = X.shape[1]
lambda_reg = 1e-2

# Logistic loss and gradient
def logistic_loss(w, X, y, lam):
    z = y * (X @ w)
    return np.sum(np.log(1 + np.exp(-z))) + lam / 2 * np.dot(w, w)

def logistic_grad(w, X, y, lam):
    z = y * (X @ w)
    grad = -(y / (1 + np.exp(z))) @ X
    return grad + lam * w

def logistic_hessian(w, X, y, lam):
    z = y * (X @ w)
    s = 1 / (1 + np.exp(-z))
    D = s * (1 - s)
    H = X.T @ (D[:, None] * X)
    return H + lam * np.eye(X.shape[1])

# Batch steepest gradient descent
def batch_gradient_descent(X, y, lam, lr=0.1, max_iter=100):
    w = np.zeros(X.shape[1])
    losses = []
    losses.append(logistic_loss(w, X, y, lam))
    for _ in range(max_iter):
        grad = logistic_grad(w, X, y, lam)
        w -= lr * grad
        losses.append(logistic_loss(w, X, y, lam))
    return w, losses

# Newton's method
def newton_method(X, y, lam, max_iter=100):
    w = np.zeros(X.shape[1])
    losses = []
    losses.append(logistic_loss(w, X, y, lam))
    for _ in range(max_iter):
        grad = logistic_grad(w, X, y, lam)
        H = logistic_hessian(w, X, y, lam)
        w -= np.linalg.solve(H, grad)
        losses.append(logistic_loss(w, X, y, lam))
    return w, losses

# Run optimization
w_gd, losses_gd = batch_gradient_descent(X, y, lambda_reg, lr=0.1, max_iter=100)
w_nt, losses_nt = newton_method(X, y, lambda_reg, max_iter=100)

# Determine reference minimum value
J_star = min(losses_gd[-1], losses_nt[-1])

# Calculate log-scale difference from optimal
epsilon = 1e-14
log_diff_gd = [max(abs(J - J_star), epsilon) for J in losses_gd]
log_diff_nt = [max(abs(J - J_star), epsilon) for J in losses_nt]

# Plot
plt.figure(figsize=(8, 5))
plt.semilogy(log_diff_gd, label="gradient descent")
plt.semilogy(log_diff_nt, label="Newton")
# plt.plot(log_diff_gd, label="gradient descent")
# plt.plot(log_diff_nt, label="Newton")
plt.xlabel("iteration")
plt.ylabel(r"$|J(w^{(t)}) - J(\hat{w})|$")
plt.legend()
plt.grid(True)
plt.title("Convergence of Optimization Methods")
plt.tight_layout()
plt.show()
