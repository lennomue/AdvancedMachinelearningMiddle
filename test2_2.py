import numpy as np
import matplotlib.pyplot as plt

def soft_thresholding(x, thresh):
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0.0)

def objective(w, A, mu, lam):
    return 0.5 * (w - mu).T @ A @ (w - mu) + lam * np.linalg.norm(w, 1)

def proximal_gradient(A, mu, lam, max_iter=100):
    L = np.linalg.eigvalsh(A).max()  # Lipschitz constant (max eigenvalue)
    eta = 1.0 / L
    w = np.zeros_like(mu)
    history = []

    for _ in range(max_iter):
        grad = A @ (w - mu)
        w_new = soft_thresholding(w - eta * grad, eta * lam)
        loss = objective(w_new, A, mu, lam)
        history.append(loss)
        w = w_new

    return w, history

# Problem 1 settings
A = np.array([[3.0, 0.5],
              [0.5, 1.0]])
mu = np.array([1.0, 2.0])

# lambda range for part 1 (L1 regularization path)
lambda_vals = np.linspace(0.01, 10.00, 1000)

w0_vals = []
w1_vals = []

# 実験ループ
for lam in lambda_vals:
    w_hat, _ = proximal_gradient(A, mu, lam)
    w0_vals.append(w_hat[0])
    w1_vals.append(w_hat[1])
    


# numpy array に変換
w0_vals = np.array(w0_vals)
w1_vals = np.array(w1_vals)

plt.figure(figsize=(10, 5))
plt.plot(lambda_vals, w0_vals, label='$w_0$')
plt.plot(lambda_vals, w1_vals, label='$w_1$')
plt.xlabel('$\\lambda$')
plt.ylabel('Optimal $\\hat{w}$ values')
plt.title('Optimal $\\hat{w}$ vs $\\lambda$')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
