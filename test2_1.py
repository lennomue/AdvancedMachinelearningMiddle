import numpy as np
import matplotlib.pyplot as plt

def soft_thresholding(x, thresh):
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0.0)

def objective(w, A, mu, lam):
    return 0.5 * (w - mu).T @ A @ (w - mu) + lam * np.linalg.norm(w, 1)

def proximal_gradient(A, mu, lam, max_iter=100):
    L = np.linalg.eigvalsh(A).max()
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

w_list = []
lambda_list = []

lam_ = 1.0

w_hat, history = proximal_gradient(A, mu, lam_)

J_star = objective(w_hat, A, mu, lam_)
errors = [abs(Jt - J_star) for Jt in history]

plt.figure(figsize=(8, 5))
plt.semilogy(errors, label='$|J(w^{(t)}) - J(\\hat{w})|$')
plt.xlabel('Iteration $t$')
plt.ylabel('Objective Difference')
plt.title(f'Proximal Gradient Convergence (λ = {lam_})')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
