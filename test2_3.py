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

def fista(A, mu, lam, max_iter=100):
    L = np.linalg.eigvalsh(A).max()
    eta = 1.0 / L

    w = np.zeros_like(mu)
    v = w.copy()
    s = 1
    history = []

    for _ in range(max_iter):
        grad = A @ (v - mu)
        w_new = soft_thresholding(v - eta * grad, eta * lam)
        s_new = (1 + np.sqrt(1 + 4 * s ** 2)) / 2
        q = (s - 1) / s_new
        v = w_new + q * (w_new - w)

        loss = objective(w_new, A, mu, lam)
        history.append(loss)

        w = w_new
        s = s_new

    return w, history

# Problem 2 settings
A2 = np.array([[300.0, 0.5],
               [0.5, 10.0]])
mu = np.array([1.0, 2.0])
lam = 0.1
iterater = 800
# Standard PG
w_pg, hist_pg = proximal_gradient(A2, mu, lam=lam, max_iter=iterater)
# fista
w_f, hist_f = fista(A2, mu, lam=lam, max_iter=iterater)

print(hist_pg[-1])
print(hist_f[-1])


J_star = min(hist_pg[-1], hist_f[-1])

loss_pg = [abs(Jt - J_star) for Jt in hist_pg]
loss_f = [abs(Jt - J_star) for Jt in hist_f]

plt.figure()
plt.semilogy(loss_pg, label='PG')
plt.semilogy(loss_f, label='FISTA')
plt.xlabel('Iteration')
plt.ylabel('Objective Difference (log scale)')
plt.title(f'PG vs AdaGrad Convergence (λ={lam})')
plt.legend()
plt.grid(True)
plt.show()
