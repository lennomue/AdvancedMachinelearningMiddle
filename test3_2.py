import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

n = 40
omega = np.random.randn(1)
noise = 0.8 * np.random.randn(n, 1)
x = np.random.randn(n, 2)
y = 2 * ((omega * x[:, 0:1] + x[:, 1:2] + noise) > 0) - 1
y = y.flatten()

# カーネル行列 K の構築: K_ij = y_i y_j x_i^T x_j
# K = (y[:, None] * x) @ (y[:, None] * x).T

# プロジェクション演算子
# def project(alpha):
#     return np.clip(alpha, 0, 1)

# パラメータ設定
# lambdas = [0.01, 0.1, 1, 10]
lambda_ = 0.1
eta = 0.01
iterations = 200

def primal_subgradient(x, y, lam, num_iters=200, eta=0.1):
    n, d = x.shape
    w = np.zeros(d)
    losses = []

    for t in range(num_iters):
        margins = y * (x @ w)
        indicator = margins < 1  # Active where hinge loss > 0
        subgrad = -np.sum((y[indicator, None] * x[indicator]), axis=0) + lam * w
        w -= eta * subgrad

        hinge_loss = np.maximum(0, 1 - margins)
        loss = np.sum(hinge_loss) + (lam / 2) * np.dot(w, w)
        losses.append(loss)

    return w, losses

w, losses = primal_subgradient(x, y, lambda_, num_iters=iterations, eta=eta)

plt.figure(figsize=(8, 5))
plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("primal lass")
plt.title(f"optimization by subgradient method (λ={lambda_})")
plt.legend()
plt.show()

print(losses[-1])
