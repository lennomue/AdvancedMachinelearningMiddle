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
K = (y[:, None] * x) @ (y[:, None] * x).T

# プロジェクション演算子
def project(alpha):
    return np.clip(alpha, 0, 1)

# パラメータ設定
# lambdas = [0.01, 0.1, 1, 10]
lambda_ = 0.1
eta = 0.01
iterations = 200

results = {}

alpha = np.zeros(n)
dual_history = []
primal_history = []

for t in range(iterations):
    # 勾配計算: ∇ = (1 / 2λ) Kα - 1
    grad = (1 / (2 * lambda_)) * K @ alpha - 1

    # 勾配ステップと射影
    alpha = project(alpha - eta * grad)

    # w の計算
    w = (1 / (2 * lambda_)) * np.sum((alpha * y)[:, None] * x, axis=0)

    # ヒンジ損失と正則化項による元の目的関数の値
    hinge_loss = np.maximum(0, 1 - y * (x @ w)).sum()
    primal_obj = hinge_loss + (lambda_ / 2) * np.dot(w, w)

    # 双対目的関数の値
    dual_obj = - (1 / (4 * lambda_)) * alpha @ K @ alpha + np.sum(alpha)

    dual_history.append(dual_obj)
    primal_history.append(primal_obj)

plt.figure(figsize=(8, 5))
plt.plot(range(iterations), dual_history, label="Dual Lagrangian", color="blue")
plt.plot(range(iterations), primal_history, label="Primal Loss", color="red", linestyle='--')
plt.xlabel("Iteration")
plt.ylabel("Function Value")
plt.title(f"Primal and Negative Dual Function Values (λ={lambda_})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(dual_history[-1])
print(primal_history[-1])