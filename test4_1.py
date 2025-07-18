import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
n = 200
d = 4
x = 3 * (np.random.rand(n, d) - 0.5)
y = (2 * x[:, 0] - x[:, 1] + 0.5 + 0.5 * np.random.randn(n)) > 0
y = 2 * y.astype(int) - 1

lam = 1
max_iter = 300
eta = 0.01

# 初期化
w = np.zeros(d)
loss_history = []
w_history = []

def soft_thresholding(x, thresh):
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0.0)

loss = np.maximum(0, 1 - y * (x @ w)).sum() + lam * np.linalg.norm(w, ord=1)
loss_history.append(loss)
# 最適化ループ
for it in range(max_iter):
    w_history.append(w)
    margin = y * x.dot(w)
    indicator = margin < 1
    subgrad = -np.sum((indicator[:, None] * y[:, None] * x), axis=0)
    w = soft_thresholding(w - eta * subgrad, eta * lam)
    loss = np.maximum(0, 1 - y * (x @ w)).sum() + lam * np.linalg.norm(w, ord=1)
    loss_history.append(loss)
    
# w_history = np.array(w_history)
# plt.figure(figsize=(8, 5))
# for i in range(4):
#     plt.plot(w_history[:, i], label=f"$w_{i+1}$")

# plt.xlabel("Iteration")
# plt.ylabel("parameter $w$")
# plt.title(f'PG(λ={lam},η={eta})')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# 結果表示
print("w in the end", w)
plt.plot(loss_history)
plt.xlabel("iteration")
plt.ylabel("loss function")
plt.title(f'PG(λ={lam},η={eta})')
plt.grid(True)
plt.show()
