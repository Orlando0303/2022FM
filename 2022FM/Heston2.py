import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
from math import sqrt, exp, log

S0 = 111
K = 142
r = 0.02
sigma0 = 0.42
T = 1.0
M = 1000
N = 10000
rho = 0.5
beta = 0.5
delta = 0.1

dt = T/N
cor = np.zeros((2,2))
cor[0,0], cor[1,1] = 1,1
cor[0,1], cor[1,0] = rho, rho
u = np.linalg.cholesky(cor) #奇异值分解
S = np.zeros(N+1)
C = np.zeros(M)
sigma = np.zeros(N+1)
x = np.linspace(0, T, N + 1)

for m in range(M):
    S[0] = S0
    sigma[0] = sigma0
    for i in range(N):
        z = random.randn(2,1)
        z1 = np.zeros((2,1))
        z1[0,0] = z[0,0] * u[0,0] + z[1,0] * u[0,1]
        z1[1,0] = z[0,0] * u[1,0] + z[1,0] * u[1,1]
        S[i + 1] = S[i] * (1 + r * dt) + sigma[i] * S[i] * sqrt(dt) * z1[0,0] + 0.5 * (sigma[i] ** 2) * S[i] * dt * (z1[0,0] ** 2 - 1)
        sigma[i + 1] = max(sigma[i] - beta * sigma[i] * dt + delta * sqrt(dt) * z1[1,0],0)
    plt.ion()
    plt.figure(1)
    plt.plot(x, S, linewidth=0.5)
    plt.figure(2)
    plt.plot(x, sigma, linewidth=0.5)
    C[m] = exp(-r * T) * max(S[-1] - K, 0)
Mean = np.average(C)

print(Mean)

plt.figure(1)
plt.title("The underlying path plot")
plt.xlabel("Time")
plt.ylabel("Price")

plt.figure(2)
plt.title("The sigma path plot")
plt.xlabel("Time")
plt.ylabel("sigma")

plt.ioff()
plt.show()