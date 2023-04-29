import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
from math import sqrt, exp, log

S0 = 1.5
K = 1.4
r = 0.03
sigma0 = 0.12
T = 1.0
M = 1000
N = 1200
U = 1.71
L = 1.29
U2 = 1.8
L2 = 1.3
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
    for t1 in range(int(T / 12 * N)):
        z = random.randn(2,1)
        z1 = np.zeros((2,1))
        z1[0,0] = z[0,0] * u[0,0] + z[1,0] * u[0,1]
        z1[1,0] = z[0,0] * u[1,0] + z[1,0] * u[1,1]
        S[t1 + 1] = S[t1] * (1 + r * dt) + sigma[t1] * S[t1] * sqrt(dt) * z1[0,0] + 0.5 * (sigma[t1] ** 2) * S[t1] * dt * (z1[0,0] ** 2 - 1)
        sigma[t1 + 1] = max(sigma[t1] - beta * sigma[t1] * dt + delta * sqrt(dt) * z1[1,0],0)
    if S[int(T / 12 * N)] != 0:
        for t2 in range(int(T / 12 * N), int(6 * T / 12 * N)):
            z = random.randn(2, 1)
            z1 = np.zeros((2, 1))
            z1[0, 0] = z[0, 0] * u[0, 0] + z[1, 0] * u[0, 1]
            z1[1, 0] = z[0, 0] * u[1, 0] + z[1, 0] * u[1, 1]
            S[t2 + 1] = S[t2] * (1 + r * dt) + sigma[t2] * S[t2] * sqrt(dt) * z1[0, 0] + 0.5 * (sigma[t2] ** 2) * S[
                t2] * dt * (z1[0, 0] ** 2 - 1)
            sigma[t2 + 1] = max(sigma[t2] - beta * sigma[t2] * dt + delta * sqrt(dt) * z1[1, 0],0)
            if S[t2 + 1] >= U:
                for i in range(t2 + 1, N):
                    S[i + 1] = 0
                break
    if S[int(6 * T / 12 * N)] != 0:
        for t3 in range(int(6 * T / 12 * N), int(8 * T / 12 * N)):
            z = random.randn(2, 1)
            z1 = np.zeros((2, 1))
            z1[0, 0] = z[0, 0] * u[0, 0] + z[1, 0] * u[0, 1]
            z1[1, 0] = z[0, 0] * u[1, 0] + z[1, 0] * u[1, 1]
            S[t3 + 1] = S[t3] * (1 + r * dt) + sigma[t3] * S[t3] * sqrt(dt) * z1[0, 0] + 0.5 * (sigma[t3] ** 2) * S[
                t3] * dt * (z1[0, 0] ** 2 - 1)
            sigma[t3 + 1] = max(sigma[t3] - beta * sigma[t3] * dt + delta * sqrt(dt) * z1[1, 0],0)
    if S[int(8 * T / 12 * N)] != 0:
        for t4 in range(int(8 * T / 12 * N), int(11 * T / 12 * N)):
            z = random.randn(2, 1)
            z1 = np.zeros((2, 1))
            z1[0, 0] = z[0, 0] * u[0, 0] + z[1, 0] * u[0, 1]
            z1[1, 0] = z[0, 0] * u[1, 0] + z[1, 0] * u[1, 1]
            S[t4 + 1] = S[t4] * (1 + r * dt) + sigma[t4] * S[t4] * sqrt(dt) * z1[0, 0] + 0.5 * (sigma[t4] ** 2) * S[
                t4] * dt * (z1[0, 0] ** 2 - 1)
            sigma[t4 + 1] = max(sigma[t4] - beta * sigma[t4] * dt + delta * sqrt(dt) * z1[1, 0],0)
            if S[t4 + 1] <= L:
                for i in range(t4 + 1, N):
                    S[i + 1] = 0
                break
    if S[int(11 * T / 12 * N)] != 0:
        for t5 in range(int(11 * T / 12 * N), N):
            z = random.randn(2, 1)
            z1 = np.zeros((2, 1))
            z1[0, 0] = z[0, 0] * u[0, 0] + z[1, 0] * u[0, 1]
            z1[1, 0] = z[0, 0] * u[1, 0] + z[1, 0] * u[1, 1]
            S[t5 + 1] = S[t5] * (1 + r * dt) + sigma[t5] * S[t5] * sqrt(dt) * z1[0, 0] + 0.5 * (sigma[t5] ** 2) * S[
                t5] * dt * (z1[0, 0] ** 2 - 1)
            sigma[t5 + 1] = max(sigma[t5] - beta * sigma[t5] * dt + delta * sqrt(dt) * z1[1, 0],0)
            if (S[t5 + 1] >= U2) or (S[t5 + 1] <= L2):
                for i in range(t5 + 1, N):
                    S[i + 1] = 0
                break

    '''plt.ion()
    plt.figure(1)
    plt.plot(x,S,linewidth = 0.5)
    plt.figure(2)
    plt.plot(x,sigma,linewidth = 0.5)'''

    C[m] = exp(-r * T) * max(S[-1] - K, 0)
Mean = np.average(C)

print(Mean)
'''
plt.figure(1)
plt.xlim(0,1)
plt.ylim(1,2)
plt.axhline(y=U,xmin=T/12,xmax=6*T/12)
plt.axhline(y=L,xmin=8*T/12,xmax=11*T/12)
plt.axhline(y=U2,xmin=11*T/12,xmax=1)
plt.axhline(y=L2,xmin=11*T/12,xmax=1)
plt.title("The underlying path plot")
plt.xlabel("Time")
plt.ylabel("Price")

plt.ioff()
plt.show()'''

