import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
from math import sqrt, exp
from scipy import stats

S0 = 1.5
K = 1.4
r = 0.03
sigma = 0.12
T = 1.0
M = 3000
N = 1200
U = 1.71
L = 1.29
U2 = 1.8
L2 = 1.3

dt = T/N
S = np.zeros(N+1)
C = np.zeros(M)
x = np.linspace(0, T, N + 1)

for n in range(10):

    for m in range(M):
        S[0] = S0
        for t1 in range(int(T / 12 * N)):
            S[t1 + 1] = S[t1] * (1 + r * dt) + sigma * S[t1] * sqrt(dt) * random.randn()
        if S[int(T / 12 * N)] != 0:
            for t2 in range(int(T / 12 * N), int(6 * T / 12 * N)):
                S[t2 + 1] = S[t2] * (1 + r * dt) + sigma * S[t2] * sqrt(dt) * random.randn()
                if S[t2 + 1] >= U:
                    for i in range(t2 + 1, N):
                        S[i + 1] = 0
                    break
        if S[int(6 * T / 12 * N)] != 0:
            for t3 in range(int(6 * T / 12 * N), int(8 * T / 12 * N)):
                S[t3 + 1] = S[t3] * (1 + r * dt) + sigma * S[t3] * sqrt(dt) * random.randn()
        if S[int(8 * T / 12 * N)] != 0:
            for t4 in range(int(8 * T / 12 * N), int(11 * T / 12 * N)):
                S[t4 + 1] = S[t4] * (1 + r * dt) + sigma * S[t4] * sqrt(dt) * random.randn()
                if S[t4 + 1] <= L:
                    for i in range(t4 + 1, N):
                        S[i + 1] = 0
                    break
        if S[int(11 * T / 12 * N)] != 0:
            for t5 in range(int(11 * T / 12 * N), N):
                S[t5 + 1] = S[t5] * (1 + r * dt) + sigma * S[t5] * sqrt(dt) * random.randn()
                if (S[t5 + 1] >= U2) or (S[t5 + 1] <= L2):
                    for i in range(t5 + 1, N):
                        S[i + 1] = 0
                    break
        #plt.plot(x, S, linewidth=0.5)
        C[m] = exp(-r * T) * max(S[-1] - K, 0)
    Mean = np.average(C)
    print(Mean)

'''
plt.xlim(0,1)
plt.ylim(1.2,2)
plt.axhline(y=U,xmin=T/12,xmax=6*T/12)
plt.axhline(y=L,xmin=8*T/12,xmax=11*T/12)
plt.axhline(y=U2,xmin=11*T/12,xmax=1)
plt.axhline(y=L2,xmin=11*T/12,xmax=1)
plt.title("The underlying path plot")
plt.xlabel("Time")
plt.ylabel("Price")

plt.show()'''