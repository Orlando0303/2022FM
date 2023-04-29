import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt

def Eucall_MC(K):
    dt = T / N
    S = np.zeros(N+1)
    C0 = 0
    x = np.linspace(0,T,N+1)
    for i in range(M):
        S[0] = S0
        for t in range(1, N + 1):
            S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * random.randn())
        C0 = C0 + np.exp(-r * T) * max(S[-1] - K, 0)
        #plt.plot(x,S,linewidth = 0.5)
    C = C0 / M
    return C


S0 = 111
K = 142
r = 0.02
sigma = 0.42
T = 1.0
M = 1000
N = 10000

for i in range(10):
    print(Eucall_MC(K))
'''
plt.title("The underlying path plot")
plt.xlabel("Time")
plt.ylabel("Price")

plt.show()'''