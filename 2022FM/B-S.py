import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
from math import log, sqrt, exp
from scipy import stats

def Eucall_MC(K):
    dt = T / N
    # simulation of index level paths
    S = np.zeros(N+1)
    C0 = 0
    for i in range(M):
        S[0] = S0
        for t in range(1, N + 1):
            S[t] = S[t - 1] * exp((r - 0.5 * sigma ** 2) * dt + sigma * sqrt(dt) * random.randn())
        C0 = C0 + exp(-r * T) * max(S[-1] - K, 0)
    C = C0/M

    return C


def Eucall_BS(S0, K, T, r, sigma):

    S0 = float(S0)
    d1 = (log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    value = S0 * stats.norm.cdf(d1, 0.0, 1.0)- K * exp(-r * T) * stats.norm.cdf(d2, 0.0, 1.0)

    return value

K=142
S0 = 111
r = 0.02
sigma = 0.42
T = 1.0
M = 1000
N = 50000

# Vega function

MC_res = []
BS_res = []
k_list = np.arange(80., 120.1, 5.)
np.random.seed(200000)
for K in k_list:
    MC_res.append(Eucall_MC(K))
    BS_res.append(Eucall_BS(S0, K, T, r, sigma))

MC_res = np.array(MC_res)
BS_res = np.array(BS_res)

fig,(ax1,ax2)= plt.subplots(2,1,sharex=True,figsize=(8,6))
ax1.plot(k_list,BS_res,'b',label='B-S formula')
ax1.plot(k_list,MC_res,'ro',label='Monte Carlo')
ax1.set_ylabel('European call option value')
ax1.grid(True)
ax1.legend(loc=0)
ax1.set_ylim(ymin=0)
wi=1.0
ax2.bar(k_list-wi/2,(np.array(BS_res)-np.array(MC_res))/np.array(BS_res)*100,wi)
ax2.set_xlabel('strike')
ax2.set_ylabel('difference in %')
ax2.set_xlim(left=75,right=125)
ax2.grid(True)

plt.show()
