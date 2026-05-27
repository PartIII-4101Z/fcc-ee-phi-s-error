#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def generate(q, C, S, D, N):

    if q==0:
        return np.random.exponential(scale = tau_b, size=N)

    # draw exponential decay times
    t = np.random.exponential(scale=1.0, size=N)

    if q == 1:
        mod = (np.cosh(y*t) + D*np.sinh(y*t)
               + C*np.cos(x*t) - S*np.sin(x*t))
    else:
        mod = (np.cosh(y*t) + D*np.sinh(y*t)
               - C*np.cos(x*t) + S*np.sin(x*t))

    # normalise modulation envelope
    mod_max = 1 + abs(D) + abs(C) + abs(S)   # safe global bound
    accept = np.random.rand(N) < (mod / mod_max)
    return t[accept]

