import numpy as np

def rayleigh_coeffs(w1, w2, zeta=0.01):
    a0 = 2*zeta*w1*w2/(w1 + w2)
    a1 = 2*zeta/(w1 + w2)
    return a0, a1

def C_rayleigh(M, K, a0, a1):
    return a0*M + a1*K

def modal_damping(C, M, K, phi, w):
    zetas = []
    for i in range(len(w)):
        v = phi[:, i:i+1]
        num = float(v.T @ C @ v)
        den = 2.0 * w[i] * float(v.T @ M @ v)
        zetas.append(num/den)
    return np.array(zetas)
