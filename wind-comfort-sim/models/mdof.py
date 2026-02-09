import numpy as np
from scipy.linalg import eigh

def shear_building(n=10, m=1.0, k=1.0e6):
    # force numeric types (handles YAML strings like "1e6")
    n = int(n); m = float(m); k = float(k)

    M = np.diag(np.full(n, float(m))).astype(float)
    K = np.zeros((n, n), dtype=float)
    for i in range(n):
        if i > 0:     K[i, i-1] = -float(k)
        if i < n-1:   K[i, i+1] = -float(k)
        K[i, i] = (2*float(k) if 0 < i < n-1 else float(k))
    return M, K

def modal_analysis(M, K):
    w2, phi = eigh(K, M)
    w = np.sqrt(np.clip(w2, 0, None))
    f = w / (2*np.pi)
    phi = phi / phi[-1, :]    # normalize by roof
    return f, w, phi
