import numpy as np

def sliding_rms(x, fs, win_sec):
    n = len(x); w = int(round(win_sec * fs))
    if w < 1: raise ValueError("Window too short")
    x2 = x**2; c = np.cumsum(np.r_[0.0, x2])
    rms = np.zeros(n)
    rms[w-1:] = np.sqrt((c[w:] - c[:-w]) / w)
    rms[:w-1] = rms[w-1]
    return rms
