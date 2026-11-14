"""
Orthonormal polynomials from their recurrence coefficients. 

evaluate : values p_k(x) and derivatives p_k'(x), k = 0,...,N-1, at given points. 
"""

import numpy as np

def evaluate(alpha, beta, x, N):
    """
    Values and derivatives of the orthonormal polynomials p_0,...,p_{N-1}.
    Returns (P, dP), arrays of shape (len(x), N) with
    P[i, k] = p_k(x_i) and dP[i, k] = p_k'(x_i).
    """
    x = np.asarray(x, dtype=float)
    P = np.zeros((len(x), N))
    dP = np.zeros((len(x), N))

    P[:, 0] = 1 / np.sqrt(beta[0])                  # Theorem 1.3   
    if N > 1:
        P[:, 1] = (x - alpha[0]) * P[:, 0] / np.sqrt(beta[1])
        dP[:, 1] = P[:, 0] / np.sqrt(beta[1])
    for k in range(1, N - 1):
        P[:, k + 1] = ((x - alpha[k]) * P[:, k] - np.sqrt(beta[k]) * P[:, k -1]) / np.sqrt(beta[k + 1])
        dP[:, k + 1] = (P[:, k] + (x - alpha[k]) * dP[:, k] - np.sqrt(beta[k]) * dP[:, k - 1]) / np.sqrt(beta[k + 1])
    return P, dP


