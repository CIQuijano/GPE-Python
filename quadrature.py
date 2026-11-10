import numpy as np                         
from scipy.linalg import eigh_tridiagonal  

def gauss_rule(alpha, beta):                  
    '''Algoritmo de Golub-Welsch para obtener los nodos (autovalores) y pesos (autovectores escalados)
       de la cuadratura gaussiana desde la matrix de Jacobi.
    '''
    alpha = np.asarray(alpha, dtype=float)   
    beta = np.asarray(beta, dtype=float)     

    nodes, V = eigh_tridiagonal(alpha, np.sqrt(beta[1:]))          

    weights = beta[0] * V[0, :] ** 2  
    return nodes, weights             


def legendre_rule(n, a=-1.0, b=1.0):
    ''' n-point Gauss-Legendre rule on [a, b] (weight w(x) = 1).'''
    k = np.arange(n)
    alpha = np.zeros(n)
    beta = k**2 / (4.0 * k**2 -1)
    beta[0] = 2.0                                                 # beta_0 = length of [-1, 1]
    x, w = gauss_rule(alpha,beta)
    return 0.5 * (b - a) *  x + 0.5 * (b + a), 0.5 * (b - a) * w  # Cambio afín


def stieltjes(x, w, n): 
    '''
    Recurrence coefficients alpha_0...alpha_{n-1}, beta_0...beta_{n-1}
    of the discrete measure sum_j w_j delta_{x_k} 
    '''
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)

    if n > len(x):
        raise ValueError("n cannot exceed the number of support points")

    alpha = np.zeros(n)
    beta = np.zeros(n)

    beta[0] = np.sum(w)                         # beta_0 = total mass
    p_prev = np.zeros_like(x)                   # p_{-1} = 0
    p = np.full_like(x, 1 / np.sqrt(beta[0]))
    
    for k in range(n):
        alpha[k] = np.sum(w * x * p**2)
        if k == n - 1:
            break
        q = (x - alpha[k]) * p - np.sqrt(beta[k]) * p_prev
        beta[k + 1] = np.sum(w * q**2)
        p_prev, p = p, q / np.sqrt(beta[k + 1])

    return alpha, beta