"""
Example 2: recurrence coefficients of exp(-x^2)dx by the Stieltjes procedure
on a discretized measure, compared with alpha_k = 0, beta_0 = sqrt(pi), beta_k = k/2.

Usage: 
    python example2.py                      (n = 20, L = 8, nd = 200)
    python example2.py --n 40 --L 12 --nd 400
"""

import argparse
import math

import numpy as np 

from quadrature import legendre_rule, stieltjes

def discretize(weight, L, nd):
    """Discrete measure approximating weight(x)dx on [-L, L]."""
    zeta, g = legendre_rule(nd, -L, L)
    return zeta, g * weight(zeta)

def hermite_exact(n):
    alpha = np.zeros(n)
    beta = np.arange(n) / 2.0
    beta[0] = math.sqrt(math.pi)
    return alpha, beta

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Stieltjes procedure on exp(-x^2)dx.")
    parser.add_argument("--n", type=int, default=20, help="number of coefficients")
    parser.add_argument("--L", type=float, default=8.0, help="half-length of the interval")
    parser.add_argument("--nd", type=int, default=200, help="discretization points")
    args = parser.parse_args()

    x, w = discretize(lambda t: np.exp(-t**2), args.L, args.nd)
    alpha, beta = stieltjes(x, w, args.n)
    alpha_ex, beta_ex = hermite_exact(args.n)

    print(f"max |alpha_k|        : {np.max(np.abs(alpha)):.1e}")
    print(f"max |beta_k - exact| : {np.max(np.abs(beta - beta_ex)):.1e}")

    