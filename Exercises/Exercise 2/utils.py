import numpy as np
from numpy.polynomial.hermite import hermgauss

def hermgauss_lognorm(n, sigma):

    x, w = hermgauss(n)
    x = np.exp(x*np.sqrt(2)*sigma-0.5*sigma**2);
    w = w/np.sqrt(np.pi);

    return x, w
