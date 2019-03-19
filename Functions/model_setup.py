# Model set-up
from Functions.struct import Struct
par = Struct()

import numpy as np
from numpy.polynomial.hermite import hermgauss

#################################
############ Model ##############
#################################

par.T = 10
par.NM = 100
par.M_max = 5 # Need to specify what we believe to be the maximum achievable income over the period
#par.grid_M = np.linspace(0,par.M_max, par.NM)
par.beta = 0.98

par.rho = 0.5
par.u = lambda x, par: (x**(1-par.rho))/(1-par.rho)

par.R = 1/par.beta

# Gaussian Quadrature
par.S = 8
par.sigma_zeta = 0.2
par.mu = 0

par.x,par.w = hermgauss(par.S)
par.Y = np.exp(par.x * np.sqrt(2) * par.sigma_zeta) # S-dimensional vector
par.w = par.w/np.sqrt(np.pi)

par.tolerance = 10**(-6)

# For infinite case
par.max_iter = 500

######################################
############ Simulation ##############
######################################

par.simN = 10000
par.M_ini = 1.5
