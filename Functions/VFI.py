# Functions for Value function iteration (VFI)
'''This module contains functions for value function iteration for DP. These include:
- find_V
- vfi_finite
- simulate_path
- Functions for tasteshocks
'''

# Import libraries
import numpy as np
import os
from numpy.polynomial.hermite import hermgauss
from scipy import interpolate
from scipy.optimize import minimize_scalar
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

def find_V(par,last = 1):
    """ Function to find V within a loop"""

    # a) Loop over states
    V = np.ones(par.NM) * np.nan
    Cstar = np.ones(par.NM) * np.nan

    for i_M in range(0,par.NM):
        Mt = par.grid_M[i_M]

        # Value-of-choice function
        if last == 1: # If we are interested in last period
            Vfunc = par.u # Assign function to Vfunc
        else:
            Vfunc = lambda C,par: par.u(C,par) + par.beta * sum(par.w * par.V_plus_interp(par.R * (Mt-C) + par.Y))

        # Convert function to negative for minimization

        Vfunc_neg = lambda x,par: -Vfunc(x,par)

        # b) Initial guess
        if i_M == 1:
            initial_guess = 0.5 * Mt
        else:
            initial_guess = Cstar[i_M-1]

        # c) Find optimum
        res = minimize_scalar(Vfunc_neg, tol = par.tolerance, args = par
                       , bounds = [0,Mt], method = "bounded")

        V[i_M] = float(-res.fun)
        Cstar[i_M] = float(res.x)

    return(V,Cstar)


def vfi_finite(par):
    '''Finds optimal consumption paths in DP. Requires the call of function find_V'''

    # 1) Allocate memory
    Cstar, Vstar = dict(), dict()
    if par.rho < 1.0:
        par.grid_M = np.linspace(0,par.M_max, par.NM)
    else:
        par.grid_M = np.linspace(par.tolerance,par.M_max, par.NM)

    # 2) Last Period
    Vstar[par.T], Cstar[par.T] = find_V(par, last = 1)

    # 3) Backwards over time
    for t in reversed(range(1,par.T)): # Start in period T-1

        # a) Create interpolant of next period ressources
        par.V_plus_interp = interpolate.interp1d(par.grid_M, Vstar[t+1], kind='linear', fill_value = "extrapolate")

        # b) Find V for all other states
        Vstar[t], Cstar[t] = find_V(par,last = 0)

    return(Cstar,Vstar)


def vfi_infinite(par):
    V = np.zeros(par.NM)
    delta = np.inf
    iteration = 0

    while iteration < par.max_iter:
        V_ = V
        par.V_plus_interp = interpolate.interp1d(par.grid_M, V_, kind='linear', fill_value = "extrapolate")
        V,C = find_V(par, last = 0)
        delta = np.max(abs(V-V_))
        iteration +=1
        if delta < par.tolerance:
                break
    return(C,V,iteration)

def vfi_infinite2(par):
    try:
        V = np.zeros(par.NM)
        delta = np.inf
        iteration = 0

        while iteration < par.max_iter:
            V_ = V
            par.V_plus_interp = interpolate.interp1d(par.grid_M, V_, kind='linear', fill_value = "extrapolate")
            V,C = find_V(par, last = 0)
            delta = np.max(abs(V-V_))
            iteration +=1
            if delta < par.tolerance:
                    break
    # Allow early stopping
    except KeyboardInterrupt:
        print('KeyboardInterrupt')

    return(C,V,iteration)


def simulate_path(Cstar, par):
    '''Simulates the consumption path given Cstar and parameters of the model'''
    # Allocate
    simM = par.M_ini * np.ones((par.simN, par.T))
    simC = np.ones((par.simN, par.T))

    for t in range(1,par.T+1):
        Cstar_interp = interpolate.interp1d(par.grid_M, Cstar[t], kind='linear', fill_value = "extrapolate")
        simC[:,t-1] = Cstar_interp(simM[:,t-1])

        if t < par.T:
            A = simM[:,t-1] - simC[:,t-1]
            Y = np.exp(np.random.normal(par.mu,par.sigma_zeta,par.simN) * par.sigma_zeta)
            simM[:,t] = par.R * A + Y
    return(simC)


########################################################################################################################
############################################# Taste Shocks #############################################################
########################################################################################################################

def logsum(v1,v2,sigma):

    #1) Setup
    V = np.array((v1,v2))
    Dim = V.shape
    #2) Max over discrete choices
    mxm = np.max(V, axis = 0)
    id_max = np.argmax(V, axis = 0)
    # 3) Logsum and probabilities
    assert abs(sigma) > 1.0*10**(-5) # Requires value to be high enough. For our purposes, it is.
    # a) Log-sum
    LogSum = mxm + sigma * np.log(np.sum(np.exp((V-mxm)/sigma), axis = 0))
    # b) Probability
    Prob = np.exp((V-LogSum)/sigma)
    return(LogSum,Prob)

def value_of_choice(C,L,Mt,last,par):

    if last == 1: # Last period
        V = par.u(C,par) - par.lambda_ * L
    else:
        M_plus = par.R * (Mt -C) + L
        V1 = par.V_plus_interp[0](M_plus)
        V2 = par.V_plus_interp[1](M_plus)
        V = par.u(C,par) - par.lambda_ * L + par.beta * logsum(V1,V2,par.sigma_eps)[0]
    return(V)


def find_V_tasteshocks(par, L, last):

    # Loop over states
    V = np.ones(par.NM) * np.nan
    Cstar = np.ones(par.NM) * np.nan

    for i_M in range(0,par.NM):
        Mt = par.grid_M[i_M]

        # Initial guess
        if i_M == 1:
            initial_guess = 0.5 * Mt
        else:
            initial_guess = Cstar[i_M-1]

        # c) Find optimum
        Vfunc_neg = lambda x,L,Mt,last,par: -value_of_choice(x,L,Mt,last,par)

        res = minimize_scalar(Vfunc_neg, tol = par.tolerance, args = (L,Mt,last,par)
                       , bounds = [0,Mt], method = "bounded")

        V[i_M] = float(-res.fun)
        Cstar[i_M] = float(res.x)

    return(V,Cstar)


def vfi_finite_tasteshocks(par):
    #1) Allocate memory
    Cstar, Vstar = dict(), dict()

    if par.rho < 1.0:
        par.grid_M = np.linspace(0,par.M_max, par.NM)
    else:
        par.grid_M = np.linspace(par.tolerance,par.M_max, par.NM)

    # 2) Last period
    for L in [0,1]:
        Vstar[par.T,L], Cstar[par.T,L] = find_V_tasteshocks(par, L, last = 1)

    # 3) Backwards over time
    for t in reversed(range(1,par.T)): # Start in period T-1

        # a) Create interpolant of next period ressources
        V_plus0 = interpolate.interp1d(par.grid_M, Vstar[t+1,0], kind='linear', fill_value = "extrapolate")
        V_plus1 = interpolate.interp1d(par.grid_M, Vstar[t+1,1], kind='linear', fill_value = "extrapolate")

        par.V_plus_interp = [V_plus0,V_plus1] # Collect the two functions

        # b) Find V for all discrete choices and states
        for L in [0,1]:
            Vstar[t,L], Cstar[t,L] = find_V_tasteshocks(par, L, last = 0)
    return(Vstar,Cstar)


def simulate_path_tasteshock(Vstar,Cstar,par):
    # 1. Allocate
    simM = par.M_ini * np.ones((par.simN, par.T))
    simC = np.ones((par.simN, par.T))
    simL = np.ones((par.simN, par.T))

    # 2. Random numbers
    eps = np.random.uniform(0,1,(par.simN, par.T)) # Uniform numbers, not normal?

    # 3. Simulate

    for t in range(1,par.T+1):
        # a) Values of discrete choices
        V_interp_L0 = interpolate.interp1d(par.grid_M, Vstar[t,0], kind='linear', fill_value = "extrapolate")
        V_L0 = V_interp_L0(simM[:,t-1])

        V_interp_L1 = interpolate.interp1d(par.grid_M, Vstar[t,1], kind='linear', fill_value = "extrapolate")
        V_L1 = V_interp_L1(simM[:,t-1])

        # b) Consumption
        Cstar_interp_L0 = interpolate.interp1d(par.grid_M, Cstar[t,0], kind='linear', fill_value = "extrapolate")
        Cstar_interp_L1 = interpolate.interp1d(par.grid_M, Cstar[t,1], kind='linear', fill_value = "extrapolate")

        prob = logsum(V_L0,V_L1,par.sigma_eps)[1]
        I = eps[:,t-1] < prob[0,:] # I collects the individuals who choose not to work

        simC[I,t-1] = Cstar_interp_L0(simM[I,t-1]) # Consumption for non-working individuals
        simC[I == 0,t-1] = Cstar_interp_L1(simM[I == 0,t-1]) # Consumption for working individuals

        # c) Labor choice
        simL[:,t-1] = I == 0

        # d) Next period
        if t < par.T:
            A = simM[:,t-1] - simC[:,t-1]
            Y = np.exp(np.random.normal(par.mu,par.sigma_zeta,par.simN) * par.sigma_zeta)
            simM[:,t] = par.R * A + Y
    return(simC, simL)
