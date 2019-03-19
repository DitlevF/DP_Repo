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

from Functions.struct import Struct

def solve(par):
    # 1. allocation solution struct and cells
    sol = Struct()
    sol.m = dict()
    sol.c = dict()

    # 2. last period (= consume all)

    sol.m[par.T] = np.linspace(0,par.a_max,par.Na)
    sol.m[par.T] = np.linspace(0,par.a_max,par.Na)

    # 2 Before last Period
    for t in reversed(range(1,par.T)): # Start in period T-1

        #a) Interpolant
        par.c_plus_interp = interpolate.interp1d(sol.m[t+1], sol.c[t+1], kind='linear', fill_value = "extrapolate")

        #b) EGM
        sol = EGM(sol, t, c_plus_interp, par)

        #c) Add zero Consumption
        sol.m[t] = np.append(par.a_min[t], sol.m[t])
        sol.c[t] = np.append(0, sol.m[t])

    return(sol)

def EGM(sol,t,c_plus_interp,par):

    sol.m[t] =  np.ones(par.Na) * np.nan # par.Na is the number of grid points of assets a
    sol.c[t] = np.ones(par.Na) * np.nan

    for i_a in range(0,par.Na):
        a_i = par.grid_a[t][i_a]
        #a) Future m and c (vector) Step 1 in Algorithm on Slide 46, Lecture 7-8
        marg_u = 0 # Initialize
        for i_psi in range(1,par.Npsi): # Integrate over psi
            for j_xi in range(1,par.Nxi): # Integrate over xi
                m_fut = par.R/(par.G * par.L[t] * par.psi[i_psi] * a_i + par.xi[j_xi])
                d_fut = par.G * par.L[t] * par.psi[i_psi] * c_plus_interp(m_fut)
                Eu_fut = par.xi_w[j_xi] * par.psi_w[i_psi] * model.marg_utility(d_fut,par) # Gauss Hermite weighting
                marg_u += Eu_fut # Sum the weighted components
        c_i = model.inv_marg_utility(par.beta * par.R * marg_u, par)
        m_i = c_i + a_i # Endogenous grid method
        # Store results
        sol.c[t][i_a] = c_i
        sol.m[t][i_a] = m_i
    return sol.c[t], sol.m[t]
