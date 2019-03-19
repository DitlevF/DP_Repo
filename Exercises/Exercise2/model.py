import numpy as np
from scipy import interpolate
from scipy.optimize import minimize_scalar
from numpy.polynomial.hermite import hermgauss
import matplotlib.pyplot as plt

from Exercises.Exercise2.struct import Struct
from Exercises.Exercise2.utils import hermgauss_lognorm

import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

# %%
class ModelSetup():

    @staticmethod
    def setup():
        par = Struct()

        #demographics
        par.T = 200
        par.TR = par.T #retirement age (if equal to t=> no retirement age)
        par.age_min=25

        #preferences
        par.rho = 2
        par.beta = 0.96

        #income  parameters
        par.G = 1.03
        par.sigma_xi = 0.1
        par.sigma_psi = 0.1

        #income shock
        par.low_p = 0.005
        par.low_val = 0

        #life-cycle
        par.L = np.ones(par.T) # if ones => no life cycle

        #saving and borrowing
        par.R = 1.04
        par.lambda_ = 0.0

        #numerical integration and create_grids
        par.a_max = 20.0
        par.a_phi = 1.1

        #number of elements
        par.Nxi = 8 #n quadrature points for xsi
        par.Npsi = 8 #n q-points for psi
        par.Na = 500 #grid points for a

        #simulation
        par.sim_mini = 2.5
        par.simN = 500000
        par.simT = 100
        par.simlifecycle = 0

        return par

class Model():

    @staticmethod
    def create_grids(par):

        # check parameters:
        assert par.rho >= 0
        assert par.lambda_ >= 0

        if par.sigma_xi == 0 and par.sigma_psi == 0 and par.low_p == 0:
            par.model = 'pf'
        else:
            par.model = 'bs'

        # shocks
        par.psi, par.psi_w = hermgauss_lognorm(par.Npsi, par.sigma_psi)
        par.xi, par.xi_w = hermgauss_lognorm(par.Nxi, par.sigma_xi)
        # add income shock
        if par.low_p > 0:
            _lv = np.array(par.low_val) # must be array
            _xi = (par.xi - par.low_p*par.low_val) / (1-par.low_p)
            par.xi = np.append(_lv, _xi)

            _lvp = np.array(par.low_p)
            _xi_w = (1 - par.low_p) * par.xi_w
            par.xi_w = np.append(_lvp, _xi_w)

        # vectorize all (wait with implementation)
        par.xi_vec,par.psi_vec = np.meshgrid(par.xi, par.psi)
        par.xi_w_vec, par.psi_w_vec = np.meshgrid(par.xi_w, par.psi_w)

        # Make into vector format
        par.xi_vec, par.psi_vec = par.xi_vec.flatten(), par.psi_vec.flatten()
        par.xi_w_vec, par.psi_w_vec = par.xi_w_vec.flatten(), par.psi_w_vec.flatten()

        par.w = par.xi_w_vec * par.psi_w_vec
        assert abs(1 - sum(par.w)) < 10**-4 # Check that it sums to one.

        # Count number of shock nodes
        par.Nshocks = np.size(par.w)
        if par.lambda_ == 0:
            par.a_min = np.zeros(par.T)
        else:
            raise NotImplementedError("borrowing is not allowed in  this model")

        # end-of-period assets
        # does not implement nonlinspace (Johan said not important)
        par.grid_a = dict()
        for t in range(par.T):
            par.grid_a[t] = np.linspace(par.a_min[t], par.a_max, par.Na)

        # conditions
        # not implemented

        return par

    @staticmethod
    def print_and_check_parameters():
        pass

    @staticmethod
    def utility(c: float, par):
        # in given implementation c is a float. In orig it can be a vector
        u = ( c**(1-par.rho) ) /(1-par.rho)
        return u

    @staticmethod
    def marg_utility(c, par):
        u = c**(-par.rho)
        return u

    @staticmethod
    def inv_marg_utility(u, par):
        c = u**(-1/par.rho)
        return c

    @classmethod
    def EGM(cls, sol,t,c_plus_interp,par):
        if t != 1:
            print(t, end = "\r")
        else:
            print("done")

        sol.m[t] =  np.ones(par.Na) * np.nan # par.Na is the number of grid points of assets a
        sol.c[t] = np.ones(par.Na) * np.nan

        for i_a in range(0,par.Na):
            a_i = par.grid_a[t][i_a]
            #a) Future m and c (vector) Step 1 in Algorithm on Slide 46, Lecture 7-8
            marg_u = 0 # Initialize
            for i_psi in range(1,par.Npsi): # Integrate over psi
                for j_xi in range(1,par.Nxi): # Integrate over xi
                    m_fut = par.R/(par.G * par.L[t] * par.psi[i_psi]) * a_i + par.xi[j_xi]
                    d_fut = par.G * par.L[t] * par.psi[i_psi] * c_plus_interp(m_fut)
                    Eu_fut = par.xi_w[j_xi] * par.psi_w[i_psi] * cls.marg_utility(d_fut,par) # Gauss Hermite weighting
                    marg_u += Eu_fut # Sum the weighted components

            c_i = cls.inv_marg_utility(par.beta * par.R * marg_u, par)
            m_i = c_i + a_i # Endogenous grid method
            # Store results
            sol.c[t][i_a] = c_i
            sol.m[t][i_a] = m_i
        return sol.c[t], sol.m[t]


    @classmethod
    def solve(cls, par):
        # 1. allocation solution struct and cells
        sol = Struct()
        sol.m = dict()
        sol.c = dict()

        # 2. last period (= consume all)

        sol.m[par.T] = np.linspace(0, par.a_max, par.Na)
        sol.c[par.T] = np.linspace(0, par.a_max, par.Na)

        # 2 Before last Period
        for t in reversed(range(1,par.T)): # Start in period T-1

            #a) Interpolant
            par.c_plus_interp = interpolate.interp1d(sol.m[t+1], sol.c[t+1], kind='linear', fill_value = "extrapolate")

            #b) EGM
            sol_c, sol_m = cls.EGM(sol, t, par.c_plus_interp, par)

            #c) Add zero Consumption
            sol.m[t] = np.append(par.a_min[t], sol_m)
            sol.c[t] = np.append(0, sol_c)

        return(sol)

    @staticmethod
    def simulate(par,sol):
        sim = Struct()

        #1) Allocate
        sim.m = np.empty((par.simN,par.simT)) * np.nan
        sim.c = np.empty((par.simN,par.simT)) * np.nan
        sim.a = np.empty((par.simN,par.simT)) * np.nan
        sim.p = np.empty((par.simN,par.simT)) * np.nan
        sim.y = np.empty((par.simN,par.simT)) * np.nan

        #2) Shocks
        shocki = np.random.choice(par.Nshocks, par.simN * par.simT, True, par.w)
        shocki = np.reshape(shocki, (par.simN, par.simT)) # Reshape into matrix format
        sim.psi = par.psi_vec[shocki] # Allocate the shocks to the different indices in the shocki matrix
        sim.xi = par.xi_vec[shocki]

        #3) Initialize values
        sim.m[:,0] = par.sim_mini
        sim.p[:,0] = 0

        #4) Simulation
        for t in range(1,par.T):
            if t < par.T - 1:
                print(t, end = "\r")
            else:
                print ('done') # Done

            if par.simlifecycle == 0: # Infinite horizon
                c_interp = interpolate.interp1d(sol.m[1], sol.c[1], kind='linear', fill_value = "extrapolate")
            else:
                c_interp = interpolate.interp1d(sol.m[t], sol.c[t], kind='linear', fill_value = "extrapolate")

            sim.c[:,t-1] = c_interp(sim.m[:,t-1])
            sim.a[:,t-1] = sim.m[:,t-1] - sim.c[:,t-1]
            # print(sim.c[:,t-1])

            if t < par.simT:
                if t > par.TR: # Retired
                    sim.m[:,t] = par.R * sim.a[:,t-1] / (par.G * par.L[t-1]) + 1
                    sim.p[:,t] = np.log(par.G) + np.log(par.L[t-1]) + sim.p[:,t-1]
                    sim.y[:,t] = sim.p[:,t]
                else:
                    sim.m[:,t] = par.R * sim.a[:,t-1] / (par.G * par.L[t-1] * sim.psi[:,t]) + sim.xi[:,t]
                    sim.p[:,t] = np.log(par.G) + np.log(par.L[t-1]) + sim.p[:,t-1] + np.log(sim.psi[:,t])
                    sim.y[:,t] = sim.p[:,t] + np.log(sim.xi[:,t])

        # 5. Re-normalize
        sim.P = np.exp(sim.p)
        sim.Y = np.exp(sim.y)
        sim.M = sim.m * sim.P
        sim.C = sim.c * sim.P
        sim.A = sim.a * sim.P

        return sim

# %%

# %%

par = ModelSetup.setup()
par.prefix = 'lifecycle'

# 1. life-cycle settings
par.T = 90-par.age_min;
par.TR = 65-par.age_min;
par.simT = par.T;
par.simlifecycle = 1;

# 2. income profile
end = len(par.L)
par.L[0:par.TR] = np.linspace(1,1/(par.G),par.TR);
par.L[par.TR] = 0.90;
par.L[par.TR:end] = par.L[par.TR:end]/par.G;

# 3. solve and simulate
par = Model.create_grids(par);
sol = Model.solve(par);
sim = Model.simulate(par,sol)
