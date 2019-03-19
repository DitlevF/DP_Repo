### Exercise Set 1, 5+ ### (same as Jupyter notebook, just copied over here)

# %% Import libraries
import numpy as np
import os
from numpy.polynomial.hermite import hermgauss
from scipy import interpolate
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

# %% Import .py files

from Functions.model_setup import par
from Functions import VFI as vfi

# %% Solve model_setup
# Get solution path

##### Possible changes to parameters of the model #####
# par.sigma_zeta = 0 # If we do this, all consumption paths are the same for the N individuals
# par.NM = 100
#######################################################

Cstar, Vstar = vfi.vfi_finite(par)

# %%
# Simulate consumption path
simC = vfi.simulate_path(Cstar,par)
simC


# %%
# Check that average of Ct is increasing over time
simC_mean = [np.mean(simC[:,x]) for x in range(par.T)]

plt.figure()
x = np.linspace(1,par.T,par.T)
plt.plot(x,simC_mean)
plt.title('Consumption Path'); plt.ylabel('Avg. Consumption'); plt.xlabel('Time'); plt.xticks(range(1,11))
plt.show()

# %%

# Marginal utility function
marg_u = lambda C,par: C**(-par.rho)

# 1) Interpolants
Cstar_interp = interpolate.interp1d(par.grid_M, Cstar[1], kind='linear', fill_value = "extrapolate")
Cstar_plus_interp = interpolate.interp1d(par.grid_M, Cstar[2], kind='linear', fill_value = "extrapolate")

# 2) Consumption today and tomorrow
C = Cstar_interp(par.M_ini)
A = par.M_ini - C
C_plus = Cstar_plus_interp(A+par.R * par.Y) # Notice, this is an array due to Gauss-Hermite

# 3.1) Euler errors
euler_error = abs(marg_u(C,par) - par.beta * par.R * sum(par.w * marg_u(C_plus,par)))
# 3.2) Normalized euler errors
norm_euler_error = np.log10(euler_error/C)

# 4) Print
print('The normalized euler error is', norm_euler_error)
