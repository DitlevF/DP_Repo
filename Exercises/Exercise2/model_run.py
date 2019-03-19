import os
os.getcwd()

from Exercises.Exercise2.model_setup import ModelSetup
from Exercises.Exercise2.model import Model
from Exercises.Exercise2.struct import Struct
from Exercises.Exercise2.estimate import Estimate

import numpy as np

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
par.Na = 100
par = Model.create_grids(par);
sol = Model.solve(par);
sim = Model.simulate(par,sol)

# 4. Create data for Simulation
data = Struct()
par.sigma_eta = 0.1
data.t = 20
data.M = sim.M[:,data.t]
data.m = sim.m[:,data.t]
data.P = sim.P[:,data.t]
data.logC = np.log(sim.C[:,data.t]) + np.sqrt(par.sigma_eta) * np.random.normal(size = par.simN)
# %%
# 5. Estimate
log_like = Estimate.log_likelihood(theta = [0.94], est_par = ['beta'], par = par, data = data)
log_like
