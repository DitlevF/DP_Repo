# Estimation
from Exercises.Exercise2.model_setup import ModelSetup
from Exercises.Exercise2.model import Model
from Exercises.Exercise2.struct import Struct
from scipy import interpolate

par = ModelSetup.setup()
import numpy as np

# %%
# for i in range(len(self.est_par)):
#     setattr(self.model.par, self.est_par[i], theta[i])

# %%

class Estimate():
    @staticmethod
    def update_par(par, est_par, theta):
        for i in range(len(est_par)):
            setattr(par, est_par[i], theta[i])
        return par

    @classmethod
    def log_likelihood(cls, theta, est_par, par, data):

        #1. Update parameters
        par = cls.update_par(par, est_par,theta)

        #2. Solve Model
        par = Model.create_grids(par)
        sol = Model.solve(par)

        # Period t = 20
        c,m,P = sol.c[data.t], sol.m[data.t], data.P
        #3. Predict (convert normalized output back to normal, i.e. c,m -> C,M)

        # Interpolate solution

        c_interp = interpolate.interp1d(m, c, kind='linear', fill_value = "extrapolate")

        c_predict = c_interp(data.m) * P

        #4. Calculate errors
        log_C = np.log(c_predict)
        log_C_data = data.logC

        # Remove -Inf values
        log_C[log_C == -np.Inf] = 0

        epsilon = log_C - log_C_data

        #5. Calculate log-likelihood
        psi = -0.5 * np.log(2*par.sigma_eta*np.pi) - epsilon**2 * (1/(2*par.sigma_eta**2))
        log_likelihood = np.sum(psi)
        return log_likelihood

    @classmethod
    def maximum_likelihood(cls, par, est_par, theta0, data, do_stderr):
        assert np.size(est_par) == np.size(theta0), "Number of parameters and initial values do not match"

        #1. Estimation


# %%
