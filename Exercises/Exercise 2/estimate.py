# Estimation
%clear
from Exercises.Exercise2.model import ModelSetup
from Exercises.Exercise2.model import Model as model
par = ModelSetup.setup()
import copy

# %%
par.beta
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
        par = model.create_grids(par)
        sol = model.solve(par)

        #3. Predict (convert normalized output back to normal, i.e. c,m -> C,M)
        sol.C = sol.c * data.P # Remember to place this in period 20
        sol.M = sol.m * data.P

        #4. Calculate errors
        log_C = np.log(sol.C)
        log_C_star = data.logC

        epsilon = log_C - log_C_star

        #5. Calculate log-likelihood
        psi = -0.5 * np.log(2*par.sigma_eta*np.pi) - epsilon^2 * (1/(2*par.sigma_eta^2))
        log_likelihood = sum(psi)
        return log_likelihood

# %%
#Estimate.update_par(par, ['beta', 'theta'], [0.70, 0.50])
