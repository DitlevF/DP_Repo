# Estimation
import model

def log_likelihood(theta, est_par, par, data):

    #1. Update parameters
    par = updatepar(par, est_par,theta)

    #2. Solve Model
    par = model.create_grids(par)
    sol = model.solve(par)

    #3. Predict (convert normalized output back to normal, i.e. c -> C)


    #4. Calculate errors

    #5. Calculate log-likelihood
