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
