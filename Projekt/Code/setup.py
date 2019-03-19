class Model():

    def __init__(self, par, state):
        self.par = par
        self.state = state

    # Functions from the paper

    # Utility function, CRRA with equivalence scale adjuting
    def utility(self, c, t):
        u = self.par.n[e,t] * ( (c/self.par.n[e,t])**(1-self.par.rho) ) /(1-self.par.rho)
        return u

    # Evolution of knowledge
    def f_next(self, f, i):
        f_next = (1-self.par.delta) * f + i
        return f_next

    # Evolution of asset
    ## Remember to add stochastic element to the return factor evolution, p. 446
    def a_next(self, f, a, y, oop, i, kappa):
        R_tilde = self.par.r_bar + self.r(f) + 1 # Excess return factor.
        R_tilde_next = (1-kappa) * self.par.R_bar + kappa * R_tilde
        costs_i = self.pi_cost(i)

        return R_tilde_next * (a + y - oop - costs_i)

    # Cash-on-hand
    def cash_on_hand(self, a, y, oop):
        return a + y - oop

    # Government transfer
    def tr(self, x):
        return max(self.par.cmin - x, 0)

    # Return function of financial literacy assumed linear (p. 450)
    def r(self, f):
        # Calculate slope
        slope = (self.par.r_max - self.par.r_min)/(self.par.f_max - self.par.f_min)
        return self.par.r_min + slope * f

    # Cost function for financial knowledge (p. 451, bottom). Combining both fixed and variable
    @staticmethod
    def pi_cost(i):
        fixed = 750 * (i > 0)
        variable = 50*(i**1.75)
        return fixed + variable
