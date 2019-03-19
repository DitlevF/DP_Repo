
from Exercises.Exercise2.struct import Struct
from Exercises.Exercise2.utils import hermgauss_lognorm
import numpy as np
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
