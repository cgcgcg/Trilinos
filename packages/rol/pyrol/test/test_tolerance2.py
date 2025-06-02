from pyrol import getCout, Objective, Problem, Solver
from pyrol.vectors import NumPyVector
from pyrol.pyrol.Teuchos import ParameterList
import numpy as np


class MyObjective(Objective):
    def __init__(self):
        super().__init__()

    def value(self, z, tol):
        return 0.5*z[0]**2

    def gradient(self, g, z, tol):
        g.zero()
        g.plus(z)

    def hessVec(self, hv, v, z, tol):
        hv.assign(v)


objective = MyObjective()
z = NumPyVector(np.ones(1))

params = ParameterList()
params['General'] = ParameterList()
params['General']['Output Level'] = 1
params['General']['Inexact Objective Function'] = True
params['General']['Inexact Gradient'] = True
params['Status Test'] = ParameterList()
params['Status Test']['Iteration Limit'] = 20
params['Status Test']['Gradient Tolerance'] = 1e-10
params['Status Test']['Use Relative Tolerances'] = True
params['Step'] = ParameterList()
params['Step']['Trust Region'] = ParameterList()
params['Step']['Trust Region']['Subproblem Solver'] = 'Truncated CG'
params['Step']['Trust Region']['Radius Growing Threshold'] = .5
params['Step']['Trust Region']['Nonmonotone Storage Limit'] = 0
params['Step']['Trust Region']['Initial Radius'] = 1e2

problem = Problem(objective, z)
solver = Solver(problem, params)
stream = getCout()
solver.solve(stream)

assert z[0] == 0., z[0]
