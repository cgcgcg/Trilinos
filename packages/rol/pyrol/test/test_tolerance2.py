from pyrol import getCout, Objective, Problem, Solver
from pyrol.pyrol.Teuchos import ParameterList
from pyrol.unsupported import StdVector_double


class MyObjective(Objective):
    def __init__(self, target, perturbation):
      super().__init__()
      self.target = target
      self.temp = target.clone()
      self.perturbation = perturbation

    def value(self, z, tol):
      self.temp.set(z)
      self.temp.axpy(-1.0, self.target)
      return 0.5*self.temp.norm()**2

    def gradient(self, g, z, tol):
      g.zero()
      g.plus(z)
      g.axpy(-1.0, self.target)
      if abs(self.perturbation) > 0:
          g.axpy(self.perturbation*tol.get(), self.target)

    def hessVec(self, hv, v, z, tol):
      hv.set(v)


doPerturb = True

stream = getCout()
n = 100
target = StdVector_double(n)
target.randomize()
target.print(stream);

objective = MyObjective(target, 1./n if doPerturb else 0.)
z = target.clone()

params = ParameterList()
params['General'] = ParameterList()
params['General']['Output Level'] = 1
params['General']['Inexact Objective Function'] = True
params['General']['Inexact Gradient'] = True
params['Status Test'] = ParameterList()
params['Status Test']['Iteration Limit'] = 50
params['Status Test']['Gradient Tolerance'] = 1e-10
params['Status Test']['Use Relative Tolerances'] = True
params['Step'] = ParameterList()
params['Step']['Trust Region'] = ParameterList()
params['Step']['Trust Region']['Subproblem Solver'] = 'Truncated CG'
params['Step']['Trust Region']['Radius Growing Threshold'] = .5
params['Step']['Trust Region']['Nonmonotone Storage Limit'] = 0
params['Step']['Trust Region']['Initial Radius'] = 1e2

problem = Problem(objective, z)
zCheck = z.clone()
zCheck.setScalar(.5)
# problem.check(True, stream, x0=zCheck, scale=0.001)
solver = Solver(problem, params)
solver.solve(stream)

temp = z.clone()
temp.set(z)
temp.axpy(-1.0, target)
assert temp.norm()**2/z.norm() <= 1e-8, temp.norm()**2/z.norm()
