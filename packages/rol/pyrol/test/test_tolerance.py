from pyrol import Objective
from pyrol.pyrol.ROL import Tolerance_double_t


class MyObjective(Objective):
    def __init__(self):
        super().__init__()

    def value(self, z, tol):
        tol.assign(0.5*tol.get())
        return 0.


obj = MyObjective()
z = None
tol = Tolerance_double_t(1e-3)
obj.value(z, tol)
assert tol.get() == 5e-4, tol.get()
