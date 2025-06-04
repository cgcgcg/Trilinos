#include "ROL_Objective.hpp"
#include "ROL_Solver.hpp"
#include "ROL_Problem.hpp"
#include "ROL_StdVector.hpp"


template <class Real>
class MyObjective : public ROL::Objective<Real> {

public:

  MyObjective(Teuchos::RCP<ROL::Vector<Real> > &target_, Real perturbation_) {
    target = target_;
    temp = target->clone();
    perturbation = perturbation_;
  }

  Real value(const ROL::Vector<Real> &x, ROL::Tolerance<Real> &tol) {
    temp->set(x);
    temp->axpy(-1.0, *target);
    return 0.5*temp->norm()*temp->norm();
  }

  void gradient( ROL::Vector<Real> &g, const ROL::Vector<Real> &z, ROL::Tolerance<Real> &tol ) {
    g.zero();
    g.plus(z);
    g.axpy(-1.0, *target);
    if (abs(perturbation)>0)
      g.axpy(perturbation*tol.get(), *target);
  }

  void hessVec( ROL::Vector<Real> &hv, const ROL::Vector<Real> &v, const ROL::Vector<Real> &x, ROL::Tolerance<Real> &tol ) {
    hv.set(v);
  }

private:
  Teuchos::RCP<ROL::Vector<Real> > target;
  Teuchos::RCP<ROL::Vector<Real> > temp;
  Real perturbation;
};


int main(int argc, char** argv) {

  bool doPerturb = true;

  int n = 100;
  Teuchos::RCP<ROL::Vector<double>> target = Teuchos::rcp(new ROL::StdVector<double>(n));

  auto z = target->clone();

  target->randomize();

  target->print(std::cout);

  Teuchos::ParameterList params;

  params.sublist("General").set("Output Level", 1);
  params.sublist("General").set("Inexact Objective Function", true);
  params.sublist("General").set("Inexact Gradient", true);
  params.sublist("Status Test").set("Iteration Limit", 50);
  params.sublist("Status Test").set("Gradient Tolerance", 1e-10);
  params.sublist("Status Test").set("Use Relative Tolerances", true);
  params.sublist("Step").sublist("Trust Region").set("Subproblem Solver", "Truncated CG");
  params.sublist("Step").sublist("Trust Region").set("Radius Growing Threshold", 0.5);
  params.sublist("Step").sublist("Trust Region").set("Nonmonotone Storage Limit", 0);
  params.sublist("Step").sublist("Trust Region").set("Initial Radius", 1e2);

  Teuchos::RCP<ROL::Objective<double>> objective = Teuchos::rcp(new MyObjective<double>(target, doPerturb ? 1.0 / n : 0.0));

  auto problem = Teuchos::rcp(new ROL::Problem(objective, z));
  auto zCheck = z->clone();
  zCheck->setScalar(.5);
  // problem->check(True, std::cout, zCheck, 0.001);
  auto solver = ROL::Solver(problem, params);
  solver.solve(std::cout);

  auto temp = z->clone();
  temp->set(*z);
  temp->axpy(-1.0, *target);

}
