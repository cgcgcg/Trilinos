#!/usr/env python
# @HEADER
# *****************************************************************************
#          PyTrilinos2: Automatic Python Interfaces to Trilinos Packages
#
# Copyright 2022 NTESS and the PyTrilinos2 contributors.
# SPDX-License-Identifier: BSD-3-Clause
# *****************************************************************************
# @HEADER

from mpi4py import MPI
from argparse import ArgumentParser
try:
    # MPI, Timers, ParameterList
    from PyTrilinos2 import Teuchos
    # Linear algebra
    from PyTrilinos2 import Tpetra
    # Linear algebra interfaces used by Stratimikos
    from PyTrilinos2 import Thyra
    # Unified solver & preconditioner interface
    from PyTrilinos2 import Stratimikos
    # Finite Elements
    from PyTrilinos2 import panzer
    from PyTrilinos2 import panzer_stk
    from PyTrilinos2.getTpetraTypeName import getTypeName, getDefaultNodeType
except ImportError:
    print("\nFailed to import PyTrilinos2. Consider setting the Python load path in your environment with\n export PYTHONPATH=${TRILINOS_BUILD_DIR}/packages/PyTrilinos2:${PYTHONPATH}\nwhere TRILINOS_BUILD_DIR is the build directory of your Trilinos build.\n")
    raise


class PoissonEquationSet(panzer.EquationSet_DefaultImpl_panzer_Traits_Residual_t):
    def __init__(self, params, default_integration_order, cell_data, global_data, build_transient_support):
        super().__init__(params, default_integration_order, cell_data, global_data, build_transient_support)

        # Set default values and validate
        valid_parameters = Teuchos.ParameterList()
        valid_parameters['Model ID'] = ""
        valid_parameters["Basis Type"] = "HGrad"
        valid_parameters["Basis Order"] = 1
        valid_parameters["Integration Order"] = -1
        params.validateParametersAndSetDefaults(valid_parameters)

        # Panzer uses strings to match fields. In this section we define the
        # name of the fields provided by this equation set. This is a bit strange
        # in that this is not the fields necessarily required by this equation set.
        # For instance for the momentum equations in Navier-Stokes only the velocity
        # fields are added, the pressure field is added by continuity.
        #
        # In this case "TEMPERATURE" is the lone field.  We also name the gradient
        # for this field. These names automatically generate evaluators for "TEMPERATURE"
        # and "GRAD_TEMPERATURE" gathering the basis coefficients of "TEMPERATURE" and
        # the values of the TEMPERATURE and GRAD_TEMPERATURE fields at quadrature points.
        #
        # After all the equation set evaluators are added to a given field manager, the
        # panzer code adds in appropriate scatter evaluators to distribute the
        # entries into the residual and the Jacobian operator. These operators will be
        # "required" by the field manager and will serve as roots of evaluation tree.
        # The leaves of this tree will include the gather evaluators whose job it is to
        # gather the solution from a vector.

        # Assemble DOF names and Residual names
        self.addDOF("TEMPERATURE", params["Basis Type"], params["Basis Order"], params["Integration Order"])
        self.addDOFGrad("TEMPERATURE")
        if self.buildTransientSupport():
            self.addDOFTimeDerivative("TEMPERATURE")

        self.addClosureModel(params["Model ID"])

        self.setupDOFs()

    def buildAndRegisterEquationSetEvaluators(self, fm, field_library, user_data):
        # We set up the weak form in residual form
        #
        # \int v * DXDT_TEMPERATURE + (\nabla v) \cdot (GRAD_TEMPERATURE) - v * SOURCE_TEMPERATURE
        #
        # where
        #  v is a test function
        #  TEMPERATURE is the unknown solution
        #  DXDT_TEMPERATURE is its time derivative
        #  GRAD_TEMPERATURE is its gradient
        # SOURCE_TEMPERATURE is the forcing term that will be defined via the closure model

        residualName = "RESIDUAL_TEMPERATURE"
        fieldName = "TEMPERATURE"
        gradientFieldName = "GRAD_"+fieldName
        timeDerivativeFieldName = "DXT_"+fieldName
        forcingFieldName = "SOURCE_TEMPERATURE"

        ir = self.getIntRuleForDOF(fieldName)
        basis = self.getBasisIRLayoutForDOF(fieldName)

        # Transient Operator:
        # \int v * DXDT_TEMPERATURE
        if self.buildTransientSupport():
            evaluator = panzer.Integrator_BasisTimesScalar(panzer.EvaluatorStyle.CONTRIBUTES, resName=residualName, valName=timeDerivativeFieldName, ir=ir, multiplier=1.0)
            self.registerEvaluator(fm, evaluator)

        # Diffusion Operator:
        # \int (\nabla v) \cdot (GRAD_TEMPERATURE)
        thermal_conductivity = 1.0
        pl = Teuchos.ParameterList("Diffusion Residual")
        pl["Residual Name"] = residualName
        pl["Flux Name"] = gradientFieldName
        pl["IR"] = ir
        pl["Multiplier"] = thermal_conductivity
        evaluator = panzer.Integrator_GradBasisDotVector(pl)
        self.registerEvaluator(fm, evaluator)

        # Source Operator
        # - \int v * SOURCE_TEMPERATURE
        evaluator = panzer.Integrator_BasisTimesScalar(panzer.EvaluatorStyle.CONTRIBUTES, resName=residualName, valName=forcingFieldName, ir=ir, multiplier=-1.0)
        self.registerEvaluator(fm, evaluator)


class EquationSetFactory(panzer.EquationSetFactory):
    def buildEquationSet(params, default_integration_order, cell_data, global_data, build_transient_support):
        if params["Type"] == "Poisson":
            eqset = PoissonEquationSet(params, default_integration_order, cell_data, global_data, build_transient_support)
        else:
            raise NotImplementedError(params["Type"])
        # eqset.buildObjects(builder)
        return eqset


# MPI communicator
comm = MPI.COMM_WORLD

#
eqset_factory = EquationSetFactory()

# construct a mesh
mesh_factory = panzer_stk.SquareTriMeshFactory()
mesh_params = Teuchos.ParameterList()
mesh_params['X Blocks'] = 1
mesh_params['Y Blocks'] = 1
mesh_params['X Elements'] = 15
mesh_params['Y Elements'] = 15
mesh_factory.setParameterList(mesh_params)
print(help(mesh_factory))
mesh = mesh_factory.buildUncommitedMesh(comm)
mesh_block = "eblock-0_0"

workset_size = 2000
build_transient_support = False

physics_block_params = Teuchos.ParameterList()

volume_cell_data = panzer.CellData(workset_size, mesh.getCellTopology(mesh_block))

gd = panzer.createGlobalData()

default_integration_order = 1

pb = panzer.PhysicsBlock(ipb, mesh_block, default_integration_order, volume_cell_data, eqset_factory, gd, build_transient_support)
