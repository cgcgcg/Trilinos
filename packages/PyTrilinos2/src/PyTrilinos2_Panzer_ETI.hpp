// @HEADER
// *****************************************************************************
//          PyTrilinos2: Automatic Python Interfaces to Trilinos Packages
//
// Copyright 2022 NTESS and the PyTrilinos2 contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef PYTRILINOS2_PANZER_ETI_HPP
#define PYTRILINOS2_PANZER_ETI_HPP

#include "Panzer_OrientationsInterface.hpp"

#include "Panzer_STK_SquareQuadMeshFactory.hpp"
#include "Panzer_STK_SquareTriMeshFactory.hpp"

#include "Panzer_EquationSet_Factory.hpp"
#include "Panzer_EquationSet_DefaultImpl_decl.hpp"


#define BINDER_PANZER_INSTANT() \
  template class panzer::EquationSet_DefaultImpl<panzer::Traits::Residual>; \
  template class panzer::EquationSet_DefaultImpl<panzer::Traits::Jacobian>; \
  template class panzer::EquationSet_DefaultImpl<panzer::Traits::Tangent>;


namespace panzer {

    template <typename T>
    void initiate(T) {};

  BINDER_PANZER_INSTANT()

}

#endif // PYTRILINOS2_PANZER_ETI
