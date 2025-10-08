// @HEADER
// *****************************************************************************
//           Galeri: Finite Element and Matrix Generation Package
//
// Copyright 2006 ETHZ/NTESS and the Galeri contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER
/*
  Direct translation of parts of Galeri matrix generator.
*/
#ifndef GALERI_XPETRAMATRIXFACTORY_DECL_HPP
#define GALERI_XPETRAMATRIXFACTORY_DECL_HPP

#include "Teuchos_ParameterList.hpp"
#include "Galeri_StencilProblems.hpp"
#include "Galeri_Elasticity2DProblem.hpp"
#include "Galeri_Elasticity3DProblem.hpp"

namespace Galeri::Xpetra {

using Teuchos::RCP;

template <typename Scalar, typename LocalOrdinal, typename GlobalOrdinal, typename Map, typename Matrix, typename MultiVector>
Teuchos::RCP<Problem<Map, Matrix, MultiVector> > BuildProblem(const std::string& MatrixType, const Teuchos::RCP<const Map>& map, Teuchos::ParameterList& list);

} // namespace Galeri::Xpetra



#endif  // ifndef GALERI_XPETRAMATRIXFACTORY_DECL_HPP
