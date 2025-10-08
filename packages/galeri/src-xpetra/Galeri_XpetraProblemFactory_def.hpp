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
#ifndef GALERI_XPETRAPROBLEMFACTORY_DEF_HPP
#define GALERI_XPETRAPROBLEMFACTORY_DEF_HPP

#include "Galeri_XpetraProblemFactory_decl.hpp"

#include "Teuchos_Assert.hpp"

namespace Galeri::Xpetra {

using Teuchos::RCP;

template <typename Scalar, typename LocalOrdinal, typename GlobalOrdinal, typename Map, typename Matrix, typename MultiVector>
RCP<Problem<Map, Matrix, MultiVector> > BuildProblem(const std::string& MatrixType, const RCP<const Map>& map, Teuchos::ParameterList& list) {
  RCP<Problem<Map, Matrix, MultiVector> > P;

  if (MatrixType == "Laplace1D")
    P.reset(new Laplace1DProblem<Scalar, LocalOrdinal, GlobalOrdinal, Map, Matrix, MultiVector>(list, map));
  else if (MatrixType == "Laplace2D")
    P.reset(new Laplace2DProblem<Scalar, LocalOrdinal, GlobalOrdinal, Map, Matrix, MultiVector>(list, map));
  else if (MatrixType == "Star2D")
    P.reset(new Star2DProblem<Scalar, LocalOrdinal, GlobalOrdinal, Map, Matrix, MultiVector>(list, map));
  else if (MatrixType == "BigStar2D")
    P.reset(new BigStar2DProblem<Scalar, LocalOrdinal, GlobalOrdinal, Map, Matrix, MultiVector>(list, map));
  else if (MatrixType == "AnisotropicDiffusion")
    P.reset(new AnisotropicDiffusion2DProblem<Scalar, LocalOrdinal, GlobalOrdinal, Map, Matrix, MultiVector>(list, map));
  else if (MatrixType == "Laplace3D")
    P.reset(new Laplace3DProblem<Scalar, LocalOrdinal, GlobalOrdinal, Map, Matrix, MultiVector>(list, map));
  else if (MatrixType == "Brick3D")
    P.reset(new Brick3DProblem<Scalar, LocalOrdinal, GlobalOrdinal, Map, Matrix, MultiVector>(list, map));
  else if (MatrixType == "Elasticity2D")
    P.reset(new Elasticity2DProblem<Scalar, LocalOrdinal, GlobalOrdinal, Map, Matrix, MultiVector>(list, map));
  else if (MatrixType == "Elasticity3D")
    P.reset(new Elasticity3DProblem<Scalar, LocalOrdinal, GlobalOrdinal, Map, Matrix, MultiVector>(list, map));
  else if (MatrixType == "Identity")
    P.reset(new IdentityProblem<Scalar, LocalOrdinal, GlobalOrdinal, Map, Matrix, MultiVector>(list, map));
  else if (MatrixType == "Recirc2D")
    P.reset(new Recirc2DProblem<Scalar, LocalOrdinal, GlobalOrdinal, Map, Matrix, MultiVector>(list, map));
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                               "`MatrixType' has incorrect value (" << MatrixType << ") in input to function CreateCrsMatrix()."
                                                                    << "Check the documentation for a list of valid choices");

  P->setObjectLabel(MatrixType);

  return P;
}

} // namespace Galeri::Xpetra


#define GALERI_XPETRAPROBLEMFACTORY_INSTANT_XPETRA(S, LO, GO, N) \
  template \
  Teuchos::RCP<Galeri::Xpetra::Problem<::Xpetra::Map<LO, GO, N>, ::Xpetra::CrsMatrixWrap<S, LO, GO, N>, ::Xpetra::MultiVector<S, LO, GO, N> >> \
  Galeri::Xpetra::BuildProblem<S, LO, GO, ::Xpetra::Map<LO, GO, N>, ::Xpetra::CrsMatrixWrap<S, LO, GO, N>, ::Xpetra::MultiVector<S, LO, GO, N >> \
  (const std::string&, const Teuchos::RCP<const ::Xpetra::Map<LO, GO, N>>&, Teuchos::ParameterList&);

#define GALERI_XPETRAPROBLEMFACTORY_INSTANT_TPETRA(S, LO, GO, N) \
  template \
  Teuchos::RCP<Galeri::Xpetra::Problem<Tpetra::Map<LO, GO, N>, Tpetra::CrsMatrix<S, LO, GO, N>, Tpetra::MultiVector<S, LO, GO, N>> > \
  Galeri::Xpetra::BuildProblem<S, LO, GO, Tpetra::Map<LO, GO, N>, Tpetra::CrsMatrix<S, LO, GO, N>, Tpetra::MultiVector<S, LO, GO, N >> \
  (const std::string&, const Teuchos::RCP<const Tpetra::Map<LO, GO, N>>&, Teuchos::ParameterList&);


#define GALERI_XPETRAPROBLEMFACTORY_INSTANT(S, LO, GO, N) \
  GALERI_XPETRAPROBLEMFACTORY_INSTANT_XPETRA(S, LO, GO, N) \
  GALERI_XPETRAPROBLEMFACTORY_INSTANT_TPETRA(S, LO, GO, N)



#endif  // ifndef GALERI_XPETRAPROBLEMFACTORY_DEF_HPP
