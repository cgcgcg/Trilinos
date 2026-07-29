// @HEADER
// *****************************************************************************
//        MueLu: A package for multigrid based preconditioning
//
// Copyright 2012 NTESS and the MueLu contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef MUELU_INVERSEAPPROXIMATIONFACTORY_DEF_HPP_
#define MUELU_INVERSEAPPROXIMATIONFACTORY_DEF_HPP_

#include <Xpetra_BlockedCrsMatrix.hpp>
#include <Xpetra_CrsGraph.hpp>
#include <Xpetra_CrsMatrixWrap.hpp>
#include <Xpetra_CrsMatrix.hpp>
#include <Xpetra_VectorFactory.hpp>
#include <Xpetra_MatrixFactory.hpp>
#include <Xpetra_Matrix.hpp>

#include "Kokkos_Sort.hpp"
#include "KokkosBlas1_set.hpp"
#include "KokkosBatched_QR_Decl.hpp"
#include "KokkosBatched_ApplyQ_Decl.hpp"
#include "KokkosBatched_Trsv_Decl.hpp"
#include "KokkosBatched_Util.hpp"

#include "MueLu_Level.hpp"
#include "MueLu_Monitor.hpp"
#include "MueLu_Utilities.hpp"
#include "MueLu_InverseApproximationFactory_decl.hpp"

#include "Ifpack2_SparseApproximateInverse.hpp"

namespace MueLu {

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<const ParameterList> InverseApproximationFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::GetValidParameterList() const {
  RCP<ParameterList> validParamList = rcp(new ParameterList());
  using Magnitude                   = typename Teuchos::ScalarTraits<Scalar>::magnitudeType;

  validParamList->set<RCP<const FactoryBase>>("A", NoFactory::getRCP(), "Matrix to build the approximate inverse on.\n");

  validParamList->set<std::string>("inverse: approximation type", "diagonal", "Method used to approximate the inverse.");
  validParamList->set<Magnitude>("inverse: drop tolerance", 0.0, "Values below this threshold  are dropped from the matrix (or fixed if diagonal fixing is active).");
  validParamList->set<bool>("inverse: fixing", false, "Keep diagonal and fix small entries with 1.0");

  return validParamList;
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void InverseApproximationFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::DeclareInput(Level& currentLevel) const {
  Input(currentLevel, "A");
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void InverseApproximationFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Build(Level& currentLevel) const {
  FactoryMonitor m(*this, "Build", currentLevel);

  using STS       = Teuchos::ScalarTraits<SC>;
  const SC one    = STS::one();
  using Magnitude = typename Teuchos::ScalarTraits<Scalar>::magnitudeType;

  const ParameterList& pL = GetParameterList();
  const bool fixing       = pL.get<bool>("inverse: fixing");

  // check which approximation type to use
  const std::string method = pL.get<std::string>("inverse: approximation type");
  TEUCHOS_TEST_FOR_EXCEPTION(method != "diagonal" && method != "lumping" && method != "sparseapproxinverse", Exceptions::RuntimeError,
                             "MueLu::InverseApproximationFactory::Build: Approximation type can be 'diagonal' or 'lumping' or "
                             "'sparseapproxinverse'.");

  RCP<Matrix> A            = Get<RCP<Matrix>>(currentLevel, "A");
  RCP<BlockedCrsMatrix> bA = Teuchos::rcp_dynamic_cast<BlockedCrsMatrix>(A);
  const bool isBlocked     = (bA == Teuchos::null ? false : true);

  // if blocked operator is used, defaults to A(0,0)
  if (isBlocked) A = bA->getMatrix(0, 0);

  const Magnitude tol = pL.get<Magnitude>("inverse: drop tolerance");
  RCP<Matrix> Ainv    = Teuchos::null;

  if (method == "diagonal") {
    const auto diag = VectorFactory::Build(A->getRangeMap(), true);
    A->getLocalDiagCopy(*diag);
    const RCP<const Vector> D = (!fixing ? Utilities::GetInverse(diag) : Utilities::GetInverse(diag, tol, one));
    Ainv                      = MatrixFactory::Build(D);
  } else if (method == "lumping") {
    const auto diag           = Utilities::GetLumpedMatrixDiagonal(*A);
    const RCP<const Vector> D = (!fixing ? Utilities::GetInverse(diag) : Utilities::GetInverse(diag, tol, one));
    Ainv                      = MatrixFactory::Build(D);
  } else if (method == "sparseapproxinverse") {
    RCP<CrsGraph> sparsityPatternNonConst = Utilities::GetThresholdedGraph(A, tol);
    if (IsPrint(Statistics1)) {
      sparsityPatternNonConst->computeGlobalConstants();
      GetOStream(Statistics1) << "NNZ Graph(A): " << A->getCrsGraph()->getGlobalNumEntries() << " , NNZ Tresholded Graph(A): " << sparsityPatternNonConst->getGlobalNumEntries() << std::endl;
    }
    RCP<const CrsGraph> sparsityPattern = sparsityPatternNonConst;
    RCP<Matrix> pAinv                   = Xpetra::toXpetra(Ifpack2::GetSparseApproximateInverse(*toTpetra(A), toTpetra(sparsityPattern)));
    Ainv                                = Utilities::GetThresholdedMatrix(pAinv, tol, fixing);
    if (IsPrint(Statistics1)) {
      rcp_const_cast<CrsGraph>(Ainv->getCrsGraph())->computeGlobalConstants();
      GetOStream(Statistics1) << "NNZ Ainv: " << pAinv->getGlobalNumEntries() << ", NNZ Tresholded Ainv (parameter: " << tol << "): " << Ainv->getGlobalNumEntries() << std::endl;
    }
  }

  GetOStream(Statistics1) << "Approximate inverse calculated by: " << method << "." << std::endl;
  GetOStream(Statistics1) << "Ainv has " << Ainv->getGlobalNumRows() << "x" << Ainv->getGlobalNumCols() << " rows and columns." << std::endl;

  Set(currentLevel, "Ainv", Ainv);
}

}  // namespace MueLu

#endif /* MUELU_INVERSEAPPROXIMATIONFACTORY_DEF_HPP_ */
