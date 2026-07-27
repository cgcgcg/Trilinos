// @HEADER
// *****************************************************************************
//        MueLu: A package for multigrid based preconditioning
//
// Copyright 2012 NTESS and the MueLu contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef MUELU_MUELUSMOOTHER_DEF_HPP
#define MUELU_MUELUSMOOTHER_DEF_HPP

#include "Ifpack2_CreateOverlapGraph.hpp"
#include "Kokkos_Macros.hpp"
#include "MueLu_ConfigDefs.hpp"

#include <Teuchos_ParameterList.hpp>

#include <Xpetra_CrsMatrix.hpp>
#include <Xpetra_Matrix.hpp>
#include <Xpetra_MultiVectorFactory.hpp>
#include <string>
#include <type_traits>

#include "Teuchos_DefaultMpiComm_decl.hpp"
#include "Teuchos_RCPDecl.hpp"
#include "Tpetra_CombineMode.hpp"
#include "Xpetra_CrsMatrixWrap_decl.hpp"
#include "Xpetra_TpetraHalfPrecisionOperator.hpp"
#include "MueLu_CreateXpetraPreconditioner.hpp"
#include "MueLu_XpetraOperator.hpp"
#include "MueLu_MueLuSmoother_decl.hpp"
#include "MueLu_Level.hpp"
#include "MueLu_Monitor.hpp"

namespace MueLu {

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
MueLuSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::MueLuSmoother(const std::string type, const Teuchos::ParameterList& paramList, const LO& overlap)
  : type_(type)
  , overlap_(overlap) {
  const bool solverSupported = (type_ == "MueLu");
  this->declareConstructionOutcome(!solverSupported, "MueLuSmoother does not provide the smoother '" + type_ + "'.");
  if (solverSupported)
    SetParameterList(paramList);
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void MueLuSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::SetParameterList(const Teuchos::ParameterList& paramList) {
  Factory::SetParameterList(paramList);
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void MueLuSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::DeclareInput(Level& currentLevel) const {
  const ParameterList& pL = this->GetParameterList();

  this->Input(currentLevel, "A");
  // this->Input(currentLevel, "Coordinates");
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void MueLuSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Setup(Level& currentLevel) {
  FactoryMonitor m(*this, "Setup Smoother", currentLevel);

  SetupMueLu(currentLevel);
  SmootherPrototype::IsSetup(true);
  this->GetOStream(Statistics1) << description() << std::endl;
}

template <class NewScalar, class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP<Xpetra::Matrix<typename Teuchos::ScalarTraits<Scalar>::halfPrecision, LocalOrdinal, GlobalOrdinal, Node>>
convert(Teuchos::RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>>& A) {
  auto tpA    = toTpetra(A);
  auto tpLowA = tpA->template convert<NewScalar>();
  return Xpetra::toXpetra(tpLowA);
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void MueLuSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::SetupMueLu(Level& currentLevel) {
  // SetupLowPrecision(currentLevel);
  SetupOverlapped(currentLevel);
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void MueLuSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::SetupOverlapped(Level& currentLevel) {
  using impl_scalar_type = typename Matrix::impl_scalar_type;

  auto A               = currentLevel.Get<RCP<Matrix>>("A");
  ParameterList params = this->GetParameterList();
  RCP<Hierarchy> H;
  auto comm = A->getRowMap()->getComm();
  if (comm->getSize() == 1) {
    H = CreateXpetraPreconditioner<Scalar, LocalOrdinal, GlobalOrdinal, Node>(A, params);
  } else if (overlap_ == 0) {
    // Localize by dropping all nonlocal entries in A

    auto rowMap    = A->getRowMap();
    auto lclRowMap = rowMap->getLocalMap();
    auto A_local   = Xpetra::applyFilter_GID(
          A, KOKKOS_LAMBDA(const GlobalOrdinal rgid, const GlobalOrdinal cgid, const impl_scalar_type val) {
            return (lclRowMap.getLocalElement(cgid) != ::Tpetra::Details::OrdinalTraits<GlobalOrdinal>::invalid());
          });
    auto newComm = rcp(new Teuchos::MpiComm<int>(MPI_COMM_SELF));
    auto newMap  = MapFactory::Build(rowMap->lib(), rowMap->getLocalNumElements(), rowMap->getMyGlobalIndicesDevice(), rowMap->getIndexBase(), newComm);

    A_local = MatrixFactory::Build(A_local->getLocalMatrixDevice(), newMap, newMap);

    H = CreateXpetraPreconditioner<Scalar, LocalOrdinal, GlobalOrdinal, Node>(A_local, params);
  } else if (overlap_ >= 1) {
    auto tpA                   = toTpetra(rcp_const_cast<const Matrix>(A));
    auto tpAoverlapped         = Ifpack2::createOverlapMatrix(tpA, overlap_);
    auto tpAoverlappedNonConst = Teuchos::rcp_const_cast<Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>>(tpAoverlapped);
    auto A_local               = Xpetra::toXpetra(tpAoverlappedNonConst);

    auto rowMap = A_local->getRowMap();

    importer_ = ImportFactory::Build(A->getRowMap(), rowMap);

    auto newComm = rcp(new Teuchos::MpiComm<int>(MPI_COMM_SELF));
    auto newMap  = MapFactory::Build(rowMap->lib(), rowMap->getLocalNumElements(), rowMap->getMyGlobalIndicesDevice(), rowMap->getIndexBase(), newComm);

    A_local = MatrixFactory::Build(A_local->getLocalMatrixDevice(), newMap, newMap, newMap, newMap);

    // Xpetra::IO<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Write("A_local_"+std::to_string(comm->getRank()), *A_local);

    H = CreateXpetraPreconditioner<Scalar, LocalOrdinal, GlobalOrdinal, Node>(A_local, params);
  } else
    throw std::logic_error("");
  op_ = rcp(new XpetraOperator<Scalar, LocalOrdinal, GlobalOrdinal, Node>(H));
  std::ostringstream oss;
  op_->describe(*Teuchos::fancyOStream(Teuchos::rcpFromRef(oss)));
  cachedDescription_ = oss.str();
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void MueLuSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::SetupLowPrecision(Level& currentLevel) {
#if defined(HAVE_TPETRA_INST_DOUBLE) && defined(HAVE_TPETRA_INST_FLOAT)
  using coordinateType        = typename Teuchos::ScalarTraits<Scalar>::coordinateType;
  using RealValuedMultiVector = Xpetra::MultiVector<coordinateType, LO, GO, NO>;

  using HalfPrecOp = Xpetra::TpetraHalfPrecisionOperator<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
  using HalfScalar = typename HalfPrecOp::HalfScalar;

  ParameterList params = this->GetParameterList();
  auto A_high          = currentLevel.Get<RCP<Matrix>>("A");
  auto A_low           = convert<HalfScalar>(A_high);
  // auto coords = currentLevel.Get<RCP<RealValuedMultiVector> >("Coordinates");

  auto H = CreateXpetraPreconditioner<HalfScalar, LocalOrdinal, GlobalOrdinal, Node>(A_low, params);
  if constexpr (std::is_same_v<HalfScalar, Scalar>)
    op_ = rcp(new XpetraOperator<Scalar, LocalOrdinal, GlobalOrdinal, Node>(H));
  else {
    auto op = rcp(new XpetraOperator<HalfScalar, LocalOrdinal, GlobalOrdinal, Node>(H));
    op_     = rcp(new HalfPrecOp(op));
  }
  std::ostringstream oss;
  op_->describe(*Teuchos::fancyOStream(Teuchos::rcpFromRef(oss)));
  cachedDescription_ = oss.str();
#else

#endif
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void MueLuSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Apply(MultiVector& X, const MultiVector& B, bool InitialGuessIsZero) const {
  TEUCHOS_TEST_FOR_EXCEPTION(SmootherPrototype::IsSetup() == false, Exceptions::RuntimeError, "MueLu::MueLuSmoother::Apply(): Setup() has not been called");

  if (importer_.is_null()) {
    // if (InitialGuessIsZero) {
    //   X.putScalar(0.0);
    // }
    RCP<const Map> mapX;
    {
      mapX = X.getMap();
      X.replaceMap(op_->getDomainMap());
    }
    op_->apply(B, X);
    {
      X.replaceMap(mapX);
    }
  } else {
    RCP<MultiVector> localX = MultiVectorFactory::Build(importer_->getTargetMap(), X.getNumVectors());
    RCP<MultiVector> localB = MultiVectorFactory::Build(importer_->getTargetMap(), B.getNumVectors());
    if (!InitialGuessIsZero)
      localX->doImport(X, *importer_, Xpetra::INSERT);
    localB->doImport(B, *importer_, Xpetra::INSERT);
    RCP<const Map> mapX;
    RCP<const Map> mapB;
    {
      mapX = localX->getMap();
      localX->replaceMap(op_->getRangeMap());
      mapB = localB->getMap();
      localB->replaceMap(op_->getDomainMap());
    }
    op_->apply(*localB, *localX);
    {
      localX->replaceMap(mapX);
      localB->replaceMap(mapB);
    }
    X.doExport(*localX, *importer_, Xpetra::ADD);
  }
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<MueLu::SmootherPrototype<Scalar, LocalOrdinal, GlobalOrdinal, Node>> MueLuSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Copy() const {
  RCP<MueLuSmoother> smoother = rcp(new MueLuSmoother(*this));
  smoother->SetParameterList(this->GetParameterList());
  return smoother;
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
std::string MueLuSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::description() const {
  std::ostringstream out;
  if (SmootherPrototype::IsSetup()) {
    out << op_->description();
    out << std::endl
        << std::endl;
    // out << cachedDescription_;
  } else {
    out << "MueLu";
  }
  return out.str();
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void MueLuSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::print(Teuchos::FancyOStream& out, const VerbLevel verbLevel) const {
  MUELU_DESCRIBE;

  if (verbLevel & Parameters1) {
    out0 << "Parameter list: " << std::endl;
    Teuchos::OSTab tab2(out);
    out << this->GetParameterList();
  }

  if (verbLevel & External) {
    if (op_ != Teuchos::null) {
      Teuchos::OSTab tab2(out);
      out << *op_ << std::endl
          << std::endl;
    }
  }

  // if (verbLevel & Debug) {
  //   out0 << "IsSetup: " << Teuchos::toString(SmootherPrototype::IsSetup()) << std::endl
  //        << "-" << std::endl
  //        << "RCP<solver_>: " << tSolver_ << std::endl;
  // }
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
size_t MueLuSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::getNodeSmootherComplexity() const {
  return Teuchos::OrdinalTraits<size_t>::invalid();
}

}  // namespace MueLu

#endif  // MUELU_MUELUSMOOTHER_DEF_HPP
