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

#include "MueLu_ConfigDefs.hpp"

#include <Teuchos_ParameterList.hpp>

#include <Xpetra_CrsMatrix.hpp>
#include <Xpetra_Matrix.hpp>
#include <Xpetra_MultiVectorFactory.hpp>
#include <type_traits>

#include "Xpetra_TpetraHalfPrecisionOperator.hpp"
#include "MueLu_CreateXpetraPreconditioner.hpp"
#include "MueLu_XpetraOperator.hpp"
#include "MueLu_MueLuSmoother_decl.hpp"
#include "MueLu_Level.hpp"
#include "MueLu_Monitor.hpp"

namespace MueLu {

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
MueLuSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::MueLuSmoother(const std::string type, const Teuchos::ParameterList& paramList)
  : type_(type) {
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
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void MueLuSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Apply(MultiVector& X, const MultiVector& B, bool InitialGuessIsZero) const {
  TEUCHOS_TEST_FOR_EXCEPTION(SmootherPrototype::IsSetup() == false, Exceptions::RuntimeError, "MueLu::MueLuSmoother::Apply(): Setup() has not been called");

  if (InitialGuessIsZero) {
    X.putScalar(0.0);
  }
  op_->apply(B, X);
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
