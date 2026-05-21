// @HEADER
// *****************************************************************************
//        MueLu: A package for multigrid based preconditioning
//
// Copyright 2012 NTESS and the MueLu contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef MUELU_RTCFACTORY_DEF_HPP
#define MUELU_RTCFACTORY_DEF_HPP

#include "MueLu_RTCFactory_decl.hpp"
#include "MueLu_Level.hpp"
#include "MueLu_Monitor.hpp"
#include "MueLu_MasterList.hpp"
#include "RTC_FunctionRTC.hh"
#include "Tpetra_Access.hpp"
#include "Xpetra_MultiVectorFactory.hpp"

namespace MueLu {

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RTCFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::RTCFactory() = default;

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RTCFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::~RTCFactory() = default;

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<const ParameterList> RTCFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::GetValidParameterList() const {
  RCP<ParameterList> validParamList = rcp(new ParameterList());

  validParamList->set<std::string>("RTC function", "value = 0", "RTC function that is evaluated on the coordinates.");
  validParamList->set<std::string>("Output", "Undefined", "Name of the output");

  validParamList->set<RCP<const FactoryBase>>("Coordinates", Teuchos::null, "Generating factory of the coordinates");

  return validParamList;
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void RTCFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::DeclareInput(Level& currentLevel) const {
  // const ParameterList& pL = GetParameterList();
  Input(currentLevel, "Coordinates");
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void RTCFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Build(Level& currentLevel) const {
  const ParameterList& pL = GetParameterList();
  auto funBody            = pL.get<std::string>("RTC function");
  auto outputName         = pL.get<std::string>("Output");

  FactoryMonitor m(*this, "RTC factory \"" + funBody + "\"", currentLevel);

  RCP<RealValuedMultiVector> Coords = Get<RCP<RealValuedMultiVector>>(currentLevel, "Coordinates");
  RCP<MultiVector> Values           = MultiVectorFactory::Build(Coords->getMap(), 1, false);

  int dim = Coords->getNumVectors();

  auto fun = PG_RuntimeCompiler::Function();
  fun.addVar("double", "value");
  fun.addVar("double", "x");
  if (dim >= 2)
    fun.addVar("double", "y");
  if (dim >= 3)
    fun.addVar("double", "z");
  fun.addBody(funBody);

  double x;
  double y;
  double z;
  double value;
  fun.varAddrFill(0, &value);
  fun.varAddrFill(1, &x);
  if (dim >= 2)
    fun.varAddrFill(2, &y);
  if (dim >= 3)
    fun.varAddrFill(3, &z);

  {
    auto lclCoords = Coords->getLocalViewHost(Tpetra::Access::ReadOnly);
    auto lclValues = Values->getLocalViewHost(Tpetra::Access::OverwriteAll);

    for (LocalOrdinal i = 0; i < Coords->getLocalLength(); ++i) {
      x = lclCoords(i, 0);
      if (dim >= 2)
        y = lclCoords(i, 1);
      if (dim >= 3)
        z = lclCoords(i, 2);
      fun.execute();
      lclValues(i, 0) = value;
    }
  }

  Set(currentLevel, outputName, Values);
}
}  // namespace MueLu

#endif
