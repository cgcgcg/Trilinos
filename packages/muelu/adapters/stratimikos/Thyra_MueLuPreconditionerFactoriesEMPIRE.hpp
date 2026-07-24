// @HEADER
// *****************************************************************************
//        MueLu: A package for multigrid based preconditioning
//
// Copyright 2012 NTESS and the MueLu contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef THYRA_MUELUPRECONDITIONERFACTORIESEMPIRE_HPP
#define THYRA_MUELUPRECONDITIONERFACTORIESEMPIRE_HPP

#include "Thyra_MueLuPreconditionerFactory.hpp"
#include "Thyra_MueLuRefMaxwellPreconditionerFactory.hpp"

namespace Thyra {

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node = Tpetra::KokkosClassic::DefaultNode::DefaultNodeType>
class MueLuPreconditionerFactoryEMPIRE : public MueLuPreconditionerFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node> {
  Teuchos::RCP<const Teuchos::ParameterList> getValidParameters() const override {
    static RCP<const ParameterList> validPL;

    if (Teuchos::is_null(validPL)) {
      validPL = MueLu::MasterList::List();
      std::string prec_string =
#include "stratimikos/EMPIRE_ES.xml.inc"
          ;
      Teuchos::updateParametersFromXmlString(prec_string, Teuchos::rcp_const_cast<Teuchos::ParameterList>(validPL).ptr());

      if constexpr (Node::is_gpu) {
        std::string gpu_prec_string =
#include "stratimikos/EMPIRE_ES_gpu.xml.inc"
            ;
        Teuchos::updateParametersFromXmlString(gpu_prec_string, Teuchos::rcp_const_cast<Teuchos::ParameterList>(validPL).ptr());
      }
    }

    return validPL;
  }
};

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node = Tpetra::KokkosClassic::DefaultNode::DefaultNodeType>
class MueLuRefMaxwellPreconditionerFactoryEMPIRE : public MueLuRefMaxwellPreconditionerFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node> {
  Teuchos::RCP<const Teuchos::ParameterList> getValidParameters() const override {
    static RCP<const ParameterList> validPL;

    if (Teuchos::is_null(validPL)) {
      validPL = MueLu::MasterList::List();
      std::string prec_string =
#include "stratimikos/EMPIRE_EM_refmaxwell.xml.inc"
          ;
      Teuchos::updateParametersFromXmlString(prec_string, Teuchos::rcp_const_cast<Teuchos::ParameterList>(validPL).ptr());

      if constexpr (Node::is_gpu) {
        std::string gpu_prec_string =
#include "stratimikos/EMPIRE_EM_refmaxwell_gpu.xml.inc"
            ;
        Teuchos::updateParametersFromXmlString(gpu_prec_string, Teuchos::rcp_const_cast<Teuchos::ParameterList>(validPL).ptr());
      }
    }

    return validPL;
  }
};

}  // namespace Thyra
#endif
