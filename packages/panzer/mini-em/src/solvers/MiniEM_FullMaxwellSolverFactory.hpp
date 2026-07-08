// @HEADER
// *****************************************************************************
//           Panzer: A partial differential equation assembly
//       engine for strongly coupled complex multiphysics systems
//
// Copyright 2011 NTESS and the Panzer contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef MINIEM_FULLMAXWELLSOLVERFACTORY_HPP
#define MINIEM_FULLMAXWELLSOLVERFACTORY_HPP

#include "Teuchos_RCP.hpp"

#include "Teko_InverseLibrary.hpp"
#include "Teko_InverseFactory.hpp"
#include "Teko_Utilities.hpp"

namespace mini_em {

class FullMaxwellSolverFactory : public Teko::InverseFactory {
public:
  FullMaxwellSolverFactory() { }
  virtual ~FullMaxwellSolverFactory() {}

  Teko::InverseLinearOp buildInverse(const Teko::LinearOp &blo) const;

  void rebuildInverse(const Teko::LinearOp& source, Teko::InverseLinearOp& dest) const;

  Teuchos::RCP<const Teuchos::ParameterList> getParameterList() const;

  std::string toString() const;

  //! Initialize from a parameter list
  virtual void initializeFromParameterList(const Teuchos::ParameterList &pl);

private:
  // Holds all inverse factories
  Teko::InverseLibrary invLib;

  bool use_discrete_curl_;
  bool simplifyFaraday_;
  bool dump;
  bool doDebug;
  bool useAsPreconditioner;
  double dt;

  // type of preconditioner for Schur complement
  std::string S_E_prec_type_;

  mutable Teko::InverseLinearOp S_E_prec_;

  // parameters
  Teuchos::ParameterList params;
};

} // namespace mini_em

#endif
