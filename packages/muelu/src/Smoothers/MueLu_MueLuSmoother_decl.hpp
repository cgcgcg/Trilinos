// @HEADER
// *****************************************************************************
//        MueLu: A package for multigrid based preconditioning
//
// Copyright 2012 NTESS and the MueLu contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef MUELU_MUELUSMOOTHER_DECL_HPP
#define MUELU_MUELUSMOOTHER_DECL_HPP

#include <Teuchos_ParameterList.hpp>

#include <Xpetra_Matrix_fwd.hpp>

#include "MueLu_ConfigDefs.hpp"

#include "MueLu_MueLuSmoother_fwd.hpp"

#include "MueLu_Level_fwd.hpp"
#include "MueLu_SmootherPrototype.hpp"

namespace MueLu {

/*!
  @class MueLuSmoother
  @ingroup MueLuSmootherClasses
  @brief Class that encapsulates Operator smoothers.

  */

template <class Scalar        = SmootherPrototype<>::scalar_type,
          class LocalOrdinal  = typename SmootherPrototype<Scalar>::local_ordinal_type,
          class GlobalOrdinal = typename SmootherPrototype<Scalar, LocalOrdinal>::global_ordinal_type,
          class Node          = typename SmootherPrototype<Scalar, LocalOrdinal, GlobalOrdinal>::node_type>
class MueLuSmoother : public SmootherPrototype<Scalar, LocalOrdinal, GlobalOrdinal, Node> {
#undef MUELU_MUELUSMOOTHER_SHORT
#include "MueLu_UseShortNames.hpp"

 public:
  //! @name Constructors / destructors
  //@{
  /*! @brief Constructor

  @param type smoother type
  @param list options for the particular smoother (e.g., fill factor or damping parameter)

  */

  MueLuSmoother(const std::string type, const Teuchos::ParameterList& paramList, const LO& overlap);

  //! Destructor
  virtual ~MueLuSmoother() = default;

  //@}

  void SetParameterList(const Teuchos::ParameterList& paramList);

  //! Input
  //@{

  void DeclareInput(Level& currentLevel) const;

  //@}

  //! @name Computational methods.
  //@{

  /*! @brief Set up the smoother.

  This creates the underlying Operator smoother object, copies any parameter list options
  supplied to the constructor to the Operator object, and computes the preconditioner.
  */
  void Setup(Level& currentLevel);

  /*! @brief Apply the preconditioner.

  Solves the linear system <tt>AX=B</tt> using the constructed smoother.

  @param X initial guess
  @param B right-hand side
  @param InitialGuessIsZero (optional) If false, some work can be avoided. Whether this actually saves any work depends on the underlying Operator implementation.
  */
  void Apply(MultiVector& X, const MultiVector& B, bool InitialGuessIsZero = false) const;

  //@}

  //! @name Utilities
  //@{

  RCP<SmootherPrototype> Copy() const;

  //@}

  //! @name Overridden from Teuchos::Describable
  //@{

  //! Return a simple one-line description of this object.
  std::string description() const;

  //! Print the object with some verbosity level to an FancyOStream object.
  // using MueLu::Describable::describe; // overloading, not hiding
  // void describe(Teuchos::FancyOStream &out, const VerbLevel verbLevel = Default) const
  void print(Teuchos::FancyOStream& out, const VerbLevel verbLevel = Default) const;

  //@}

  //! Get a rough estimate of cost per iteration
  size_t getNodeSmootherComplexity() const;

 private:
  void SetupMueLu(Level& currentLevel);
  void SetupOverlapped(Level& currentLevel);
  void SetupLowPrecision(Level& currentLevel);

 private:
  std::string type_;
  LocalOrdinal overlap_;
  std::string cachedDescription_;

  //! matrix, used in apply if solving residual equation
  RCP<Operator> op_;
  RCP<Import> importer_;

};  // class MueLuSmoother

}  // namespace MueLu

#define MUELU_MUELUSMOOTHER_SHORT
#endif  // MUELU_MUELUSMOOTHER_DECL_HPP
