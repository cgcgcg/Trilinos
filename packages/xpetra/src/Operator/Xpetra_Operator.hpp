// @HEADER
// *****************************************************************************
//             Xpetra: A linear algebra interface package
//
// Copyright 2012 NTESS and the Xpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef XPETRA_OPERATOR_HPP
#define XPETRA_OPERATOR_HPP

#include "Xpetra_ConfigDefs.hpp"

#include <Teuchos_Describable.hpp>
#include <Teuchos_BLAS_types.hpp>
#include <Teuchos_ScalarTraits.hpp>

#include "Xpetra_Map.hpp"
#include "Xpetra_MultiVector.hpp"

namespace Xpetra {

template <class Scalar,
          class LocalOrdinal,
          class GlobalOrdinal,
          class Node = Tpetra::KokkosClassic::DefaultNode::DefaultNodeType>
using Operator=Tpetra::Operator<Scalar,LocalOrdinal, GlobalOrdinal, Node>;

}  // namespace Xpetra

#define XPETRA_OPERATOR_SHORT
#endif  // XPETRA_OPERATOR_HPP
