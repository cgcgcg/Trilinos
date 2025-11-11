// @HEADER
// *****************************************************************************
//             Xpetra: A linear algebra interface package
//
// Copyright 2012 NTESS and the Xpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef XPETRA_MULTIVECTOR_DECL_HPP
#define XPETRA_MULTIVECTOR_DECL_HPP

#include <Teuchos_LabeledObject.hpp>
#include "Xpetra_ConfigDefs.hpp"
#include "Xpetra_DistObject.hpp"
#include "Tpetra_MultiVector.hpp"
#include "Xpetra_Access.hpp"

namespace Xpetra {
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  using MultiVector = Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
}  // namespace Xpetra

#define XPETRA_MULTIVECTOR_SHORT

#endif  // XPETRA_MULTIVECTOR_DECL_HPP
