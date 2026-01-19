// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
//
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef TPETRAEXT_DETAILS_SORTEDNESS_HPP
#define TPETRAEXT_DETAILS_SORTEDNESS_HPP

namespace Tpetra::Details {

template <class Scalar,
          class LocalOrdinal,
          class GlobalOrdinal,
          class Node>
class MatrixTraits {
 public:
  static constexpr bool spgemmNeedsSortedInputs() {
    return true;
  }
};

#if defined(HAVE_TPETRA_SERIAL)
template <class Scalar,
          class LocalOrdinal,
          class GlobalOrdinal>
class MatrixTraits<Scalar, LocalOrdinal, GlobalOrdinal, KokkosCompat::KokkosSerialWrapperNode> {
 public:
  static constexpr bool spgemmNeedsSortedInputs() {
    return false;
  }
};
#endif

}  // namespace Tpetra::Details

#endif
