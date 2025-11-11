// @HEADER
// *****************************************************************************
//             Xpetra: A linear algebra interface package
//
// Copyright 2012 NTESS and the Xpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef XPETRA_DISTOBJECT_HPP
#define XPETRA_DISTOBJECT_HPP

#include "Xpetra_Import.hpp"
#include "Xpetra_Export.hpp"
#include "Tpetra_DistObject.hpp"

namespace Xpetra {

template <class Packet,
          class LocalOrdinal,
          class GlobalOrdinal,
          class Node = Tpetra::KokkosClassic::DefaultNode::DefaultNodeType>
using DistObject = Tpetra::DistObject<Packet, LocalOrdinal, GlobalOrdinal, Node>;

}  // namespace Xpetra

#define XPETRA_DISTOBJECT_SHORT
#endif  // XPETRA_DISTOBJECT_HPP
