// @HEADER
// *****************************************************************************
//             Xpetra: A linear algebra interface package
//
// Copyright 2012 NTESS and the Xpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef XPETRA_MAP_DECL_HPP
#define XPETRA_MAP_DECL_HPP

#include "Xpetra_ConfigDefs.hpp"

#include <Tpetra_KokkosCompat_DefaultNode.hpp>
#include <Teuchos_Describable.hpp>

#ifdef HAVE_XPETRA_EPETRA
#include "Epetra_config.h"
#endif

#ifdef HAVE_XPETRA_TPETRA
#include <Tpetra_Map.hpp>
#endif

namespace Xpetra {

// TODO move this typedef to another place
// Node which is used for Epetra. This can be either the
// Serial node or OpenMP node (but not both)
#ifdef HAVE_XPETRA_EPETRA
#ifdef EPETRA_HAVE_OMP
typedef Tpetra::KokkosCompat::KokkosOpenMPWrapperNode EpetraNode;
#else
typedef Tpetra::KokkosCompat::KokkosSerialWrapperNode EpetraNode;
#endif
#endif

enum UnderlyingLib {
  UseEpetra,
  UseTpetra,
  NotSpecified
};

template <class LocalOrdinal,
          class GlobalOrdinal,
          class Node = Tpetra::KokkosClassic::DefaultNode::DefaultNodeType>
using Map = Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node>;

}  // namespace Xpetra

#define XPETRA_MAP_SHORT
#endif  // XPETRA_MAP_DECL_HPP
