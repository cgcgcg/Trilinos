// @HEADER
// *****************************************************************************
//             Xpetra: A linear algebra interface package
//
// Copyright 2012 NTESS and the Xpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef XPETRA_CRSGRAPH_HPP
#define XPETRA_CRSGRAPH_HPP

#include <Teuchos_ParameterList.hpp>

#include <Teuchos_Describable.hpp>
#include <Tpetra_KokkosCompat_DefaultNode.hpp>
#include "Tpetra_CrsGraph_decl.hpp"
#include "Xpetra_ConfigDefs.hpp"
#include "Xpetra_DistObject.hpp"
#include "Xpetra_Exceptions.hpp"

#include "Xpetra_Map.hpp"
#include "Tpetra_CrsGraph.hpp"

#ifdef HAVE_XPETRA_TPETRA
#include <KokkosSparse_StaticCrsGraph.hpp>
#endif

namespace Xpetra {

using Teuchos::ParameterList;

struct RowInfo {
  size_t localRow;
  size_t allocSize;
  size_t numEntries;
  size_t offset1D;
};

enum ELocalGlobal {
  LocalIndices,
  GlobalIndices
};

template <class LocalOrdinal,
          class GlobalOrdinal,
          class Node = Tpetra::KokkosClassic::DefaultNode::DefaultNodeType>
usimg CrsGraph = Tpetra::CrsGraph<LocalOrdinal, GlobalOrdinal, Node>;
}  // namespace Xpetra

#define XPETRA_CRSGRAPH_SHORT
#endif  // XPETRA_CRSGRAPH_HPP
