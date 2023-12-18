// @HEADER
//
// ***********************************************************************
//
//        MueLu: A package for multigrid based preconditioning
//                  Copyright 2012 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact
//                    Jonathan Hu       (jhu@sandia.gov)
//                    Andrey Prokopenko (aprokop@sandia.gov)
//                    Ray Tuminaro      (rstumin@sandia.gov)
//
// ***********************************************************************
//
// @HEADER
#ifndef MUELU_LWGRAPH_DECL_HPP
#define MUELU_LWGRAPH_DECL_HPP

#include <Xpetra_ConfigDefs.hpp>   // global_size_t
#include <Xpetra_CrsGraph_fwd.hpp>     // inline functions requires class declaration
#include <Xpetra_Map_fwd.hpp>

#include "MueLu_ConfigDefs.hpp"

#include "MueLu_LWGraph_fwd.hpp"
#include "MueLu_GraphBase.hpp"
#include "MueLu_LWGraph_kokkos.hpp"
#include "MueLu_Exceptions.hpp"

namespace MueLu {


  template<class LocalOrdinal = DefaultLocalOrdinal,
           class GlobalOrdinal = DefaultGlobalOrdinal,
           class Node = DefaultNode>
  typename LWGraph_kokkos<LocalOrdinal, GlobalOrdinal, Node>::local_graph_type
  constructLocalGraph(const ArrayRCP<const size_t>& rowPtrs, const ArrayRCP<const LocalOrdinal>& colPtrs) {
    using local_graph_type = typename LWGraph_kokkos<LocalOrdinal, GlobalOrdinal, Node>::local_graph_type;
    using entries_type = typename local_graph_type::entries_type;
    using row_map_type = typename local_graph_type::row_map_type;

    Kokkos::View<size_t*,       Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > rows(const_cast<size_t*>(rowPtrs().getRawPtr()), rowPtrs().size());
    Kokkos::View<LocalOrdinal*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > cols(const_cast<LocalOrdinal*>(colPtrs().getRawPtr()), colPtrs().size());

    typename row_map_type::non_const_type rowsDevice("rows", rowPtrs().size());
    typename entries_type::non_const_type colsDevice("cols", colPtrs().size());

    Kokkos::deep_copy(rowsDevice, rows);
    Kokkos::deep_copy(colsDevice, cols);
    local_graph_type lclGraph(colsDevice, rowsDevice);
    return lclGraph;
  }

/*!
   @class LWGraph
   @brief Lightweight MueLu representation of a compressed row storage graph.

   This class is lightweight in the sense that it holds to local graph information.  These were built without using
   fillComplete.
   TODO handle systems
*/
  template<class LocalOrdinal = DefaultLocalOrdinal,
           class GlobalOrdinal = DefaultGlobalOrdinal,
           class Node = DefaultNode>
  class LWGraph : public MueLu::LWGraph_kokkos<LocalOrdinal,GlobalOrdinal,Node> {
#undef MUELU_LWGRAPH_SHORT
#include "MueLu_UseShortNamesOrdinal.hpp"

  public:

    //! LWGraph constructor
    //
    // @param[in] rowPtrs: Array containing row offsets (CSR format)
    // @param[in] colPtrs: Array containing local column indices (CSR format)
    // @param[in] domainMap: non-overlapping (domain) map for graph. Usually provided by AmalgamationFactory stored in UnAmalgamationInfo container
    // @param[in] importMap: overlapping map for graph. Usually provided by AmalgamationFactory stored in UnAmalgamationInfo container
    // @param[in] objectLabel: label string
    LWGraph(const ArrayRCP<const size_t>& rowPtrs, const ArrayRCP<const LocalOrdinal>& colPtrs,
            const RCP<const Map>& domainMap, const RCP<const Map>& importMap, const std::string& objectLabel = "")
      : LWGraph_kokkos(constructLocalGraph(rowPtrs, colPtrs), domainMap, importMap, objectLabel),
        rows_(rowPtrs),
        columns_(colPtrs) { }

    LWGraph(const RCP<const Xpetra::CrsGraph<LocalOrdinal, GlobalOrdinal, Node> >& G, const std::string& objectLabel = "") : LWGraph_kokkos(G, objectLabel) {}

    //! Return the row pointers of the local graph
    const ArrayRCP<const size_t> getRowPtrs() const {
      return rows_;
    }

    //! Return the list entries in the local graph
    const ArrayRCP<const LO> getEntries() const {
      return columns_;
    }

  private:

    //! Indices into columns_ array.  Part of local graph information.
    const ArrayRCP<const size_t> rows_;
    //! Columns corresponding to connections.  Part of local graph information.
    const ArrayRCP<const LO> columns_;

  };
} // namespace MueLu

#define MUELU_LWGRAPH_SHORT
#endif // MUELU_LWGRAPH_DECL_HPP
