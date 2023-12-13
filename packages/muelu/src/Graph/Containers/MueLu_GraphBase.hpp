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
#ifndef MUELU_GRAPHBASE_HPP
#define MUELU_GRAPHBASE_HPP

#include <Xpetra_ConfigDefs.hpp>   // global_size_t
#include <Xpetra_Map_fwd.hpp>

#include "MueLu_ConfigDefs.hpp"

#include "MueLu_BaseClass.hpp"

#include <Kokkos_StaticCrsGraph.hpp>

namespace MueLu {

/*!
   @class GraphBase
   @brief MueLu representation of a graph.

   Pure virtual base class for MueLu representations of graphs.
*/
  template<class LocalOrdinal = DefaultLocalOrdinal,
           class GlobalOrdinal = DefaultGlobalOrdinal,
           class Node = DefaultNode>
  class GraphBase
    : public BaseClass {
#undef MUELU_GRAPHBASE_SHORT
#include "MueLu_UseShortNamesOrdinal.hpp"

  public:
    // For Zoltan2 compatibility
    using lno_t  = LocalOrdinal;
    using gno_t  = GlobalOrdinal;
    using node_t = Node;

    using device_type         = typename Node::device_type;
    using memory_space        = typename device_type::memory_space;
    using local_graph_type    = Kokkos::StaticCrsGraph<LocalOrdinal,
                                                       Kokkos::LayoutLeft,
                                                       device_type, void, size_t>;
    using boundary_nodes_type = Kokkos::View<const bool*, memory_space>;

    //! @name Constructors/Destructors.
    //@{
    virtual ~GraphBase() {};
    //@}

    virtual const RCP<const Teuchos::Comm<int> > GetComm() const = 0;
    virtual const RCP<const Map> GetDomainMap() const = 0;
    virtual const RCP<const Map> GetImportMap() const = 0;

    //! @name Query graph attributes.
    //@{

    //! Return number of vertices owned by the calling node.
    virtual size_t GetNodeNumVertices() const = 0;

    //! Return number of edges owned by the calling node.
    virtual size_t GetNodeNumEdges()    const = 0;

    virtual void   SetBoundaryNodeMap(const boundary_nodes_type boundaryArray) = 0;

    virtual size_t getLocalMaxNumRowEntries() const = 0;

    virtual const boundary_nodes_type GetBoundaryNodeMap() const = 0;

    //! Return the list of vertices adjacent to the vertex 'v'.
    virtual Kokkos::GraphRowViewConst<local_graph_type> getNeighborVertices(LocalOrdinal v) const = 0;

    virtual RCP<CrsGraph> GetCrsGraph() const = 0;

    //! Return true if vertex with local id 'v' is on current process.
    virtual bool isLocalNeighborVertex(LocalOrdinal v) const = 0;
    //@}

    //! @name Print graph.
    //@{
    /// Return a simple one-line description of the Graph.
    virtual std::string description() const = 0;

    //! Print the Graph with some verbosity level to an FancyOStream object.
    //using MueLu::Describable::describe; // overloading, not hiding
    //void describe(Teuchos::FancyOStream &out, const VerbLevel verbLevel = Default) const;;
    virtual void print(Teuchos::FancyOStream &out, const VerbLevel verbLevel = Default) const = 0;
    //@}

  };

} // namespace MueLu

#define MUELU_GRAPHBASE_SHORT
#endif // MUELU_GRAPHBASE_HPP
