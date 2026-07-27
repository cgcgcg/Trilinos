// @HEADER
// *****************************************************************************
//        MueLu: A package for multigrid based preconditioning
//
// Copyright 2012 NTESS and the MueLu contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef MUELU_AGGRESSIVEAGGREGATION_DEF_HPP
#define MUELU_AGGRESSIVEAGGREGATION_DEF_HPP

#include <climits>

#include <Xpetra_Map.hpp>
#include <Xpetra_Vector.hpp>
#include <Xpetra_MultiVectorFactory.hpp>
#include <Xpetra_VectorFactory.hpp>

#include "MueLu_AggressiveAggregationFactory_decl.hpp"

#include "MueLu_Level.hpp"
#include "MueLu_Aggregates.hpp"
#include "MueLu_MasterList.hpp"
#include "MueLu_Monitor.hpp"

namespace MueLu {

template <class LocalOrdinal, class GlobalOrdinal, class Node>
AggressiveAggregationFactory<LocalOrdinal, GlobalOrdinal, Node>::AggressiveAggregationFactory() = default;

template <class LocalOrdinal, class GlobalOrdinal, class Node>
AggressiveAggregationFactory<LocalOrdinal, GlobalOrdinal, Node>::~AggressiveAggregationFactory() = default;

template <class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<const ParameterList> AggressiveAggregationFactory<LocalOrdinal, GlobalOrdinal, Node>::GetValidParameterList() const {
  RCP<ParameterList> validParamList = rcp(new ParameterList());

#define SET_VALID_ENTRY(name) validParamList->setEntry(name, MasterList::getEntry(name))
#undef SET_VALID_ENTRY

  // general variables needed in AggregationFactory
  validParamList->set<RCP<const FactoryBase>>("Graph", null, "Generating factory of the graph");
  validParamList->set<RCP<const FactoryBase>>("DofsPerNode", null, "Generating factory for variable \'DofsPerNode\', usually the same as for \'Graph\'");

  return validParamList;
}

template <class LocalOrdinal, class GlobalOrdinal, class Node>
void AggressiveAggregationFactory<LocalOrdinal, GlobalOrdinal, Node>::DeclareInput(Level& currentLevel) const {
  Input(currentLevel, "Graph");
  Input(currentLevel, "DofsPerNode");
}

template <class LocalOrdinal, class GlobalOrdinal, class Node>
void AggressiveAggregationFactory<LocalOrdinal, GlobalOrdinal, Node>::Build(Level& currentLevel) const {
  FactoryMonitor m(*this, "Build", currentLevel);

  RCP<const LWGraph> graph;
  RCP<const LWGraph_kokkos> graph_kokkos;
  RCP<Aggregates> aggregates;
  RCP<const Teuchos::Comm<int>> comm;
  LO numRows;

  // "Graph" can have type "LWGraph" or "LWGraph_kokkos".
  if (IsType<RCP<LWGraph>>(currentLevel, "Graph")) {
    RCP<LWGraph> tmp_graph = Get<RCP<LWGraph>>(currentLevel, "Graph");
    graph_kokkos           = tmp_graph->copyToDevice();
    aggregates             = rcp(new Aggregates(*graph_kokkos));
    comm                   = graph_kokkos->GetComm();
    numRows                = graph_kokkos->GetNodeNumVertices();
  } else if (IsType<RCP<LWGraph_kokkos>>(currentLevel, "Graph")) {
    graph_kokkos = Get<RCP<LWGraph_kokkos>>(currentLevel, "Graph");
    aggregates   = rcp(new Aggregates(*graph_kokkos));
    comm         = graph_kokkos->GetComm();
    numRows      = graph_kokkos->GetNodeNumVertices();
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Graph has bad type.");
  }

  // Build
  aggregates->setObjectLabel("aggressive");

  // construct aggStat information
  using AggStatType   = typename AggregationAlgorithmBase<LocalOrdinal, GlobalOrdinal, Node>::AggStatType;
  AggStatType aggStat = AggStatType(Kokkos::ViewAllocateWithoutInitializing("aggregation status"), numRows);
  Kokkos::deep_copy(aggStat, READY);

  // Dirichlet nodes
  {
    auto dirichletBoundaryMap = graph_kokkos->GetBoundaryNodeMap();
    Kokkos::parallel_for(
        "MueLu - UncoupledAggregation: tagging boundary nodes in aggStat",
        Kokkos::RangePolicy<LocalOrdinal, typename LWGraph_kokkos::execution_space>(0, numRows),
        KOKKOS_LAMBDA(const LocalOrdinal nodeIdx) {
          if (dirichletBoundaryMap(nodeIdx) == true) {
            aggStat(nodeIdx) = BOUNDARY;
          }
        });
  }

  {
    using exec_space  = typename Node::execution_space;
    auto vertex2AggId = aggregates->GetVertex2AggId()->getLocalViewDevice(Tpetra::Access::ReadWrite);
    auto procWinner   = aggregates->GetProcWinner()->getLocalViewDevice(Tpetra::Access::OverwriteAll);
    int rank          = comm->getRank();
    Kokkos::parallel_for(
        Kokkos::RangePolicy<exec_space>(0, numRows),
        KOKKOS_LAMBDA(LocalOrdinal i) {
          if (aggStat(i) == READY) {
            procWinner(i, 0)   = rank;
            aggStat(i)         = AGGREGATED;
            vertex2AggId(i, 0) = 0;
          } else {
            procWinner(i, 0)   = MUELU_UNASSIGNED;
            aggStat(i)         = IGNORED;
            vertex2AggId(i, 0) = MUELU_UNAGGREGATED;
          }
        });
  }
  aggregates->SetNumAggregates(1);

  aggregates->AggregatesCrossProcessors(false);
  aggregates->ComputeAggregateSizes(true /*forceRecompute*/);

  Set(currentLevel, "Aggregates", aggregates);
}

}  // namespace MueLu

#endif /* MUELU_UNCOUPLEDAGGREGATIONFACTORY_DEF_HPP_ */
