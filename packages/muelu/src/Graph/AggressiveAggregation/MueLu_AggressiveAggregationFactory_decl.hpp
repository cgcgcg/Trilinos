// @HEADER
// *****************************************************************************
//        MueLu: A package for multigrid based preconditioning
//
// Copyright 2012 NTESS and the MueLu contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef MUELU_AGGRESSIVEAGGREGATION_DECL_HPP
#define MUELU_AGGRESSIVEAGGREGATION_DECL_HPP

#include <Xpetra_Map_fwd.hpp>
#include <Xpetra_Vector_fwd.hpp>

#include "MueLu_ConfigDefs.hpp"
#include "MueLu_SingleLevelFactoryBase.hpp"
#include "MueLu_UncoupledAggregationFactory_fwd.hpp"

#include "MueLu_AggregationAlgorithmBase.hpp"
#include "MueLu_InterfaceAggregationAlgorithm_fwd.hpp"
#include "MueLu_OnePtAggregationAlgorithm_fwd.hpp"
#include "MueLu_PreserveDirichletAggregationAlgorithm_fwd.hpp"

#include "MueLu_AggregationPhase1Algorithm_fwd.hpp"
#include "MueLu_AggregationPhase2aAlgorithm_fwd.hpp"
#include "MueLu_AggregationPhase2bAlgorithm_fwd.hpp"
#include "MueLu_AggregationPhase3Algorithm_fwd.hpp"

#include "MueLu_Level_fwd.hpp"

#include "MueLu_LWGraph_fwd.hpp"
#include "MueLu_Aggregates_fwd.hpp"
#include "MueLu_Exceptions.hpp"

namespace MueLu {

/*!
    @class UncoupledAggregationFactory
    @brief Factory for building uncoupled aggregates.

    Factory for creating uncoupled aggregates from the amalgamated graph of A. The uncoupled aggregation method
    uses several aggregation phases which put together all nodes into aggregates.

    Internally, each node has a status which can be one of the following:

    Node status | Meaning
    ------------|---------
    READY       | Node is not aggregated and can be used for building a new aggregate or can be added to an existing aggregate.
    AGGREGATED  | Node is aggregated.
    IGNORED     | Node is not considered for aggregation (it may have been dropped or put into a singleton aggregate)
    BOUNDARY    | Node is a Dirichlet boundary node (with one or more Dirichlet boundary conditions).
    ONEPT       | The user forces the aggregation algorithm to treat the node as a singleton. Important: Do not forget to set aggregation: allow user-specified singletons to true! Otherwise Phase3 will just handle the ONEPT nodes and probably not build singletons

    @ingroup Aggregation

    ## Input/output of UncoupledAggregationFactory ##

    ### User parameters of UncoupledAggregationFactory ###
    Parameter | type | default | master.xml | validated | requested | description
    ----------|------|---------|:----------:|:---------:|:---------:|------------
     Graph              | Factory | null |   | * | * | Generating factory of the graph of A
     DofsPerNode        | Factory | null |   | * | * | Generating factory for variable 'DofsPerNode', usually the same as for 'Graph'
     OnePt aggregate map name  | string |  | | * | * | Name of input map for single node aggregates (default=''). Makes only sense if the parameter 'aggregation: allow user-specified singletons' is set to true.
     OnePt aggregate map factory | Factory | null |   | * | * | Generating factory of (DOF) map for single node aggregates.  Makes only sense if the parameter 'aggregation: allow user-specified singletons' is set to true.
     aggregation: max agg size | int | see master.xml | * | * |  | Maximum number of nodes per aggregate.
     aggregation: min agg size | int | see master.xml | * | * |  | Minimum number of nodes necessary to build a new aggregate.
     aggregation: max selected neighbors | int | see master.xml | * | * |  | Maximum number of neighbor nodes already in aggregate (needed in Phase1)
     aggregation: ordering | string | "natural" | * | * |  | Ordering of node aggregation (can be either "natural", "graph" or "random").
     aggregation: enable phase 1 | bool | true | * | * |   |Turn on/off phase 1 aggregation
     aggregation: enable phase 2a | bool | true | * | * |  |Turn on/off phase 2a aggregation
     aggregation: enable phase 2b | bool | true | * | * |  |Turn on/off phase 2b aggregation
     aggregation: enable phase 3 | bool | true | * | * |   |Turn on/off phase 3 aggregation
     aggregation: preserve Dirichlet points | bool | false | * | * |   | preserve Dirichlet points as singleton nodes (default=false, i.e., drop Dirichlet nodes during aggregation)
     aggregation: allow user-specified singletons | bool | false | * | * |  | Turn on/off OnePtAggregationAlgorithm (default=false)


    The * in the @c master.xml column denotes that the parameter is defined in the @c master.xml file.<br>
    The * in the @c validated column means that the parameter is declared in the list of valid input parameters (see UncoupledAggregationFactory::GetValidParameters).<br>
    The * in the @c requested column states that the data is requested as input with all dependencies (see UncoupledAggregationFactory::DeclareInput).

    ### Variables provided by UncoupledAggregationFactory ###

    After UncoupledAggregationFactory::Build the following data is available (if requested)

    Parameter | generated by | description
    ----------|--------------|------------
    | Aggregates   | UncoupledAggregationFactory   | Container class with aggregation information. See also Aggregates.
*/

template <class LocalOrdinal  = DefaultLocalOrdinal,
          class GlobalOrdinal = DefaultGlobalOrdinal,
          class Node          = DefaultNode>
class AggressiveAggregationFactory : public SingleLevelFactoryBase {
#undef MUELU_AGGRESSIVEAGGREGATIONFACTORY_SHORT
#include "MueLu_UseShortNamesOrdinal.hpp"

 public:
  //! @name Constructors/Destructors.
  //@{

  //! Constructor.
  AggressiveAggregationFactory();

  //! Destructor.
  virtual ~AggressiveAggregationFactory();

  RCP<const ParameterList> GetValidParameterList() const;

  //@}

  //! @name Set/get methods.
  //@{

  //! Input
  //@{

  void DeclareInput(Level& currentLevel) const;

  //@}

  //! @name Build methods.
  //@{

  /*! @brief Build aggregates. */
  void Build(Level& currentLevel) const;

  //@}

};  // class AggressiveAggregationFactory

}  // namespace MueLu

#define MUELU_AGGRESSIVEAGGREGATIONFACTORY_SHORT
#endif /* MUELU_AGGRESSIVEAGGREGATIONFACTORY_DECL_HPP_ */
