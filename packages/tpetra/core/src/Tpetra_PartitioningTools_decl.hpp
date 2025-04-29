// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
//
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef TPETRA_PARTITIONINGTOOLS_DECL_HPP
#define TPETRA_PARTITIONINGTOOLS_DECL_HPP

#include "Teuchos_RCP.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_Import.hpp"
#include "Tpetra_Export.hpp"
#include "Tpetra_Vector.hpp"


namespace Tpetra {

  /// @brief Tools that help with (re)partitionining.
  ///
  /// This class helps to construct an importer from a partitioning solution in the form of a ordinal valued vector.
  /// Such a solution can, for example, be constructed using Zoltan/Zoltan2.
  ///
  template <class PartitionVectorType>
  class PartitioningTools {

  private:

    using PartitionNumber = typename PartitionVectorType::scalar_type;
    using LocalOrdinal  = typename PartitionVectorType::local_ordinal_type;
    using GlobalOrdinal = typename PartitionVectorType::global_ordinal_type;
    using Node      = typename PartitionVectorType::node_type;

    using map_type = Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node>;
    using import_type = Tpetra::Import<LocalOrdinal, GlobalOrdinal, Node>;
    using export_type = Tpetra::Export<LocalOrdinal, GlobalOrdinal, Node>;

  public:

    /// \brief Reassign partition numbers with the goal of reducing the communication volume.
    /// \param decomposition [in/out] Vector with partition numbers
    /// \param weights [in] Vector with weights
    /// \param numPartitions [in]
    /// \param willAcceptPartition [in]
    /// \param allSubdomainsAcceptPartitions [in]
    /// \param maxLocalEdges [in]
    static
    void
    ReduceCommVolume(PartitionVectorType& decomposition,
                     const Teuchos::RCP<PartitionVectorType>& weights = Teuchos::null,
                     PartitionNumber numPartitions = Teuchos::OrdinalTraits<PartitionNumber>::invalid(),
                     bool willAcceptPartition = true,
                     bool allSubdomainsAcceptPartitions = true,
                     int maxLocalEdges = 4);

    /// \brief Construct an importer from a vector of partition numbers.
    ///
    /// \param parts [in] Vector of partition numbers
    ///
    /// Returns an importer.
    static
    Teuchos::RCP<import_type>
    buildImporterFromPartitioningAlltoall(const Teuchos::RCP<PartitionVectorType>& parts);

    /// \brief Construct an importer from a vector of partition numbers.
    ///
    /// \param parts [in] Vector of partition numbers
    ///
    /// Returns an importer.
    static
    Teuchos::RCP<import_type>
    buildImporterFromPartitioningSendRecv(const Teuchos::RCP<PartitionVectorType>& parts);
  };
}

#endif
