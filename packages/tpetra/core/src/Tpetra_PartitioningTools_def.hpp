// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
//
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef TPETRA_PARTITIONINGTOOLS_DEF_HPP
#define TPETRA_PARTITIONINGTOOLS_DEF_HPP

#include "Tpetra_PartitioningTools_decl.hpp"


namespace Tpetra {

template <typename T, typename W>
struct Triplet {
  T i, j;
  W v;
};
template <typename T, typename W>
static bool compareTriplets(const Triplet<T, W>& a, const Triplet<T, W>& b) {
  return (a.v > b.v);  // descending order
}


template <class PartitionVectorType>
void
PartitioningTools<PartitionVectorType>::
ReduceCommVolume(PartitionVectorType& decomposition,
                 const Teuchos::RCP<PartitionVectorType>& weights,
                 const typename PartitionVectorType::scalar_type numPartitions,
                 const bool willAcceptPartition,
                 const bool allSubdomainsAcceptPartitions,
                 const int maxLocalEdges) {

  auto rowMap = decomposition.getMap();

  Teuchos::RCP<const Teuchos::Comm<int> > comm = rowMap->getComm()->duplicate();
  int numProcs                        = comm->getSize();

  Teuchos::RCP<const Teuchos::MpiComm<int> > tmpic = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int> >(comm);
  TEUCHOS_TEST_FOR_EXCEPTION(tmpic == Teuchos::null, std::runtime_error, "Cannot cast base Teuchos::Comm to Teuchos::MpiComm object.");
  Teuchos::RCP<const Teuchos::OpaqueWrapper<MPI_Comm> > rawMpiComm = tmpic->getRawMpiComm();

  // maxLocalEdges is a constant which determines the number of largest edges which are being exchanged
  // The idea is that we do not want to construct the full bipartite graph, but simply a subset of
  // it, which requires less communication. By selecting largest local edges we hope to achieve
  // similar results but at a lower cost.
  const int dataSize = 2 * maxLocalEdges;

  typename PartitionVectorType::dual_view_type::t_host_um decompEntries;
  if (decomposition.getLocalLength() > 0) {
    decompEntries = decomposition.getLocalViewHost(Access::ReadWrite);
  }

  // Step 1: Sort local edges by weight
  // Each edge of a bipartite graph corresponds to a triplet (i, j, v) where
  //   i: processor id that has some piece of part with part_id = j
  //   j: part id
  //   v: weight of the edge
  // We set edge weights to be the total number of nonzeros in rows on this processor which
  // correspond to this part_id. The idea is that when we redistribute matrix, this weight
  // is a good approximation of the amount of data to move.
  // We use two maps, original which maps a partition id of an edge to the corresponding weight,
  // and a reverse one, which is necessary to sort by edges.
  std::map<GlobalOrdinal, GlobalOrdinal> lEdges;
  if (willAcceptPartition) {
    if (!weights.is_null()) {
      auto lclWeight = weights->getLocalViewHost(Access::ReadOnly);
      for (LocalOrdinal i = 0; i < decompEntries.extent(0); i++)
        lEdges[decompEntries(i, 0)] += lclWeight(i, 0);
    } else {
      for (LocalOrdinal i = 0; i < decompEntries.extent(0); i++)
        ++lEdges[decompEntries(i, 0)];
    }
  }

  // Reverse map, so that edges are sorted by weight.
  // This results in multimap, as we may have edges with the same weight
  std::multimap<GlobalOrdinal, GlobalOrdinal> revlEdges;
  for (auto it = lEdges.begin(); it != lEdges.end(); it++)
    revlEdges.insert(std::make_pair(it->second, it->first));

  // Both lData and gData are arrays of data which we communicate. The data is stored
  // in pairs, so that data[2*i+0] is the part index, and data[2*i+1] is the corresponding edge weight.
  // We do not store processor id in data, as we can compute that by looking on the offset in the gData.
  Teuchos::Array<GlobalOrdinal> lData(dataSize, -1);
  Teuchos::Array<GlobalOrdinal> gData(numProcs * dataSize);
  int numEdges = 0;
  for (auto rit = revlEdges.rbegin(); rit != revlEdges.rend() && numEdges < maxLocalEdges; rit++) {
    lData[2 * numEdges + 0] = rit->second;  // part id
    lData[2 * numEdges + 1] = rit->first;   // edge weight
    numEdges++;
  }

  // Step 2: Gather most edges
  // Each processors contributes maxLocalEdges edges by providing maxLocalEdges pairs <part id, weight>, which is of size dataSize
  MPI_Datatype MpiType = Teuchos::Details::MpiTypeTraits<GlobalOrdinal>::getType();
  MPI_Allgather(static_cast<void*>(lData.getRawPtr()), dataSize, MpiType, static_cast<void*>(gData.getRawPtr()), dataSize, MpiType, *rawMpiComm);

  // Step 3: Construct mapping

  // Construct the set of triplets
  Teuchos::Array<Triplet<int, int> > gEdges(numProcs * maxLocalEdges);
  Teuchos::Array<bool> procWillAcceptPartition(numProcs, allSubdomainsAcceptPartitions);
  size_t k = 0;
  for (LocalOrdinal i = 0; i < gData.size(); i += 2) {
    int procNo           = i / dataSize;  // determine the processor by its offset (since every processor sends the same amount)
    GlobalOrdinal part   = gData[i + 0];
    GlobalOrdinal weight = gData[i + 1];
    if (part != -1) {  // skip nonexistent edges
      gEdges[k].i                     = procNo;
      gEdges[k].j                     = part;
      gEdges[k].v                     = weight;
      procWillAcceptPartition[procNo] = true;
      k++;
    }
  }
  gEdges.resize(k);

  // Sort edges by weight
  // NOTE: compareTriplets is actually a reverse sort, so the edges weight is in decreasing order
  std::sort(gEdges.begin(), gEdges.end(), compareTriplets<int, int>);

  // Do matching
  std::map<int, int> match;
  Teuchos::Array<char> matchedRanks(numProcs, 0);
  Teuchos::Array<char> matchedParts(numPartitions, 0);
  int numMatched = 0;
  for (typename Teuchos::Array<Triplet<int, int> >::const_iterator it = gEdges.begin(); it != gEdges.end(); it++) {
    GlobalOrdinal rank = it->i;
    GlobalOrdinal part = it->j;
    if (matchedRanks[rank] == 0 && matchedParts[part] == 0) {
      matchedRanks[rank] = 1;
      matchedParts[part] = 1;
      match[part]        = rank;
      numMatched++;
    }
  }
  // GetOStream(Statistics1) << "Number of unassigned partitions before cleanup stage: " << (numPartitions - numMatched) << " / " << numPartitions << std::endl;

  // Step 4: Assign unassigned partitions if necessary.
  // We do that through desperate matching for remaining partitions:
  // We select the lowest rank that can still take a partition.
  // The reason it is done this way is that we don't need any extra communication, as we don't
  // need to know which parts are valid.
  if (numPartitions - numMatched > 0) {
    Teuchos::Array<char> partitionCounts(numPartitions, 0);
    for (auto it = match.begin(); it != match.end(); it++)
      partitionCounts[it->first] += 1;
    for (int part = 0, matcher = 0; part < numPartitions; part++) {
      if (partitionCounts[part] == 0) {
        // Find first non-matched rank that accepts partitions
        while (matchedRanks[matcher] || !procWillAcceptPartition[matcher])
          matcher++;

        match[part] = matcher++;
        numMatched++;
      }
    }
  }

  TEUCHOS_TEST_FOR_EXCEPTION(numMatched != numPartitions, std::runtime_error, "MueLu::RepartitionFactory::DeterminePartitionPlacement: Only " << numMatched << " partitions out of " << numPartitions << " got assigned to ranks.");

  // Step 5: Permute entries in the decomposition vector
  for (LocalOrdinal i = 0; i < decompEntries.extent(0); i++)
    decompEntries(i, 0) = match[decompEntries(i, 0)];
}

template <class PartitionVectorType>
Teuchos::RCP<typename PartitioningTools<PartitionVectorType>::import_type>
PartitioningTools<PartitionVectorType>::
buildImporterFromPartitioningAlltoall(const Teuchos::RCP<PartitionVectorType>& decomposition) {
  auto map                                             = decomposition->getMap();
  auto comm                                            = map->getComm();
  Teuchos::RCP<const Teuchos::OpaqueWrapper<MPI_Comm> > rawComm = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int> >(comm, true)->getRawMpiComm();

  MPI_Datatype MpiTypeInt           = Teuchos::Details::MpiTypeTraits<int>::getType();
  MPI_Datatype MpiTypeGlobalOrdinal = Teuchos::Details::MpiTypeTraits<GlobalOrdinal>::getType();
  int status                        = -1;
  int numRanks                      = comm->getSize();
  int numDecomposition                      = decomposition->normInf();
  TEUCHOS_ASSERT(numDecomposition >= 1);
  LocalOrdinal numElements = map->getLocalNumElements();

  auto lclDecomposition = decomposition->getLocalViewHost(Access::ReadOnly);

  std::vector<int> send_counts(numRanks);
  std::vector<int> send_offsets(numDecomposition + 1);
  std::vector<GlobalOrdinal> send_gids(numElements);
  std::vector<int> recv_counts(numRanks);
  std::vector<int> recv_offsets(numRanks + 1);

  for (GlobalOrdinal i = 0; i < numElements; i++) {
    int partNum = lclDecomposition(i, 0);
    ++send_counts[partNum];
  }

  send_offsets[0] = 0;
  send_offsets[1] = 0;
  // send_offsets[2] = send_counts[0];
  // send_offsets[3] = send_counts[0]+send_counts[1];
  // etc
  for (GlobalOrdinal i = 1; i < numDecomposition; i++) {
    send_offsets[i + 1] = send_offsets[i] + send_counts[i - 1];
  }

  // We increment send_offsets as we enter values into send_gids.
  for (GlobalOrdinal i = 0; i < numElements; i++) {
    int partNum                            = lclDecomposition(i, 0);
    send_gids[send_offsets[partNum + 1]++] = map->getGlobalElement(i);
  }
  // Now:
  // send_offsets[0] = 0;
  // send_offsets[1] = send_counts[0];
  // send_offsets[2] = send_counts[0]+send_counts[1];
  // etc

  status = MPI_Alltoall(send_counts.data(), 1, MpiTypeInt,
                        recv_counts.data(), 1, MpiTypeInt,
                        (*rawComm)());
  TEUCHOS_ASSERT(status == 0);

  recv_offsets[0] = 0;
  for (GlobalOrdinal i = 0; i < numRanks; i++) {
    recv_offsets[i + 1] = recv_offsets[i] + recv_counts[i];
  }

  std::vector<GlobalOrdinal> recv_gids(recv_offsets[numRanks]);

  status = MPI_Alltoallv(send_gids.data(), send_counts.data(), send_offsets.data(), MpiTypeGlobalOrdinal,
                         recv_gids.data(), recv_counts.data(), recv_offsets.data(), MpiTypeGlobalOrdinal,
                         (*rawComm)());
  TEUCHOS_ASSERT(status == 0);

  // NOTE 2: The general sorting algorithm could be sped up by using the knowledge that original myGIDs and all received chunks
  // (i.e. it->second) are sorted. Therefore, a merge sort would work well in this situation.
  std::sort(recv_gids.begin(), recv_gids.end());

  auto newMap   = rcp(new map_type(map->getGlobalNumElements(), recv_gids, map->getIndexBase(), comm));
  auto importer = rcp(new import_type(map, newMap));
  return importer;
}

template <class PartitionVectorType>
Teuchos::RCP<typename PartitioningTools<PartitionVectorType>::import_type>
PartitioningTools<PartitionVectorType>::
buildImporterFromPartitioningSendRecv(const Teuchos::RCP<PartitionVectorType>& decomposition) {
  using LO = typename PartitionVectorType::local_ordinal_type;
  using GO = typename PartitionVectorType::global_ordinal_type;

  auto rowMap  = decomposition->getMap();
  auto comm    = rowMap->getComm();
  auto myRank  = comm->getRank();
  int numProcs = comm->getSize();

  Teuchos::RCP<const Teuchos::MpiComm<int> > tmpic = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int> >(comm);
  TEUCHOS_TEST_FOR_EXCEPTION(tmpic == Teuchos::null, std::runtime_error, "Cannot cast base Teuchos::Comm to Teuchos::MpiComm object.");
  Teuchos::RCP<const Teuchos::OpaqueWrapper<MPI_Comm> > rawMpiComm = tmpic->getRawMpiComm();

  Teuchos::ArrayRCP<const PartitionNumber> decompEntries;
  if (decomposition->getLocalLength() > 0)
    decompEntries = decomposition->getData(0);

#ifdef HAVE_MUELU_DEBUG
  // Test range of partition ids
  int incorrectRank = -1;
  for (int i = 0; i < decompEntries.size(); i++)
    if (decompEntries[i] >= numProcs || decompEntries[i] < 0) {
      incorrectRank = myRank;
      break;
    }

  int incorrectGlobalRank = -1;
  MueLu_maxAll(comm, incorrectRank, incorrectGlobalRank);
  TEUCHOS_TEST_FOR_EXCEPTION(incorrectGlobalRank > -1, Exceptions::RuntimeError, "pid " + Teuchos::toString(incorrectGlobalRank) + " encountered a partition number is that out-of-range");
#endif

  Teuchos::Array<GO> myGIDs;
  myGIDs.reserve(decomposition->getLocalLength());

  // Step 0: Construct mapping
  //    part number -> GIDs I own which belong to this part
  // NOTE: my own part GIDs are not part of the map
  using map_t = std::map<GO, Teuchos::Array<GO>>;
  map_t sendMap;
  for (LO i = 0; i < decompEntries.size(); i++) {
    GO id  = decompEntries[i];
    GO GID = rowMap->getGlobalElement(i);

    if (id == myRank)
      myGIDs.push_back(GID);
    else
      sendMap[id].push_back(GID);
  }
  decompEntries = Teuchos::null;

  int numSend = sendMap.size();
  int numRecv;

  // Arrayify map keys
  Teuchos::Array<GO> myParts(numSend);
  Teuchos::Array<GO> myPart(1);
  int cnt   = 0;
  myPart[0] = myRank;
  for (auto it = sendMap.begin(); it != sendMap.end(); it++)
    myParts[cnt++] = it->first;

  // Step 1: Find out how many processors send me data
  // partsIndexBase starts from zero, as the processors ids start from zero
  {
    // SubFactoryMonitor m1(*this, "Mapping Step 1", currentLevel);
    GO partsIndexBase = 0;
    auto partsIHave   = Teuchos::rcp(new map_type(Teuchos::OrdinalTraits<GO>::invalid(), myParts.data(), myParts.size(), partsIndexBase, comm));
    auto partsIOwn    = Teuchos::rcp(new map_type(numProcs, myPart(), partsIndexBase, comm));
    auto partsExport  = Teuchos::rcp(new export_type(partsIHave, partsIOwn));

    auto partsISend    = Teuchos::rcp(new PartitionVectorType(partsIHave));
    auto numPartsIRecv = Teuchos::rcp(new PartitionVectorType(partsIOwn));
    if (numSend > 0) {
      auto partsISendData = partsISend->getDataNonConst(0);
      for (int i = 0; i < numSend; i++)
        partsISendData[i] = 1;
    }
    (numPartsIRecv->getDataNonConst(0))[0] = 0;

    numPartsIRecv->doExport(*partsISend, *partsExport, Tpetra::ADD);
    numRecv = (numPartsIRecv->getData(0))[0];
  }

  // Step 2: Get my GIDs from everybody else
  MPI_Datatype MpiType = Teuchos::Details::MpiTypeTraits<GO>::getType();
  int msgTag           = 12345;  // TODO: use Comm::dup for all internal messaging

  // Post sends
  Teuchos::Array<MPI_Request> sendReqs(numSend);
  cnt = 0;
  for (auto it = sendMap.begin(); it != sendMap.end(); it++)
    MPI_Isend(static_cast<void*>(it->second.getRawPtr()), it->second.size(), MpiType, Teuchos::as<GO>(it->first), msgTag, *rawMpiComm, &sendReqs[cnt++]);

  map_t recvMap;
  size_t totalGIDs = myGIDs.size();
  for (int i = 0; i < numRecv; i++) {
    MPI_Status status;
    MPI_Probe(MPI_ANY_SOURCE, msgTag, *rawMpiComm, &status);

    // Get rank and number of elements from status
    int fromRank = status.MPI_SOURCE;
    int count;
    MPI_Get_count(&status, MpiType, &count);

    recvMap[fromRank].resize(count);
    MPI_Recv(static_cast<void*>(recvMap[fromRank].getRawPtr()), count, MpiType, fromRank, msgTag, *rawMpiComm, &status);

    totalGIDs += count;
  }

  // Do waits on send requests
  if (numSend) {
    Teuchos::Array<MPI_Status> sendStatuses(numSend);
    MPI_Waitall(numSend, sendReqs.getRawPtr(), sendStatuses.getRawPtr());
  }

  // Merge GIDs
  myGIDs.reserve(totalGIDs);
  for (auto it = recvMap.begin(); it != recvMap.end(); it++) {
    int offset = myGIDs.size();
    int len    = it->second.size();
    if (len) {
      myGIDs.resize(offset + len);
      memcpy(myGIDs.getRawPtr() + offset, it->second.getRawPtr(), len * sizeof(GO));
    }
  }
  // NOTE 2: The general sorting algorithm could be sped up by using the knowledge that original myGIDs and all received chunks
  // (i.e. it->second) are sorted. Therefore, a merge sort would work well in this situation.
  std::sort(myGIDs.begin(), myGIDs.end());

  // Step 3: Construct importer
  auto newRowMap = Teuchos::rcp(new map_type(rowMap->getGlobalNumElements(), myGIDs(), rowMap->getIndexBase(), comm));
  auto importer = Teuchos::rcp(new import_type(rowMap, newRowMap));
  return importer;
}

} // namespace Tpetra


#define TPETRA_PARTITIONINGTOOLS_INSTANT(LO, GO, NO) \
  template class PartitioningTools<typename Tpetra::Vector<LO, LO, GO, NO> >;


#endif // TPETRA_PARTITIONINGTOOLS_DEF_HPP
