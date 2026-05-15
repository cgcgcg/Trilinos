// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
//
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef TPETRA_EXPORT_DEF_HPP
#define TPETRA_EXPORT_DEF_HPP

#include "Tpetra_Export_decl.hpp"
#include "Tpetra_Distributor.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_ImportExportData.hpp"
#include "Tpetra_Util.hpp"
#include "Tpetra_Import.hpp"
#include "Tpetra_Details_DualViewUtil.hpp"
#include "Tpetra_Details_Profiling.hpp"
#include "Teuchos_FancyOStream.hpp"
#include "Teuchos_ParameterList.hpp"

template <class LO>
struct pair_lo {
  LO numPermutes;
  LO numExports;

  KOKKOS_INLINE_FUNCTION
  pair_lo() {
    numPermutes = 0;
    numExports  = 0;
  }

  KOKKOS_INLINE_FUNCTION
  pair_lo(const pair_lo& rhs) {
    numPermutes = rhs.numPermutes;
    numExports  = rhs.numExports;
  }

  KOKKOS_INLINE_FUNCTION
  pair_lo& operator+=(const pair_lo& src) {
    numPermutes += src.numPermutes;
    numExports += src.numExports;
    return *this;
  }
};

namespace Kokkos {

template <class LO>
struct reduction_identity<pair_lo<LO>> {
  KOKKOS_FORCEINLINE_FUNCTION static pair_lo<LO> sum() {
    return pair_lo<LO>();
  }
};
}  // namespace Kokkos

namespace Tpetra {

template <class LocalOrdinal, class GlobalOrdinal, class Node>
Export<LocalOrdinal, GlobalOrdinal, Node>::
    Export(const Teuchos::RCP<const typename Export<LocalOrdinal, GlobalOrdinal, Node>::map_type>& source,
           const Teuchos::RCP<const typename Export<LocalOrdinal, GlobalOrdinal, Node>::map_type>& target,
           const Teuchos::RCP<Teuchos::FancyOStream>& out,
           const Teuchos::RCP<Teuchos::ParameterList>& plist)
  : base_type(source, target, out, plist, "Export") {
  using std::endl;
  using ::Tpetra::Details::ProfilingRegion;
  ProfilingRegion regionExport("Tpetra::Export::Export");

  if (this->verbose()) {
    std::ostringstream os;
    const int myRank = source->getComm()->getRank();
    os << myRank << ": Export ctor" << endl;
    this->verboseOutputStream() << os.str();
  }
  auto exportGIDs = setupSamePermuteExport();
  if (source->isDistributed()) {
    setupRemote(exportGIDs);
  }

  TEUCHOS_ASSERT(!this->TransferData_->permuteFromLIDs_.need_sync_device());
  TEUCHOS_ASSERT(!this->TransferData_->permuteFromLIDs_.need_sync_host());
  TEUCHOS_ASSERT(!this->TransferData_->permuteToLIDs_.need_sync_device());
  TEUCHOS_ASSERT(!this->TransferData_->permuteToLIDs_.need_sync_host());
  TEUCHOS_ASSERT(!this->TransferData_->remoteLIDs_.need_sync_device());
  TEUCHOS_ASSERT(!this->TransferData_->remoteLIDs_.need_sync_host());
  TEUCHOS_ASSERT(!this->TransferData_->exportLIDs_.need_sync_device());
  TEUCHOS_ASSERT(!this->TransferData_->exportLIDs_.need_sync_host());

  this->detectRemoteExportLIDsContiguous();

  if (this->verbose()) {
    std::ostringstream os;
    const int myRank = source->getComm()->getRank();
    os << myRank << ": Export ctor: done" << endl;
    this->verboseOutputStream() << os.str();
  }
}

template <class LocalOrdinal, class GlobalOrdinal, class Node>
Export<LocalOrdinal, GlobalOrdinal, Node>::
    Export(const Teuchos::RCP<const map_type>& source,
           const Teuchos::RCP<const map_type>& target)
  : Export(source, target, Teuchos::null, Teuchos::null) {}

template <class LocalOrdinal, class GlobalOrdinal, class Node>
Export<LocalOrdinal, GlobalOrdinal, Node>::
    Export(const Teuchos::RCP<const map_type>& source,
           const Teuchos::RCP<const map_type>& target,
           const Teuchos::RCP<Teuchos::FancyOStream>& out)
  : Export(source, target, out, Teuchos::null) {}

template <class LocalOrdinal, class GlobalOrdinal, class Node>
Export<LocalOrdinal, GlobalOrdinal, Node>::
    Export(const Teuchos::RCP<const map_type>& source,
           const Teuchos::RCP<const map_type>& target,
           const Teuchos::RCP<Teuchos::ParameterList>& plist)
  : Export(source, target, Teuchos::null, plist) {}

template <class LocalOrdinal, class GlobalOrdinal, class Node>
Export<LocalOrdinal, GlobalOrdinal, Node>::
    Export(const Export<LocalOrdinal, GlobalOrdinal, Node>& rhs)
  : base_type(rhs) {}

template <class LocalOrdinal, class GlobalOrdinal, class Node>
Export<LocalOrdinal, GlobalOrdinal, Node>::
    Export(const Import<LocalOrdinal, GlobalOrdinal, Node>& importer)
  : base_type(importer, typename base_type::reverse_tag()) {}

template <class LocalOrdinal, class GlobalOrdinal, class Node>
void Export<LocalOrdinal, GlobalOrdinal, Node>::
    describe(Teuchos::FancyOStream& out,
             const Teuchos::EVerbosityLevel verbLevel) const {
  // Call the base class' method.  It does all the work.
  this->describeImpl(out, "Tpetra::Export", verbLevel);
}

template <class LocalOrdinal, class GlobalOrdinal, class Node>
void Export<LocalOrdinal, GlobalOrdinal, Node>::
    print(std::ostream& os) const {
  auto out = Teuchos::getFancyOStream(Teuchos::rcpFromRef(os));
  // "Print" traditionally meant "everything."
  this->describe(*out, Teuchos::VERB_EXTREME);
}

template <class LocalOrdinal, class GlobalOrdinal, class Node>
Export<LocalOrdinal, GlobalOrdinal, Node>::gids_view_type Export<LocalOrdinal, GlobalOrdinal, Node>::
    setupSamePermuteExport() {
  using std::endl;
  using Teuchos::ArrayView;
  using ::Tpetra::Details::ProfilingRegion;
  using ::Tpetra::Details::view_alloc_no_init;
  using GO                   = GlobalOrdinal;
  using size_type            = typename ArrayView<const GO>::size_type;
  const char tfecfFuncName[] = "setupSamePermuteExport: ";
  ProfilingRegion regionExport("Tpetra::Export::setupSamePermuteExport");

  std::unique_ptr<std::string> prefix;
  if (this->verbose()) {
    auto srcMap      = this->getSourceMap();
    auto comm        = srcMap.is_null() ? Teuchos::null : srcMap->getComm();
    const int myRank = comm.is_null() ? -1 : comm->getRank();

    std::ostringstream os;
    os << "Proc " << myRank << ": Tpetra::Export::setupSamePermuteExport: ";
    prefix = std::unique_ptr<std::string>(new std::string(os.str()));

    std::ostringstream os2;
    os2 << *prefix << "Start" << std::endl;
    this->verboseOutputStream() << os2.str();
  }

  const map_type& source = *(this->getSourceMap());
  const map_type& target = *(this->getTargetMap());
  auto sourceGIDs        = source.getMyGlobalIndicesDevice();
  auto targetGIDs        = target.getMyGlobalIndicesDevice();

  const size_type numSrcGids = sourceGIDs.extent(0);
  const size_type numTgtGids = targetGIDs.extent(0);
  const size_type numGids    = std::min(numSrcGids, numTgtGids);

  // Compute numSameIDs_: the number of initial GIDs that are the
  // same (and occur in the same order) in both Maps.  The point of
  // numSameIDs_ is for the common case of an Export where all the
  // overlapping GIDs are at the end of the source Map, but
  // otherwise the source and target Maps are the same.  This allows
  // a fast contiguous copy for the initial "same IDs."
  size_type numSameGids = 0;

  Kokkos::parallel_reduce(
      Kokkos::RangePolicy<execution_space>(0, numGids), KOKKOS_LAMBDA(const size_type lid, size_type& mySameGids) {
        if (sourceGIDs(lid) != targetGIDs(lid))
          mySameGids = Kokkos::min(lid, mySameGids);
      },
      Kokkos::Min<size_type>(numSameGids));
  if (numSameGids == Kokkos::Experimental::finite_max_v<size_type>)
    numSameGids = numGids;

  this->TransferData_->numSameIDs_ = numSameGids;

  if (this->verbose()) {
    std::ostringstream os;
    os << *prefix << "numIDs: " << numGids
       << ", numSameIDs: " << numSameGids << endl;
    this->verboseOutputStream() << os.str();
  }

  // Compute permuteToLIDs_, permuteFromLIDs_, exportGIDs, and
  // exportLIDs_.  The first two arrays are IDs to be permuted, and
  // the latter two arrays are IDs to sent out ("exported"), called
  // "export" IDs.
  //
  // IDs to permute are in both the source and target Maps, which
  // means we don't have to send or receive them, but we do have to
  // rearrange (permute) them in general.  IDs to send are in the
  // source Map, but not in the target Map.

  // Iterate over the source Map's LIDs, since we only need to do
  // GID -> LID lookups for the target Map.
  const LocalOrdinal LINVALID   = Teuchos::OrdinalTraits<LocalOrdinal>::invalid();
  const LocalOrdinal numSrcLids = static_cast<LocalOrdinal>(numSrcGids);
  LocalOrdinal numPermutes      = 0;
  LocalOrdinal numExports       = 0;

  auto lclTarget = target.getLocalMap();
  Kokkos::parallel_reduce(
      Kokkos::RangePolicy<execution_space>(numSameGids, numSrcLids), KOKKOS_LAMBDA(const LocalOrdinal srcLid, LocalOrdinal& myNumPermutes, LocalOrdinal& myNumExports) {
        const auto curSrcGid = sourceGIDs(srcLid);
        // getLocalElement() returns LINVALID if the GID isn't in the
        // target Map.  This saves us a lookup (which
        // isNodeGlobalElement() would do).
        const auto tgtLid = lclTarget.getLocalElement(curSrcGid);
        if (tgtLid != LINVALID) {  // if target.isNodeGlobalElement (curSrcGid)
          ++myNumPermutes;
        } else {
          ++myNumExports;
        }
      },
      numPermutes, numExports);

  if (this->verbose()) {
    std::ostringstream os;
    os << *prefix << "numPermutes: " << numPermutes
       << ", numExports: " << numExports << endl;
    this->verboseOutputStream() << os.str();
  }
  TEUCHOS_ASSERT(numPermutes + numExports ==
                 numSrcLids - numSameGids);

  typename decltype(this->TransferData_->permuteToLIDs_)::t_dev
      permuteToLIDs(view_alloc_no_init("permuteToLIDs"), numPermutes);
  typename decltype(this->TransferData_->permuteToLIDs_)::t_dev
      permuteFromLIDs(view_alloc_no_init("permuteFromLIDs"), numPermutes);
  typename decltype(this->TransferData_->permuteToLIDs_)::t_dev
      exportLIDs(view_alloc_no_init("exportLIDs"), numExports);

  gids_view_type exportGIDs(view_alloc_no_init("exportGIDs"), numExports);
  {
    using my_pair = pair_lo<LocalOrdinal>;
    my_pair counts;
    Kokkos::parallel_scan(
        Kokkos::RangePolicy<LocalOrdinal, execution_space>(numSameGids, numSrcLids),
        KOKKOS_LAMBDA(const LocalOrdinal srcLid, my_pair& offsets, const bool is_final) {
          auto& myNumPermutes  = offsets.numPermutes;
          auto& myNumExports   = offsets.numExports;
          const auto curSrcGid = sourceGIDs(srcLid);
          const auto tgtLid    = lclTarget.getLocalElement(curSrcGid);
          if (tgtLid != LINVALID) {
            if (is_final) {
              permuteToLIDs(myNumPermutes)   = tgtLid;
              permuteFromLIDs(myNumPermutes) = srcLid;
            }
            ++myNumPermutes;
          } else {
            if (is_final) {
              exportGIDs(myNumExports) = curSrcGid;
              exportLIDs(myNumExports) = srcLid;
            }
            ++myNumExports;
          }
        },
        counts);
    LocalOrdinal numPermutes2 = counts.numPermutes;
    LocalOrdinal numExports2  = counts.numExports;

    TEUCHOS_ASSERT(numPermutes == numPermutes2);
    TEUCHOS_ASSERT(numExports == numExports2);
    TEUCHOS_ASSERT(size_t(numExports) == size_t(exportGIDs.extent(0)));
  }

  // Defer making this->TransferData_->exportLIDs_ until after
  // getRemoteIndexList, since we might need to shrink it then.

  // exportLIDs is the list of this process' LIDs that it has to
  // send out.  Since this is an Export, and therefore the target
  // Map is nonoverlapping, we know that each export LID only needs
  // to be sent to one process.  However, the source Map may be
  // overlapping, so multiple processes might send to the same LID
  // on a receiving process.

  if (numExports != 0 && !source.isDistributed()) {
    // This Export has export LIDs, meaning that the source Map has
    // entries on this process that are not in the target Map on
    // this process.  However, the source Map is not distributed
    // globally.  This implies that this Import is not locally
    // complete on this process.
    this->TransferData_->isLocallyComplete_ = false;
    if (this->verbose()) {
      std::ostringstream os;
      os << *prefix << "Export is not locally complete" << endl;
      this->verboseOutputStream() << os.str();
    }
    // mfh 12 Sep 2016: I disagree that this is "abuse"; it may be
    // correct behavior, depending on the circumstances.
    TPETRA_ABUSE_WARNING(true, std::runtime_error,
                         "::setupSamePermuteExport(): Source has "
                         "export LIDs but Source is not distributed globally.  Exporting to "
                         "a submap of the target map.");
  }

  // Compute exportPIDs_ ("outgoing" process IDs).
  //
  // For each GID in exportGIDs (GIDs to which this process must
  // send), find its corresponding owning process (a.k.a. "image")
  // ID in the target Map.  Store these process IDs in
  // exportPIDs_.  These are the process IDs to which the Export
  // needs to send data.
  //
  // We only need to do this if the source Map is distributed;
  // otherwise, the Export doesn't have to perform any
  // communication.
  pids_view_type exportPIDs;
  if (source.isDistributed()) {
    if (this->verbose()) {
      std::ostringstream os;
      os << *prefix << "Source Map is distributed; "
                       "call targetMap.getRemoteiNdexList"
         << endl;
      this->verboseOutputStream() << os.str();
    }
    // This call will assign any GID in the target Map with no
    // corresponding process ID a fake process ID of -1.  We'll use
    // this below to remove exports for processses that don't exist.
    exportPIDs          = pids_view_type("exportPIDs", exportGIDs.size());
    LookupStatus lookup = target.getRemoteIndexList(exportGIDs,
                                                    exportPIDs);
    // mfh 12 Sep 2016: I disagree that this is "abuse"; it may be
    // correct behavior, depending on the circumstances.
    TPETRA_ABUSE_WARNING(lookup == IDNotPresent, std::runtime_error,
                         "::setupSamePermuteExport(): The source Map has GIDs not found "
                         "in the target Map.");

    // Get rid of process IDs not in the target Map.  This prevents
    // exporting to GIDs which don't belong to any process in the
    // target Map.
    if (lookup == IDNotPresent) {
      // There is at least one GID owned by the calling process in
      // the source Map, which is not owned by any process in the
      // target Map.
      this->TransferData_->isLocallyComplete_ = false;

      const size_type totalNumExports = exportPIDs.extent(0);
      size_type numInvalidExports;
      Kokkos::parallel_reduce(
          Kokkos::RangePolicy<execution_space>(0, exportPIDs.extent(0)), KOKKOS_LAMBDA(const LocalOrdinal e, size_type& myNumInvalidExports) {
            if (exportPIDs(e) == -1)
              ++myNumInvalidExports;
          },
          numInvalidExports);

      if (this->verbose()) {
        std::ostringstream os;
        os << *prefix << "totalNumExports: " << totalNumExports
           << ", numInvalidExports: " << numInvalidExports << endl;
        this->verboseOutputStream() << os.str();
      }
      TEUCHOS_TEST_FOR_EXCEPTION_CLASS_FUNC(numInvalidExports == 0, std::logic_error,
                                            "targetMap.getRemoteIndexList returned IDNotPresent, but no export "
                                            "PIDs are -1.  Please report this bug to the Tpetra developers.");

      // We know that at least one export ID is invalid, that is,
      // not in any process on the target Map.  If all export IDs
      // are invalid, we can delete all exports.  Otherwise, keep
      // the valid exports and discard the rest.  This is legit
      // Petra Object Model behavior, but it's a less common case.

      if (numInvalidExports == totalNumExports) {
        exportGIDs = gids_view_type();
        exportLIDs = lids_view_type();
        exportPIDs = pids_view_type();
      } else {
        size_type numValidExports = 0;

        // count valid exports
        Kokkos::parallel_reduce(
            Kokkos::RangePolicy<execution_space>(0, totalNumExports), KOKKOS_LAMBDA(const size_type e, size_type& myNumValidExports) {
              if (exportPIDs(e) != -1)
                ++myNumValidExports;
            },
            numValidExports);

        pids_view_type newExportPIDs("newRemoteProcIDs", numValidExports);
        gids_view_type newExportGIDs("newRemoteGIDs", numValidExports);
        lids_view_type newExportLIDs("newRemoteLIDs", numValidExports);

        Kokkos::parallel_scan(
            Kokkos::RangePolicy<execution_space>(0, totalNumExports), KOKKOS_LAMBDA(const size_type e, size_type& myNumValidExports, const bool is_final) {
              if (exportPIDs(e) != -1) {
                if (is_final) {
                  newExportPIDs(myNumValidExports) = exportPIDs(e);
                  newExportGIDs(myNumValidExports) = exportGIDs(e);
                  newExportLIDs(myNumValidExports) = exportLIDs(e);
                }
                ++myNumValidExports;
              }
            });

        exportPIDs = newExportPIDs;
        exportGIDs = newExportGIDs;
        exportLIDs = newExportLIDs;
      }
    }
  }

  // FIXME (mfh 03 Feb 2019) These three DualViews could share a
  // single device allocation, in order to avoid high cudaMalloc
  // cost and device memory fragmentation.
  Details::makeDualViewFromOwningDeviceView(this->TransferData_->permuteToLIDs_, permuteToLIDs);
  Details::makeDualViewFromOwningDeviceView(this->TransferData_->permuteFromLIDs_, permuteFromLIDs);
  Details::makeDualViewFromOwningDeviceView(this->TransferData_->exportLIDs_, exportLIDs);
  Details::makeDualViewFromOwningDeviceView(this->TransferData_->exportPIDs_, exportPIDs);
  this->TransferData_->exportPIDs_.sync_host();

  if (this->verbose()) {
    std::ostringstream os;
    os << *prefix << "Done!" << std::endl;
    this->verboseOutputStream() << os.str();
  }

  return exportGIDs;
}

template <class LocalOrdinal, class GlobalOrdinal, class Node>
void Export<LocalOrdinal, GlobalOrdinal, Node>::
    setupRemote(gids_view_type& exportGIDs) {
  using std::endl;
  using ::Tpetra::Details::view_alloc_no_init;

  std::unique_ptr<std::string> prefix;
  if (this->verbose()) {
    auto srcMap      = this->getSourceMap();
    auto comm        = srcMap.is_null() ? Teuchos::null : srcMap->getComm();
    const int myRank = comm.is_null() ? -1 : comm->getRank();

    std::ostringstream os;
    os << "Proc " << myRank << ": Tpetra::Export::setupRemote: ";
    prefix = std::unique_ptr<std::string>(new std::string(os.str()));

    std::ostringstream os2;
    os2 << *prefix << "Start" << std::endl;
    this->verboseOutputStream() << os2.str();
  }

  TEUCHOS_ASSERT(!this->getTargetMap().is_null());
  const map_type& tgtMap = *(this->getTargetMap());

  // Sort exportPIDs_ in ascending order, and apply the same
  // permutation to exportGIDs_ and exportLIDs_.  This ensures that
  // exportPIDs_[i], exportGIDs_[i], and exportLIDs_[i] all
  // refer to the same thing.
  {
    TEUCHOS_ASSERT(size_t(this->TransferData_->exportLIDs_.extent(0)) ==
                   size_t(this->TransferData_->exportPIDs_.extent(0)));
    this->TransferData_->exportLIDs_.modify_device();
    auto exportLIDs = this->TransferData_->exportLIDs_.view_device();
    this->TransferData_->exportPIDs_.modify_device();
    auto exportPIDs = this->TransferData_->exportPIDs_.view_device();
    sort3<execution_space>(exportPIDs,
                           exportGIDs,
                           exportLIDs);
    this->TransferData_->exportLIDs_.sync_host();
    // FIXME (mfh 03 Feb 2019) We actually end up sync'ing
    // exportLIDs_ to device twice, once in setupSamePermuteExport,
    // and once here.  We could avoid the first sync.
  }

  if (this->verbose()) {
    std::ostringstream os;
    os << *prefix << "Call createFromSends" << endl;
    this->verboseOutputStream() << os.str();
  }

  // Construct the list of entries that calling image needs to send
  // as a result of everyone asking for what it needs to receive.
  //
  // mfh 05 Jan 2012: I understand the above comment as follows:
  // Construct the communication plan from the list of image IDs to
  // which we need to send.
  Distributor& distributor = this->TransferData_->distributor_;
  this->TransferData_->exportPIDs_.sync_host();
  auto exportPIDs_h         = this->TransferData_->exportPIDs_.view_host();
  auto exportPIDs_av        = Kokkos::Compat::getArrayView(exportPIDs_h);
  const size_t numRemoteIDs = distributor.createFromSends(exportPIDs_av);

  if (this->verbose()) {
    std::ostringstream os;
    os << *prefix << "numRemoteIDs: " << numRemoteIDs
       << "; call doPostsAndWaits" << endl;
    this->verboseOutputStream() << os.str();
  }

  // Use the communication plan with ExportGIDs to find out who is
  // sending to us and get the proper ordering of GIDs for incoming
  // remote entries (these will be converted to LIDs when done).

  gids_view_type remoteGIDs(view_alloc_no_init("remoteGIDs"), numRemoteIDs);
  distributor.doPostsAndWaits(exportGIDs, 1, remoteGIDs);

  // Remote (incoming) IDs come in as GIDs; convert to LIDs.  LIDs
  // tell this process where to store the incoming remote data.
  lids_view_type remoteLIDs(view_alloc_no_init("remoteLIDs"), numRemoteIDs);

  auto lclTgtMap = tgtMap.getLocalMap();
  Kokkos::parallel_for(
      Kokkos::RangePolicy<execution_space>(0, numRemoteIDs), KOKKOS_LAMBDA(const LocalOrdinal j) {
        remoteLIDs(j) = lclTgtMap.getLocalElement(remoteGIDs(j));
      });
  Details::makeDualViewFromOwningDeviceView(this->TransferData_->remoteLIDs_, remoteLIDs);

  if (this->verbose()) {
    std::ostringstream os;
    os << *prefix << "Done!" << endl;
    this->verboseOutputStream() << os.str();
  }
}

}  // namespace Tpetra

// Explicit instantiation macro.
// Only invoke this when in the Tpetra namespace.
// Most users do not need to use this.
//
// LO: The local ordinal type.
// GO: The global ordinal type.
// NODE: The Kokkos Node type.
#define TPETRA_EXPORT_INSTANT(LO, GO, NODE) \
  template class Export<LO, GO, NODE>;

#endif  // TPETRA_EXPORT_DEF_HPP
