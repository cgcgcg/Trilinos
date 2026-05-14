// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
//
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef TPETRA_IMPORTEXPORTDATA_DEF_HPP
#define TPETRA_IMPORTEXPORTDATA_DEF_HPP

#include <cstddef>
#include "Kokkos_Macros.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_Details_makeValidVerboseStream.hpp"
#include "Teuchos_FancyOStream.hpp"
#include "Teuchos_ParameterList.hpp"
#include "View/Kokkos_ViewCtor.hpp"
#include "decl/Kokkos_Declare_SERIAL.hpp"
#include "impl/Kokkos_Profiling.hpp"

namespace Tpetra {

template <class LocalOrdinal, class GlobalOrdinal, class Node>
ImportExportData<LocalOrdinal, GlobalOrdinal, Node>::
    ImportExportData(const Teuchos::RCP<const map_type>& source,
                     const Teuchos::RCP<const map_type>& target,
                     const Teuchos::RCP<Teuchos::FancyOStream>& out,
                     const Teuchos::RCP<Teuchos::ParameterList>& plist)
  : source_(source)
  ,  // NOT allowed to be null
  target_(target)
  ,  // allowed to be null
  out_(::Tpetra::Details::makeValidVerboseStream(out))
  , numSameIDs_(0)
  ,  // Import/Export constructor may change this
  distributor_(source->getComm(), out_, plist)
  ,                         // Im/Ex ctor will init
  isLocallyComplete_(true)  // Im/Ex ctor may change this
{
  TEUCHOS_ASSERT(!out_.is_null());
}

template <class LocalOrdinal, class GlobalOrdinal, class Node>
ImportExportData<LocalOrdinal, GlobalOrdinal, Node>::
    ImportExportData(const Teuchos::RCP<const map_type>& source,
                     const Teuchos::RCP<const map_type>& target)
  : ImportExportData(source, target, Teuchos::null, Teuchos::null) {}

template <class LocalOrdinal, class GlobalOrdinal, class Node>
ImportExportData<LocalOrdinal, GlobalOrdinal, Node>::
    ImportExportData(const Teuchos::RCP<const map_type>& source,
                     const Teuchos::RCP<const map_type>& target,
                     const Teuchos::RCP<Teuchos::FancyOStream>& out)
  : ImportExportData(source, target, out, Teuchos::null) {}

template <class LocalOrdinal, class GlobalOrdinal, class Node>
ImportExportData<LocalOrdinal, GlobalOrdinal, Node>::
    ImportExportData(const Teuchos::RCP<const map_type>& source,
                     const Teuchos::RCP<const map_type>& target,
                     const Teuchos::RCP<Teuchos::ParameterList>& plist)
  : ImportExportData(source, target, Teuchos::null, plist) {}

template <class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP<ImportExportData<LocalOrdinal, GlobalOrdinal, Node>>
ImportExportData<LocalOrdinal, GlobalOrdinal, Node>::
    reverseClone() {
  using Teuchos::ArrayView;
  using data_type = ImportExportData<LocalOrdinal, GlobalOrdinal, Node>;

  auto tData = Teuchos::rcp(new data_type(target_, source_, out_));

  // Things that stay the same
  tData->numSameIDs_ = numSameIDs_;

  // Things that reverse
  tData->distributor_     = *distributor_.getReverse();
  tData->permuteToLIDs_   = permuteFromLIDs_;
  tData->permuteFromLIDs_ = permuteToLIDs_;

  // Remotes / exports (easy part)
  tData->exportLIDs_ = remoteLIDs_;
  tData->remoteLIDs_ = exportLIDs_;
  tData->exportPIDs_ = Kokkos::DualView<int*, device_type>(Kokkos::ViewAllocateWithoutInitializing("exportPIDs"),
                                                           tData->exportLIDs_.extent(0));

  // Remotes / exports (hard part) - extract the exportPIDs from the remotes of my distributor
  const size_t NumReceives            = distributor_.getNumReceives();
  ArrayView<const int> ProcsFrom      = distributor_.getProcsFrom();
  ArrayView<const size_t> LengthsFrom = distributor_.getLengthsFrom();

  Kokkos::View<const int*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> ProcsFrom_h(ProcsFrom.data(), ProcsFrom.size());
  Kokkos::View<const size_t*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> LengthsFrom_h(LengthsFrom.data(), LengthsFrom.size());

  Kokkos::View<int*, memory_space> ProcsFrom_d(Kokkos::ViewAllocateWithoutInitializing("ProcsFrom_d"), ProcsFrom.size());
  Kokkos::View<size_t*, memory_space> LengthsFrom_d(Kokkos::ViewAllocateWithoutInitializing("LengthsFrom_d"), LengthsFrom.size());
  Kokkos::deep_copy(ProcsFrom_d, ProcsFrom_h);
  Kokkos::deep_copy(LengthsFrom_d, LengthsFrom_h);

  // isLocallyComplete is a local predicate.
  // It could be true in one direction but false in another.

  Kokkos::RangePolicy<execution_space> policy(0, NumReceives);
  auto exportPIDs_dev = tData->exportPIDs_.view_device();

  // TODO: merge the parallel regions

  Kokkos::parallel_reduce(
      policy, KOKKOS_LAMBDA(const size_t i, bool& isLocallyComplete) {
        const int pid = ProcsFrom_d(i);
        if (pid == -1) {
          isLocallyComplete = false;
        }
      },
      Kokkos::LAnd<bool>(tData->isLocallyComplete_));

  Kokkos::parallel_scan(
      policy, KOKKOS_LAMBDA(const size_t i, size_t& j, const bool is_final) {
        const int pid = ProcsFrom_d(i);
        if (!is_final)
          j += LengthsFrom_d(i);
        else {
          for (size_t k = 0; k < LengthsFrom_d(i); ++k) {
            exportPIDs_dev(j) = pid;
            ++j;
          }
        }
      });

  return tData;
}

}  // namespace Tpetra

// Explicit instantiation macro.
// Only invoke this when in the Tpetra namespace.
// Most users do not need to use this.
//
// LO: The local ordinal type.
// GO: The global ordinal type.
// NODE: The Kokkos Node type.
#define TPETRA_IMPORTEXPORTDATA_INSTANT(LO, GO, NODE) \
  template class ImportExportData<LO, GO, NODE>;

#endif  // TPETRA_IMPORTEXPORTDATA_DEF_HPP
