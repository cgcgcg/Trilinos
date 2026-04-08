// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
//
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef TPETRA_FEMULTIVECTOR_DEF_HPP
#define TPETRA_FEMULTIVECTOR_DEF_HPP

/// \file Tpetra_MultiVector_def.hpp
/// \brief Definition of the Tpetra::MultiVector class

#include <cstddef>
#include <stdexcept>
#include "Tpetra_Access.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_MultiVector.hpp"
#include "Tpetra_Import.hpp"
#include "Tpetra_Details_Behavior.hpp"
#include "Tpetra_FEMultiVector_decl.hpp"

namespace Tpetra {

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
FEMultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    FEMultiVector(const Teuchos::RCP<const typename FEMultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::map_type>& map,
                  const Teuchos::RCP<const Import<LocalOrdinal, GlobalOrdinal, Node>>& importer,
                  const size_t numVecs,
                  const bool zeroOut)
  : base_type(importer.is_null() ? map : importer->getTargetMap(),
              numVecs, zeroOut)
  , activeMultiVector_(Teuchos::rcp(new FE::WhichActive(FE::ACTIVE_OWNED_PLUS_SHARED)))
  , importer_(importer)
  , overlappingConstruction_(true)
  , upperBoundNonlocalEntries_(0) {
  const char tfecfFuncName[] = "FEMultiVector constructor: ";

  if (!importer_.is_null()) {
    const bool debug = ::Tpetra::Details::Behavior::debug();
    if (debug) {
      // Checking Map sameness may require an all-reduce, so we should
      // reserve it for debug mode.
      TEUCHOS_TEST_FOR_EXCEPTION_CLASS_FUNC(!importer_->getSourceMap()->isSameAs(*map),
                                            std::runtime_error,
                                            "If you provide a nonnull Import, then the input Map "
                                            "must be the same as the input Import's source Map.");

      // Checking whether one Map is locally fitted to another could be
      // expensive.
      const bool locallyFitted =
          importer->getTargetMap()->isLocallyFitted(*(importer->getSourceMap()));
      TEUCHOS_TEST_FOR_EXCEPTION_CLASS_FUNC(!locallyFitted, std::runtime_error,
                                            "If you provide a nonnull Import, then its target Map must be "
                                            "locally fitted (see Map::isLocallyFitted documentation) to its "
                                            "source Map.");
    }

    // Memory aliasing is required for FEMultiVector
    inactiveMultiVector_ =
        Teuchos::rcp(new base_type(*this, importer_->getSourceMap(), 0));
  }
  fillState_ = Teuchos::rcp(new FE::FillState(FE::FillState::closed));
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
FEMultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    FEMultiVector(const Teuchos::RCP<const typename FEMultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::map_type>& map,
                  const size_t numVecs,
                  const size_t upperBoundNonlocalEntries,
                  const bool zeroOut)
  : base_type(map, numVecs, zeroOut)
  , activeMultiVector_(Teuchos::rcp(new FE::WhichActive(FE::ACTIVE_OWNED)))
  , importer_(Teuchos::null)
  , overlappingConstruction_(false)
  , upperBoundNonlocalEntries_(upperBoundNonlocalEntries) {
  const char tfecfFuncName[] = "FEMultiVector constructor: ";

  TEUCHOS_TEST_FOR_EXCEPTION_CLASS_FUNC(!map->isOneToOne(),
                                        std::runtime_error,
                                        "The input map must be one-to-one");
  fillState_ = Teuchos::rcp(new FE::FillState(FE::FillState::closed));
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void FEMultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    beginFill() {
  if (overlappingConstruction_) {
    // The FEMultiVector is in owned+shared mode on construction, so we
    // do not throw in that case.
    if (*activeMultiVector_ == FE::ACTIVE_OWNED) {
      switchActiveMultiVector();
    }
  } else {
    // allocate container to hold nonlocal entries
    nonlocal_entries_ = Teuchos::rcp(new nonlocal_entries_map_type(upperBoundNonlocalEntries_));
  }
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void FEMultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    endFill() {
  const char tfecfFuncName[] = "endFill: ";

  if (overlappingConstruction_) {
    if (*activeMultiVector_ == FE::ACTIVE_OWNED_PLUS_SHARED) {
      doOwnedPlusSharedToOwned(Tpetra::ADD);
      switchActiveMultiVector();
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION_CLASS_FUNC(true, std::runtime_error,
                                            "Owned+Shared MultiVector already active; "
                                            "cannot call endFill.");
    }
  } else {
    // construct overlapping map, importer and export

    auto map = this->getMap();

    {
      auto nonlocal_entries = *nonlocal_entries_;

      global_size_t numLocalEntries    = map->getLocalNumElements();
      global_size_t numNonlocalEntries = nonlocal_entries.size();
      Kokkos::View<GlobalOrdinal*, Kokkos::HostSpace> overlappingMapGIDs(Kokkos::ViewAllocateWithoutInitializing("overlappingMapGIDs"), map->getLocalNumElements() + numNonlocalEntries);
      auto localRange = Kokkos::make_pair((global_size_t)0, numLocalEntries);
      Kokkos::deep_copy(Kokkos::subview(overlappingMapGIDs, localRange), map->getMyGlobalIndices());

      Kokkos::RangePolicy<Kokkos::HostSpace> policy(0, nonlocal_entries.capacity());
      Kokkos::parallel_scan(
          policy, KOKKOS_LAMBDA(const uint32_t i, size_t& offset, const bool is_final) {
            if (nonlocal_entries.valid_at(i)) {
              auto key = nonlocal_entries.key_at(i);
              if (is_final) {
                overlappingMapGIDs(numLocalEntries + offset) = key;
              }
              ++offset;
            }
          });

      auto overlappingMap = Teuchos::rcp(new Map<LocalOrdinal, GlobalOrdinal, Node>(Teuchos::OrdinalTraits<GlobalOrdinal>::invalid(),
                                                                                    overlappingMapGIDs,
                                                                                    map->getIndexBase(),
                                                                                    map->getComm()));
      Teuchos::RCP<Teuchos::ParameterList> params;
      Import<LocalOrdinal, GlobalOrdinal, Node> import(map, overlappingMap, params);
      base_type overlappingMV(overlappingMap, this->getNumVectors(), false);
      {
        auto lclOverlappingMap = overlappingMap->getLocalMap();
        auto lclOverlappingMV  = overlappingMV.getLocalViewHost(Access::OverwriteAll);
        auto lclMV             = this->getLocalViewHost(Access::ReadOnly);

        // Local part
        Kokkos::deep_copy(Kokkos::subview(lclOverlappingMV, localRange, Kokkos::ALL()), lclMV);

        Kokkos::parallel_for(
            policy, KOKKOS_LAMBDA(const uint32_t i) {
              if (nonlocal_entries.valid_at(i)) {
                auto key                         = nonlocal_entries.key_at(i);
                auto val                         = nonlocal_entries.value_at(i);
                auto lid                         = lclOverlappingMap.getLocalElement(key);
                lclOverlappingMV(lid, val.first) = val.second;
              }
            });
      }

      // do export
      this->doExport(overlappingMV, import, Tpetra::ADD);
    }
    // deallocate container for nonlocal entries
    nonlocal_entries_ = Teuchos::null;
  }
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void FEMultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::beginAssembly() {
  const char tfecfFuncName[] = "FEMultiVector::beginAssembly: ";
  TEUCHOS_TEST_FOR_EXCEPTION_CLASS_FUNC(
      *fillState_ != FE::FillState::closed,
      std::runtime_error,
      "Cannot beginAssembly, matrix is not in a closed state");
  *fillState_ = FE::FillState::open;
  this->beginFill();
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void FEMultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::endAssembly() {
  const char tfecfFuncName[] = "FEMultiVector::endAssembly: ";
  TEUCHOS_TEST_FOR_EXCEPTION_CLASS_FUNC(
      *fillState_ != FE::FillState::open,
      std::runtime_error,
      "Cannot endAssembly, matrix is not open to fill.");
  *fillState_ = FE::FillState::closed;
  this->endFill();
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void FEMultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::beginModify() {
  const char tfecfFuncName[] = "FEMultiVector::beginModify: ";
  TEUCHOS_TEST_FOR_EXCEPTION_CLASS_FUNC(
      *fillState_ != FE::FillState::closed,
      std::runtime_error,
      "Cannot beginModify, matrix is not in a closed state");
  *fillState_ = FE::FillState::modify;
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void FEMultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::endModify() {
  const char tfecfFuncName[] = "FEMultiVector::endModify: ";
  TEUCHOS_TEST_FOR_EXCEPTION_CLASS_FUNC(
      *fillState_ != FE::FillState::modify,
      std::runtime_error,
      "Cannot endModify, matrix is not open to modify.");
  *fillState_ = FE::FillState::closed;
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void FEMultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    globalAssemble() {
  endFill();
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void FEMultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    replaceMap(const Teuchos::RCP<const map_type>& /* newMap */) {
  const char tfecfFuncName[] = "replaceMap: ";

  TEUCHOS_TEST_FOR_EXCEPTION_CLASS_FUNC(true, std::runtime_error, "This method is not implemented.");
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void FEMultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    doOwnedPlusSharedToOwned(const CombineMode CM) {
  if (!importer_.is_null() &&
      *activeMultiVector_ == FE::ACTIVE_OWNED_PLUS_SHARED) {
    inactiveMultiVector_->doExport(*this, *importer_, CM);
  }
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void FEMultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    doOwnedToOwnedPlusShared(const CombineMode CM) {
  if (!importer_.is_null() &&
      *activeMultiVector_ == FE::ACTIVE_OWNED) {
    inactiveMultiVector_->doImport(*this, *importer_, CM);
  }
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void FEMultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    switchActiveMultiVector() {
  if (*activeMultiVector_ == FE::ACTIVE_OWNED_PLUS_SHARED) {
    *activeMultiVector_ = FE::ACTIVE_OWNED;
  } else {
    *activeMultiVector_ = FE::ACTIVE_OWNED_PLUS_SHARED;
  }

  if (importer_.is_null()) {
    return;
  }

  // Use MultiVector's swap routine here
  this->swap(*inactiveMultiVector_);
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void FEMultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    sumIntoGlobalValue(const GlobalOrdinal gblRow,
                       const size_t col,
                       const impl_scalar_type& value,
                       const bool atomic) {
  if (overlappingConstruction_) {
    base_type::sumIntoGlobalValue(gblRow, col, value, atomic);
  } else {
    auto lclMap = this->getMap()->getLocalMap();
    auto lid    = lclMap.getLocalElement(gblRow);
    if (lid == Teuchos::OrdinalTraits<LocalOrdinal>::invalid()) {
      // off-rank insert

    } else {
      // rank-local insert
      this->sumIntoLocalValue(lid, col, value, atomic);
    }
  }
}

}  // namespace Tpetra

//
// Explicit instantiation macro
//
// Must be expanded from within the Tpetra namespace!
//

#define TPETRA_FEMULTIVECTOR_INSTANT(SCALAR, LO, GO, NODE) \
  template class FEMultiVector<SCALAR, LO, GO, NODE>;

#endif  // TPETRA_FEMULTIVECTOR_DEF_HPP
