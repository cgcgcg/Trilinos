// @HEADER
// *****************************************************************************
//        MueLu: A package for multigrid based preconditioning
//
// Copyright 2012 NTESS and the MueLu contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef MUELU_INVERSEAPPROXIMATIONFACTORY_DEF_HPP_
#define MUELU_INVERSEAPPROXIMATIONFACTORY_DEF_HPP_

#include <Xpetra_BlockedCrsMatrix.hpp>
#include <Xpetra_CrsGraph.hpp>
#include <Xpetra_CrsMatrixWrap.hpp>
#include <Xpetra_CrsMatrix.hpp>
#include <Xpetra_VectorFactory.hpp>
#include <Xpetra_MatrixFactory.hpp>
#include <Xpetra_Matrix.hpp>

#include "Kokkos_Sort.hpp"
#include "KokkosBlas1_set.hpp"
#include "KokkosBatched_QR_Decl.hpp"
#include "KokkosBatched_ApplyQ_Decl.hpp"
#include "KokkosBatched_Trsv_Decl.hpp"
#include "KokkosBatched_Util.hpp"

#include "MueLu_Level.hpp"
#include "MueLu_Monitor.hpp"
#include "MueLu_Utilities.hpp"
#include "MueLu_InverseApproximationFactory_decl.hpp"

namespace MueLu {

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<const ParameterList> InverseApproximationFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::GetValidParameterList() const {
  RCP<ParameterList> validParamList = rcp(new ParameterList());
  using Magnitude                   = typename Teuchos::ScalarTraits<Scalar>::magnitudeType;

  validParamList->set<RCP<const FactoryBase>>("A", NoFactory::getRCP(), "Matrix to build the approximate inverse on.\n");

  validParamList->set<std::string>("inverse: approximation type", "diagonal", "Method used to approximate the inverse.");
  validParamList->set<Magnitude>("inverse: drop tolerance", 0.0, "Values below this threshold  are dropped from the matrix (or fixed if diagonal fixing is active).");
  validParamList->set<bool>("inverse: fixing", false, "Keep diagonal and fix small entries with 1.0");

  return validParamList;
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void InverseApproximationFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::DeclareInput(Level& currentLevel) const {
  Input(currentLevel, "A");
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void InverseApproximationFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Build(Level& currentLevel) const {
  FactoryMonitor m(*this, "Build", currentLevel);

  using STS       = Teuchos::ScalarTraits<SC>;
  const SC one    = STS::one();
  using Magnitude = typename Teuchos::ScalarTraits<Scalar>::magnitudeType;

  const ParameterList& pL = GetParameterList();
  const bool fixing       = pL.get<bool>("inverse: fixing");

  // check which approximation type to use
  const std::string method = pL.get<std::string>("inverse: approximation type");
  TEUCHOS_TEST_FOR_EXCEPTION(method != "diagonal" && method != "lumping" && method != "sparseapproxinverse", Exceptions::RuntimeError,
                             "MueLu::InverseApproximationFactory::Build: Approximation type can be 'diagonal' or 'lumping' or "
                             "'sparseapproxinverse'.");

  RCP<Matrix> A            = Get<RCP<Matrix>>(currentLevel, "A");
  RCP<BlockedCrsMatrix> bA = Teuchos::rcp_dynamic_cast<BlockedCrsMatrix>(A);
  const bool isBlocked     = (bA == Teuchos::null ? false : true);

  // if blocked operator is used, defaults to A(0,0)
  if (isBlocked) A = bA->getMatrix(0, 0);

  const Magnitude tol = pL.get<Magnitude>("inverse: drop tolerance");
  RCP<Matrix> Ainv    = Teuchos::null;

  if (method == "diagonal") {
    const auto diag = VectorFactory::Build(A->getRangeMap(), true);
    A->getLocalDiagCopy(*diag);
    const RCP<const Vector> D = (!fixing ? Utilities::GetInverse(diag) : Utilities::GetInverse(diag, tol, one));
    Ainv                      = MatrixFactory::Build(D);
  } else if (method == "lumping") {
    const auto diag           = Utilities::GetLumpedMatrixDiagonal(*A);
    const RCP<const Vector> D = (!fixing ? Utilities::GetInverse(diag) : Utilities::GetInverse(diag, tol, one));
    Ainv                      = MatrixFactory::Build(D);
  } else if (method == "sparseapproxinverse") {
    RCP<CrsGraph> sparsityPattern = Utilities::GetThresholdedGraph(A, tol, A->getGlobalMaxNumRowEntries());
    GetOStream(Statistics1) << "NNZ Graph(A): " << A->getCrsGraph()->getGlobalNumEntries() << " , NNZ Tresholded Graph(A): " << sparsityPattern->getGlobalNumEntries() << std::endl;
    RCP<Matrix> pAinv = GetSparseInverse(A, sparsityPattern);
    Ainv              = Utilities::GetThresholdedMatrix(pAinv, tol, fixing, pAinv->getGlobalMaxNumRowEntries());
    GetOStream(Statistics1) << "NNZ Ainv: " << pAinv->getGlobalNumEntries() << ", NNZ Tresholded Ainv (parameter: " << tol << "): " << Ainv->getGlobalNumEntries() << std::endl;
  }

  GetOStream(Statistics1) << "Approximate inverse calculated by: " << method << "." << std::endl;
  GetOStream(Statistics1) << "Ainv has " << Ainv->getGlobalNumRows() << "x" << Ainv->getGlobalNumCols() << " rows and columns." << std::endl;

  Set(currentLevel, "Ainv", Ainv);
}

template <class view_type, class comparator_type>
KOKKOS_INLINE_FUNCTION void serialHeapSort(view_type& v, comparator_type comparator) {
  auto N       = v.extent(0);
  size_t start = N / 2;
  size_t end   = N;
  while (end > 1) {
    if (start > 0)
      start = start - 1;
    else {
      end       = end - 1;
      auto temp = v(0);
      v(0)      = v(end);
      v(end)    = temp;
    }
    size_t root = start;
    while (2 * root + 1 < end) {
      size_t child = 2 * root + 1;
      if ((child + 1 < end) and (comparator(v(child), v(child + 1))))
        ++child;

      if (comparator(v(root), v(child))) {
        auto temp = v(root);
        v(root)   = v(child);
        v(child)  = temp;
        root      = child;
      } else
        break;
    }
  }
}

template <class view_type>
struct Comparator {
  view_type view;

  KOKKOS_INLINE_FUNCTION
  bool operator()(size_t x, size_t y) const {
    return x < y;
  }
};

template <class local_matrix_type>
class LocalSPAIFunctor {
 private:
  using scalar_type        = typename local_matrix_type::value_type;
  using local_ordinal_type = typename local_matrix_type::ordinal_type;
  using execution_space    = typename local_matrix_type::execution_space;
  using impl_scalar_type   = typename KokkosKernels::ArithTraits<scalar_type>::val_type;
  using impl_ATS           = KokkosKernels::ArithTraits<impl_scalar_type>;

 public:
  using shared_matrix    = Kokkos::View<impl_scalar_type**, typename execution_space::scratch_memory_space, Kokkos::MemoryUnmanaged>;
  using shared_vector    = Kokkos::View<impl_scalar_type*, typename execution_space::scratch_memory_space, Kokkos::MemoryUnmanaged>;
  using shared_lo_vector = Kokkos::View<local_ordinal_type*, typename execution_space::scratch_memory_space, Kokkos::MemoryUnmanaged>;

 private:
  local_matrix_type lclA;
  local_matrix_type lclAinv;
  local_ordinal_type maxUniqueColEntries;

 public:
  LocalSPAIFunctor(local_matrix_type lclA_, local_matrix_type lclAinv_, local_ordinal_type maxUniqueColEntries_)
    : lclA(lclA_)
    , lclAinv(lclAinv_)
    , maxUniqueColEntries(maxUniqueColEntries_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::TeamPolicy<execution_space>::member_type& thread) const {
    auto k       = thread.league_rank();
    auto rowAinv = lclAinv.row(k);

    // Loop over entries in row k of Ainv and collect all of A's column indices.
    shared_lo_vector column_indices(thread.team_shmem(), maxUniqueColEntries);
    local_ordinal_type numColEntries = 0;
    for (local_ordinal_type ii = 0; ii < rowAinv.length; ++ii) {
      auto i    = rowAinv.colidx(ii);
      auto rowA = lclA.rowConst(i);
      for (local_ordinal_type jj = 0; jj < rowA.length; ++jj) {
        auto j                        = rowA.colidx(jj);
        column_indices(numColEntries) = j;
        ++numColEntries;
      }
    }

    // Get merged list of column indices.
    // Kokkos::sort(Kokkos::subview(column_indices, Kokkos::make_pair(0, numColEntries)));

    auto temp = Kokkos::subview(column_indices, Kokkos::make_pair(0, numColEntries));
    Comparator<decltype(temp)> comp{temp};
    serialHeapSort(temp, comp);
    // [&column_indices](local_ordinal_type i, local_ordinal_type j) { return column_indices(i)<column_indices(j); }

    local_ordinal_type numUniqeColEntries = 0;
    if (numColEntries > 0)
      ++numUniqeColEntries;
    local_ordinal_type pos     = 0;
    local_ordinal_type diagPos = 0;
    for (local_ordinal_type m = 1; m < numColEntries; ++m) {
      if (column_indices(pos) != column_indices(m)) {
        column_indices(pos + 1) = column_indices(m);
        ++pos;
        ++numUniqeColEntries;
        if (column_indices(pos) == k)
          diagPos = pos;
      }
    }

    // Extract local part of A.
    shared_matrix localA(thread.team_shmem(), numUniqeColEntries, rowAinv.length);
    KokkosBlas::SerialSet::invoke(impl_ATS::zero(), localA);

    // Now fill localA.
    for (local_ordinal_type ii = 0; ii < rowAinv.length; ++ii) {
      auto i    = rowAinv.colidx(ii);
      auto rowA = lclA.rowConst(i);
      for (local_ordinal_type jj = 0; jj < rowA.length; ++jj) {
        auto j = rowA.colidx(jj);
        auto v = rowA.value(jj);
        // Determine local index.
        // Sequential search might not be a great idea.
        for (local_ordinal_type m = 0; m < numUniqeColEntries; ++m) {
          if (column_indices(m) == j) {
            localA(m, ii) = v;
            break;
          }
        }
      }
    }

    shared_matrix ek(thread.team_shmem(), numUniqeColEntries, 1);
    // set to zero, set diagonal entry to one
    for (local_ordinal_type i = 0; i < numUniqeColEntries; ++i) {
      ek(i, 0) = (i == diagPos) ? impl_ATS::one() : impl_ATS::zero();
    }

    // QR solve
    shared_vector tau(thread.team_shmem(), numUniqeColEntries);
    shared_vector work(thread.team_shmem(), numUniqeColEntries);
    // factorize localA = Q*R in-place
    KokkosBatched::SerialQR<KokkosBatched::Algo::QR::Unblocked>::invoke(localA, tau, work);
    // ek := Q^T ek
    KokkosBatched::SerialApplyQ<KokkosBatched::Side::Left, KokkosBatched::Trans::Transpose, KokkosBatched::Algo::ApplyQ::Unblocked>::invoke(localA, tau, ek, work);
    // ek[:rowLength] := R^{-1} ek[:rowLength]
    auto sub_A  = Kokkos::subview(localA, Kokkos::make_pair(0, rowAinv.length), Kokkos::ALL());
    auto sub_ek = Kokkos::subview(ek, Kokkos::make_pair(0, rowAinv.length), 0);
    KokkosBatched::SerialTrsv<KokkosBatched::Uplo::Upper, KokkosBatched::Trans::NoTranspose, KokkosBatched::Diag::NonUnit, KokkosBatched::Algo::Trsv::Unblocked>::invoke(impl_ATS::one(), sub_A, sub_ek);

    // Set entries of Ainv.
    for (local_ordinal_type i = 0; i < rowAinv.length; ++i) {
      rowAinv.value(i) = sub_ek(i);
    }
  }
};

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>>
InverseApproximationFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::GetSparseInverse(const RCP<Matrix>& Aorg, const RCP<const CrsGraph>& sparsityPattern) const {
  using execution_space = typename Node::execution_space;

  // construct the inverse matrix with the given sparsity pattern
  RCP<Matrix> Ainv = MatrixFactory::Build(sparsityPattern);
  Ainv->resumeFill();

  // gather missing rows from other procs to generate an overlapping map
  RCP<Import> rowImport = ImportFactory::Build(sparsityPattern->getRowMap(), sparsityPattern->getColMap());
  RCP<Matrix> A         = MatrixFactory::Build(Aorg, *rowImport);

  auto maxRowEntriesAinv   = Ainv->getLocalMaxNumRowEntries();
  auto maxRowEntriesA      = A->getLocalMaxNumRowEntries();
  auto maxUniqueColEntries = maxRowEntriesAinv * maxRowEntriesA;
  {
    auto lclA    = A->getLocalMatrixDevice();
    auto lclAinv = Ainv->getLocalMatrixDevice();

    LocalSPAIFunctor spaiFunctor(lclA, lclAinv, maxUniqueColEntries);
    Kokkos::TeamPolicy<execution_space> policy(lclAinv.numRows(), 1);

    using shared_matrix    = typename decltype(spaiFunctor)::shared_matrix;
    using shared_vector    = typename decltype(spaiFunctor)::shared_vector;
    using shared_lo_vector = typename decltype(spaiFunctor)::shared_lo_vector;

    int size = shared_matrix::shmem_size(maxUniqueColEntries, maxRowEntriesAinv) + shared_matrix::shmem_size(maxUniqueColEntries, 1) + 2 * shared_vector::shmem_size(maxUniqueColEntries) + shared_lo_vector::shmem_size(maxUniqueColEntries);

    if (size < policy.scratch_size_max(/*level=*/(int)0))
      policy.set_scratch_size(/*level=*/(int)0, Kokkos::PerTeam(size));
    else if (size < policy.scratch_size_max(/*level=*/(int)1))
      policy.set_scratch_size(/*level=*/(int)1, Kokkos::PerTeam(size));
    else
      throw Exceptions::RuntimeError("Neither L0 scratch memory (max size " + std::to_string(policy.scratch_size_max((int)0)) +
                                     "), nor L1 scratch memory (max size " + std::to_string(policy.scratch_size_max((int)1)) +
                                     ") is large enough for requested allocation of size " + std::to_string(size));

    Kokkos::parallel_for("MueLu::InverseFactory::LocalSpai", policy, spaiFunctor);
  }

  Ainv->fillComplete();

  return Ainv;
}

}  // namespace MueLu

#endif /* MUELU_INVERSEAPPROXIMATIONFACTORY_DEF_HPP_ */
