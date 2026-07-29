// @HEADER
// *****************************************************************************
//       Ifpack2: Templated Object-Oriented Algebraic Preconditioner Package
//
// Copyright 2009 NTESS and the Ifpack2 contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef IFPACK2_SPARSEAPPROXIMATEINVERSE_DEF_HPP
#define IFPACK2_SPARSEAPPROXIMATEINVERSE_DEF_HPP

#include <type_traits>
#include "Kokkos_Macros.hpp"
#include "Teuchos_TypeNameTraits.hpp"
#include "Teuchos_StandardParameterEntryValidators.hpp"
#include "Teuchos_Time.hpp"
#include "Tpetra_CombineMode.hpp"
#include "Tpetra_CrsGraph_decl.hpp"
#include "Tpetra_CrsMatrix.hpp"

#include "Ifpack2_Parameters.hpp"
#include "Ifpack2_Details_getParamTryingTypes.hpp"
#include "Tpetra_Filter.hpp"

#include "Kokkos_Sort.hpp"
#include "KokkosBlas1_set.hpp"
#include "KokkosBatched_QR_Decl.hpp"
#include "KokkosBatched_ApplyQ_Decl.hpp"
#include "KokkosBatched_Trsv_Decl.hpp"
#include "KokkosBatched_Util.hpp"

namespace Ifpack2 {

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
  const local_matrix_type lclA;
  local_matrix_type lclAinv;
  const local_ordinal_type maxUniqueColEntries;
  const int scratchLevel;

 public:
  LocalSPAIFunctor(const local_matrix_type& lclA_, local_matrix_type& lclAinv_, local_ordinal_type maxUniqueColEntries_, int scratchLevel_)
    : lclA(lclA_)
    , lclAinv(lclAinv_)
    , maxUniqueColEntries(maxUniqueColEntries_)
    , scratchLevel(scratchLevel_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::TeamPolicy<execution_space>::member_type& thread) const {
    auto rlid    = thread.league_rank();
    auto rowAinv = lclAinv.row(rlid);

    // Loop over entries in row rlid of Ainv and collect all of A's column indices.
    shared_lo_vector column_indices(thread.team_scratch(scratchLevel), maxUniqueColEntries);
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
    local_ordinal_type numUniqeColEntries = 0;
    local_ordinal_type diagOffset         = 0;
    {
      // Sort
      Kokkos::Experimental::sort_team(thread, Kokkos::subview(column_indices, Kokkos::make_pair(0, numColEntries)));
      // Merge
      if (numColEntries > 0)
        ++numUniqeColEntries;
      local_ordinal_type pos = 0;
      for (local_ordinal_type m = 1; m < numColEntries; ++m) {
        if (column_indices(pos) != column_indices(m)) {
          column_indices(pos + 1) = column_indices(m);
          ++pos;
          ++numUniqeColEntries;
          if (column_indices(pos) == rlid)
            diagOffset = pos;
        }
      }
    }

    // Extract local part of A into a dense view.
    shared_matrix localA(thread.team_scratch(scratchLevel), numUniqeColEntries, rowAinv.length);
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

    shared_matrix ek(thread.team_scratch(scratchLevel), numUniqeColEntries, 1);
    // set to zero, set diagonal entry to one
    for (local_ordinal_type i = 0; i < numUniqeColEntries; ++i) {
      ek(i, 0) = (i == diagOffset) ? impl_ATS::one() : impl_ATS::zero();
    }

    // QR solve
    shared_vector tau(thread.team_scratch(scratchLevel), rowAinv.length);
    shared_vector work(thread.team_scratch(scratchLevel), numUniqeColEntries);
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
Teuchos::RCP<Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>>
GetSparseApproximateInverse(const Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>& A,
                            const Teuchos::RCP<const Tpetra::CrsGraph<LocalOrdinal, GlobalOrdinal, Node>>& sparsityPattern) {
  using execution_space = typename Node::execution_space;
  using import_type     = Tpetra::Import<LocalOrdinal, GlobalOrdinal, Node>;
  using crs_matrix_type = Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>;

  // construct the inverse matrix with the given sparsity pattern
  auto Ainv = Teuchos::rcp(new crs_matrix_type(sparsityPattern));
  Ainv->resumeFill();

  // gather missing rows from other procs to generate an overlapping map
  Teuchos::RCP<const import_type> rowImport;
  if (A.getRowMap()->isSameAs(*sparsityPattern->getDomainMap())) {
    rowImport = sparsityPattern->getImporter();
  } else {
    rowImport = rcp(new import_type(A.getRowMap(), sparsityPattern->getColMap()));
  }
  Teuchos::RCP<const crs_matrix_type> A_columnMap;
  if (!rowImport.is_null()) {
    A_columnMap = Tpetra::importAndFillCompleteCrsMatrix<crs_matrix_type>(Teuchos::rcpFromRef(A), *rowImport);
  } else {
    A_columnMap = Teuchos::rcpFromRef(A);
  }

  auto maxRowEntriesAinv   = Ainv->getLocalMaxNumRowEntries();
  auto maxRowEntriesA      = A_columnMap->getLocalMaxNumRowEntries();
  auto maxUniqueColEntries = maxRowEntriesAinv * maxRowEntriesA;
  {
    auto lclA    = A_columnMap->getLocalMatrixDevice();
    auto lclAinv = Ainv->getLocalMatrixDevice();

    Kokkos::TeamPolicy<execution_space> policy(lclAinv.numRows(), 1);

    using spai_functor_type = LocalSPAIFunctor<decltype(lclAinv)>;
    using shared_matrix     = typename spai_functor_type::shared_matrix;
    using shared_vector     = typename spai_functor_type::shared_vector;
    using shared_lo_vector  = typename spai_functor_type::shared_lo_vector;

    int size = shared_matrix::shmem_size(maxUniqueColEntries, maxRowEntriesAinv) + shared_matrix::shmem_size(maxUniqueColEntries, 1) + shared_vector::shmem_size(3 * maxUniqueColEntries) + shared_vector::shmem_size(maxRowEntriesAinv) + shared_lo_vector::shmem_size(maxUniqueColEntries);

    int scratchLevel = -1;
    if (size < policy.scratch_size_max(/*level=*/(int)0)) {
      policy.set_scratch_size(/*level=*/(int)0, Kokkos::PerTeam(size));
      scratchLevel = 0;
    } else if (size < policy.scratch_size_max(/*level=*/(int)1)) {
      policy.set_scratch_size(/*level=*/(int)1, Kokkos::PerTeam(size));
      scratchLevel = 1;
    } else
      throw std::runtime_error("Neither L0 scratch memory (max size " + std::to_string(policy.scratch_size_max((int)0)) +
                               "), nor L1 scratch memory (max size " + std::to_string(policy.scratch_size_max((int)1)) +
                               ") is large enough for requested allocation of size " + std::to_string(size));

    LocalSPAIFunctor spaiFunctor(lclA, lclAinv, maxUniqueColEntries, scratchLevel);

    Kokkos::parallel_for("Ifpack2::SparseApproximateInverse::LocalSpai", policy, spaiFunctor);
  }

  Ainv->fillComplete();

  return Ainv;
}

template <class MatrixType>
SparseApproximateInverse<MatrixType>::SparseApproximateInverse(const Teuchos::RCP<const row_matrix_type>& A)
  : A_(A)
  , InitializeTime_(0.0)
  , ComputeTime_(0.0)
  , ApplyTime_(0.0)
  , NumInitialize_(0)
  , NumCompute_(0)
  , NumApply_(0)
  , IsInitialized_(false)
  , IsComputed_(false) {}

template <class MatrixType>
void SparseApproximateInverse<MatrixType>::setParameters(const Teuchos::ParameterList& params) {
  using Ifpack2::Details::getParamTryingTypes;
  const char prefix[] = "Ifpack2::SparseApproximateInverse: ";

  // Don't actually change the instance variables until we've checked
  // all parameters.  This ensures that setParameters satisfies the
  // strong exception guarantee (i.e., is transactional).

  magnitude_type dropTol = DropTolerance_;
  {
    const std::string paramName("fact: drop tolerance");
    getParamTryingTypes<magnitude_type, magnitude_type, double>(dropTol, params, paramName, prefix);
  }

  DropTolerance_ = dropTol;
}

template <class MatrixType>
Teuchos::RCP<const Teuchos::Comm<int>>
SparseApproximateInverse<MatrixType>::getComm() const {
  TEUCHOS_TEST_FOR_EXCEPTION(
      A_.is_null(), std::runtime_error,
      "Ifpack2::SparseApproximateInverse::getComm: "
      "The matrix is null.  Please call setMatrix() with a nonnull input "
      "before calling this method.");
  return A_->getComm();
}

template <class MatrixType>
Teuchos::RCP<const typename SparseApproximateInverse<MatrixType>::row_matrix_type>
SparseApproximateInverse<MatrixType>::getMatrix() const {
  return A_;
}

template <class MatrixType>
Teuchos::RCP<const typename SparseApproximateInverse<MatrixType>::map_type>
SparseApproximateInverse<MatrixType>::getDomainMap() const {
  TEUCHOS_TEST_FOR_EXCEPTION(
      A_.is_null(), std::runtime_error,
      "Ifpack2::SparseApproximateInverse::getDomainMap: "
      "The matrix is null.  Please call setMatrix() with a nonnull input "
      "before calling this method.");
  return A_->getDomainMap();
}

template <class MatrixType>
Teuchos::RCP<const typename SparseApproximateInverse<MatrixType>::map_type>
SparseApproximateInverse<MatrixType>::getRangeMap() const {
  TEUCHOS_TEST_FOR_EXCEPTION(
      A_.is_null(), std::runtime_error,
      "Ifpack2::SparseApproximateInverse::getRangeMap: "
      "The matrix is null.  Please call setMatrix() with a nonnull input "
      "before calling this method.");
  return A_->getRangeMap();
}

template <class MatrixType>
bool SparseApproximateInverse<MatrixType>::hasTransposeApply() const {
  return true;
}

template <class MatrixType>
int SparseApproximateInverse<MatrixType>::getNumInitialize() const {
  return NumInitialize_;
}

template <class MatrixType>
int SparseApproximateInverse<MatrixType>::getNumCompute() const {
  return NumCompute_;
}

template <class MatrixType>
int SparseApproximateInverse<MatrixType>::getNumApply() const {
  return NumApply_;
}

template <class MatrixType>
double SparseApproximateInverse<MatrixType>::getInitializeTime() const {
  return InitializeTime_;
}

template <class MatrixType>
double SparseApproximateInverse<MatrixType>::getComputeTime() const {
  return ComputeTime_;
}

template <class MatrixType>
double SparseApproximateInverse<MatrixType>::getApplyTime() const {
  return ApplyTime_;
}

template <class MatrixType>
size_t SparseApproximateInverse<MatrixType>::getNodeSmootherComplexity() const {
  TEUCHOS_TEST_FOR_EXCEPTION(
      A_.is_null(), std::runtime_error,
      "Ifpack2::SparseApproximateInverse::getNodeSmootherComplexity: "
      "The input matrix A is null.  Please call setMatrix() with a nonnull "
      "input matrix, then call compute(), before calling this method.");
  return invA_->getLocalNumEntries();
}

template <class MatrixType>
global_size_t SparseApproximateInverse<MatrixType>::getGlobalNumEntries() const {
  return invA_->getGlobalNumEntries();
}

template <class MatrixType>
size_t SparseApproximateInverse<MatrixType>::getLocalNumEntries() const {
  return invA_->getLocalNumEntries();
}

template <class MatrixType>
void SparseApproximateInverse<MatrixType>::setMatrix(const Teuchos::RCP<const row_matrix_type>& A) {
  if (A.getRawPtr() != A_.getRawPtr()) {
    // Check in serial or one-process mode if the matrix is square.
    TEUCHOS_TEST_FOR_EXCEPTION(
        !A.is_null() && A->getComm()->getSize() == 1 &&
            A->getLocalNumRows() != A->getLocalNumCols(),
        std::runtime_error,
        "Ifpack2::SparseApproximateInverse::setMatrix: If A's communicator only "
        "contains one process, then A must be square.  Instead, you provided a "
        "matrix A with "
            << A->getLocalNumRows() << " rows and "
            << A->getLocalNumCols() << " columns.");

    // It's legal for A to be null; in that case, you may not call
    // initialize() until calling setMatrix() with a nonnull input.
    // Regardless, setting the matrix invalidates any previous
    // factorization.
    IsInitialized_ = false;
    IsComputed_    = false;

    invA_ = Teuchos::null;
    A_    = A;
  }
}

template <class MatrixType>
void SparseApproximateInverse<MatrixType>::initialize() {
  Teuchos::Time timer("SparseApproximateInverse::initialize");
  double startTime = timer.wallTime();
  {
    Teuchos::TimeMonitor timeMon(timer);

    // Check that the matrix is nonnull.
    TEUCHOS_TEST_FOR_EXCEPTION(
        A_.is_null(), std::runtime_error,
        "Ifpack2::SparseApproximateInverse::initialize: "
        "The matrix to precondition is null.  Please call setMatrix() with a "
        "nonnull input before calling this method.");

    // Clear any previous computations.
    IsInitialized_ = false;
    IsComputed_    = false;
    invA_          = Teuchos::null;

    IsInitialized_ = true;
    ++NumInitialize_;
  }  // timer scope
  InitializeTime_ += (timer.wallTime() - startTime);
}

template <class MatrixType>
void SparseApproximateInverse<MatrixType>::compute() {
  using impl_scalar_type    = typename MatrixType::impl_scalar_type;
  using local_ordinal_type  = typename MatrixType::local_ordinal_type;
  using global_ordinal_type = typename MatrixType::global_ordinal_type;
  using node_type           = typename MatrixType::node_type;
  using implATS             = KokkosKernels::ArithTraits<impl_scalar_type>;

  // Don't count initialization in the compute() time.
  if (!isInitialized()) {
    initialize();
  }

  Teuchos::Time timer("SparseApproximateInverse::compute");
  double startTime = timer.wallTime();
  {  // Timer scope for timing compute()
    Teuchos::TimeMonitor timeMon(timer, true);

    // Get A as CrsMatrix.
    auto Acrs = Teuchos::rcp_dynamic_cast<const crs_matrix_type>(A_, true);

    auto tol = DropTolerance_;

    // Construct the graph for the approximate inverse by filtering the matrix Acrs based on entry size.
    Teuchos::RCP<const Tpetra::CrsGraph<local_ordinal_type, global_ordinal_type, node_type>> sparsityPattern = Tpetra::applyGraphFilter_GID<Tpetra::CrsGraph<local_ordinal_type, global_ordinal_type, node_type>>(
        *Acrs, KOKKOS_LAMBDA(const global_ordinal_type rgid, const global_ordinal_type cgid, const impl_scalar_type val) {
          return (rgid == cgid) || (implATS::magnitude(val) >= tol);
        });

    // Compute the sparse approximate inverse.
    auto Ainv = GetSparseApproximateInverse(*Acrs, sparsityPattern);

    // Filter out small entries from the sparse approximate inverse.
    Ainv = Tpetra::applyFilter_GID(
        *Ainv, KOKKOS_LAMBDA(const global_ordinal_type rgid, const global_ordinal_type cgid, const impl_scalar_type val) {
          return (rgid == cgid) || (implATS::magnitude(val) >= tol);
        });

    invA_ = Ainv;

  }  // Timer scope for timing compute()
  ComputeTime_ += (timer.wallTime() - startTime);
  IsComputed_ = true;
  ++NumCompute_;
}  // compute()

template <class MatrixType>
void SparseApproximateInverse<MatrixType>::
    apply(const Tpetra::MultiVector<scalar_type, local_ordinal_type, global_ordinal_type, node_type>& X,
          Tpetra::MultiVector<scalar_type, local_ordinal_type, global_ordinal_type, node_type>& Y,
          Teuchos::ETransp mode,
          scalar_type alpha,
          scalar_type beta) const {
  TEUCHOS_TEST_FOR_EXCEPTION(
      !isComputed(), std::runtime_error,
      "Ifpack2::SparseApproximateInverse::apply: You must call compute() to compute the incomplete "
      "factorization, before calling apply().");

  TEUCHOS_TEST_FOR_EXCEPTION(
      X.getNumVectors() != Y.getNumVectors(), std::runtime_error,
      "Ifpack2::SparseApproximateInverse::apply: X and Y must have the same number of columns.  "
      "X has "
          << X.getNumVectors() << " columns, but Y has "
          << Y.getNumVectors() << " columns.");

  Teuchos::Time timer("SparseApproximateInverse::apply");
  double startTime = timer.wallTime();
  {  // Start timing
    Teuchos::TimeMonitor timeMon(timer, true);
    invA_->apply(X, Y, mode, alpha, beta);
  }  // end timing

  ++NumApply_;
  ApplyTime_ += (timer.wallTime() - startTime);
}  // apply()

template <class MatrixType>
std::string SparseApproximateInverse<MatrixType>::description() const {
  std::ostringstream os;

  // Output is a valid YAML dictionary in flow style.  If you don't
  // like everything on a single line, you should call describe()
  // instead.
  os << "\"Ifpack2::SparseApproximateInverse\": {";
  os << "Initialized: " << (isInitialized() ? "true" : "false") << ", "
     << "Computed: " << (isComputed() ? "true" : "false") << ", ";

  if (A_.is_null()) {
    os << "Matrix: null";
  } else {
    os << "Global matrix dimensions: ["
       << A_->getGlobalNumRows() << ", " << A_->getGlobalNumCols() << "]"
       << ", Global nnz: " << A_->getGlobalNumEntries();
  }

  os << "}";
  return os.str();
}

template <class MatrixType>
void SparseApproximateInverse<MatrixType>::
    describe(Teuchos::FancyOStream& out,
             const Teuchos::EVerbosityLevel verbLevel) const {
  using std::endl;
  using Teuchos::Comm;
  using Teuchos::OSTab;
  using Teuchos::RCP;
  using Teuchos::TypeNameTraits;
  using Teuchos::VERB_DEFAULT;
  using Teuchos::VERB_EXTREME;
  using Teuchos::VERB_HIGH;
  using Teuchos::VERB_LOW;
  using Teuchos::VERB_MEDIUM;
  using Teuchos::VERB_NONE;

  const Teuchos::EVerbosityLevel vl =
      (verbLevel == VERB_DEFAULT) ? VERB_LOW : verbLevel;
  OSTab tab0(out);

  if (vl > VERB_NONE) {
    out << "\"Ifpack2::SparseApproximateInverse\":" << endl;
    OSTab tab1(out);
    out << "MatrixType: " << TypeNameTraits<MatrixType>::name() << endl;
    if (this->getObjectLabel() != "") {
      out << "Label: \"" << this->getObjectLabel() << "\"" << endl;
    }
    out << "Initialized: " << (isInitialized() ? "true" : "false")
        << endl
        << "Computed: " << (isComputed() ? "true" : "false")
        << endl;
  }

  if (isComputed() && vl >= VERB_HIGH) {
    const double fillFraction =
        (double)getGlobalNumEntries() / (double)A_->getGlobalNumEntries();

    out << "Number of initialize calls: " << getNumInitialize() << endl
        << "Number of compute calls: " << getNumCompute() << endl
        << "Number of apply calls: " << getNumApply() << endl
        << "Total time in seconds for initialize: " << getInitializeTime() << endl
        << "Total time in seconds for compute: " << getComputeTime() << endl
        << "Total time in seconds for apply: " << getApplyTime() << endl;
  }
}

}  // namespace Ifpack2

#define IFPACK2_SPARSEAPPROXIMATEINVERSE_INSTANT(S, LO, GO, N)                       \
  template class Ifpack2::SparseApproximateInverse<Tpetra::RowMatrix<S, LO, GO, N>>; \
  template Teuchos::RCP<Tpetra::CrsMatrix<S, LO, GO, N>>                             \
  Ifpack2::GetSparseApproximateInverse(const Tpetra::CrsMatrix<S, LO, GO, N>&,       \
                                       const Teuchos::RCP<const Tpetra::CrsGraph<LO, GO, N>>&);

#endif /* IFPACK2_SPARSEAPPROXIMATEINVERSE_DEF_HPP */
