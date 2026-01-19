#ifndef TPETRAEXT_MATRIXMATRIX_KERNELWRAPPERS_HPP
#define TPETRAEXT_MATRIXMATRIX_KERNELWRAPPERS_HPP

namespace Tpetra::MMdetails {

template <class Scalar,
          class LocalOrdinal,
          class GlobalOrdinal,
          class Node,
          class LocalOrdinalViewType>
void mult_A_B_reuse_kernel_wrapper_fun(CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Node>& Aview,
                                   CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Node>& Bview,
                                   const LocalOrdinalViewType& targetMapToOrigRow_dev,
                                   const LocalOrdinalViewType& targetMapToImportRow_dev,
                                   const LocalOrdinalViewType& Bcol2Ccol_dev,
                                   const LocalOrdinalViewType& Icol2Ccol_dev,
                                   CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>& C,
                                   Teuchos::RCP<const Import<LocalOrdinal, GlobalOrdinal, Node>> Cimport,
                                   const std::string& label,
                                   const Teuchos::RCP<Teuchos::ParameterList>& params) {
  using Teuchos::RCP;
  using Teuchos::rcp;

  // Lots and lots of typedefs
  typedef typename Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::local_matrix_host_type KCRS;
  typedef typename KCRS::StaticCrsGraphType graph_t;
  typedef typename graph_t::row_map_type::const_type c_lno_view_t;
  typedef typename graph_t::entries_type::non_const_type lno_nnz_view_t;
  typedef typename KCRS::values_type::non_const_type scalar_view_t;

  typedef Scalar SC;
  typedef LocalOrdinal LO;
  typedef GlobalOrdinal GO;
  typedef Node NO;
  typedef Map<LO, GO, NO> map_type;
  const size_t ST_INVALID = Teuchos::OrdinalTraits<LO>::invalid();
  const LO LO_INVALID     = Teuchos::OrdinalTraits<LO>::invalid();
  const SC SC_ZERO        = Teuchos::ScalarTraits<Scalar>::zero();

  Tpetra::Details::ProfilingRegion MM("TpetraExt: MMM: Reuse SerialCore");
  // Since this is being run on Cuda, we need to fence because the below code will use UVM
  // typename graph_t::execution_space().fence();

  // KDDKDD UVM Without UVM, need to copy targetMap arrays to host.
  // KDDKDD UVM Ideally, this function would run on device and use
  // KDDKDD UVM KokkosKernels instead of this host implementation.
  auto targetMapToOrigRow =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                          targetMapToOrigRow_dev);
  auto targetMapToImportRow =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                          targetMapToImportRow_dev);
  auto Bcol2Ccol =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                          Bcol2Ccol_dev);
  auto Icol2Ccol =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                          Icol2Ccol_dev);

  // Sizes
  RCP<const map_type> Ccolmap = C.getColMap();
  size_t m                    = Aview.origMatrix->getLocalNumRows();
  size_t n                    = Ccolmap->getLocalNumElements();

  // Grab the  Kokkos::SparseCrsMatrices & inner stuff
  const KCRS Amat = Aview.origMatrix->getLocalMatrixHost();
  const KCRS Bmat = Bview.origMatrix->getLocalMatrixHost();
  const KCRS Cmat = C.getLocalMatrixHost();

  c_lno_view_t Arowptr         = Amat.graph.row_map,
               Browptr         = Bmat.graph.row_map,
               Crowptr         = Cmat.graph.row_map;
  const lno_nnz_view_t Acolind = Amat.graph.entries,
                       Bcolind = Bmat.graph.entries,
                       Ccolind = Cmat.graph.entries;
  const scalar_view_t Avals = Amat.values, Bvals = Bmat.values;
  scalar_view_t Cvals = Cmat.values;

  c_lno_view_t Irowptr;
  lno_nnz_view_t Icolind;
  scalar_view_t Ivals;
  if (!Bview.importMatrix.is_null()) {
    auto lclB = Bview.importMatrix->getLocalMatrixHost();
    Irowptr   = lclB.graph.row_map;
    Icolind   = lclB.graph.entries;
    Ivals     = lclB.values;
  }

  // Classic csr assembly (low memory edition)
  // mfh 27 Sep 2016: The c_status array is an implementation detail
  // of the local sparse matrix-matrix multiply routine.

  // The status array will contain the index into colind where this entry was last deposited.
  //   c_status[i] <  CSR_ip - not in the row yet
  //   c_status[i] >= CSR_ip - this is the entry where you can find the data
  // We start with this filled with INVALID's indicating that there are no entries yet.
  // Sadly, this complicates the code due to the fact that size_t's are unsigned.
  std::vector<size_t> c_status(n, ST_INVALID);

  // For each row of A/C
  size_t CSR_ip = 0, OLD_ip = 0;
  for (size_t i = 0; i < m; i++) {
    // First fill the c_status array w/ locations where we're allowed to
    // generate nonzeros for this row
    OLD_ip = Crowptr[i];
    CSR_ip = Crowptr[i + 1];
    for (size_t k = OLD_ip; k < CSR_ip; k++) {
      c_status[Ccolind[k]] = k;

      // Reset values in the row of C
      Cvals[k] = SC_ZERO;
    }

    for (size_t k = Arowptr[i]; k < Arowptr[i + 1]; k++) {
      LO Aik        = Acolind[k];
      const SC Aval = Avals[k];
      if (Aval == SC_ZERO)
        continue;

      if (targetMapToOrigRow[Aik] != LO_INVALID) {
        // Local matrix
        size_t Bk = static_cast<size_t>(targetMapToOrigRow[Aik]);

        for (size_t j = Browptr[Bk]; j < Browptr[Bk + 1]; ++j) {
          LO Bkj = Bcolind[j];
          LO Cij = Bcol2Ccol[Bkj];

          TEUCHOS_TEST_FOR_EXCEPTION(c_status[Cij] < OLD_ip || c_status[Cij] >= CSR_ip,
                                     std::runtime_error, "Trying to insert a new entry (" << i << "," << Cij << ") into a static graph "
                                                                                          << "(c_status = " << c_status[Cij] << " of [" << OLD_ip << "," << CSR_ip << "))");

          Cvals[c_status[Cij]] += Aval * Bvals[j];
        }

      } else {
        // Remote matrix
        size_t Ik = static_cast<size_t>(targetMapToImportRow[Aik]);
        for (size_t j = Irowptr[Ik]; j < Irowptr[Ik + 1]; ++j) {
          LO Ikj = Icolind[j];
          LO Cij = Icol2Ccol[Ikj];

          TEUCHOS_TEST_FOR_EXCEPTION(c_status[Cij] < OLD_ip || c_status[Cij] >= CSR_ip,
                                     std::runtime_error, "Trying to insert a new entry (" << i << "," << Cij << ") into a static graph "
                                                                                          << "(c_status = " << c_status[Cij] << " of [" << OLD_ip << "," << CSR_ip << "))");

          Cvals[c_status[Cij]] += Aval * Ivals[j];
        }
      }
    }
  }

  {
    Tpetra::Details::ProfilingRegion MM3("TpetraExt: MMM: Reuse ESFC");
    C.fillComplete(C.getDomainMap(), C.getRangeMap());
  }
}


template <class Scalar,
          class LocalOrdinal,
          class GlobalOrdinal,
          class Node,
          class LocalOrdinalViewType>
void jacobi_A_B_reuse_kernel_wrapper_fun(Scalar omega,
                                                                                                                       const Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>& Dinv,
                                                                                                                       CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Node>& Aview,
                                                                                                                       CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Node>& Bview,
                                                                                                                       const LocalOrdinalViewType& targetMapToOrigRow_dev,
                                                                                                                       const LocalOrdinalViewType& targetMapToImportRow_dev,
                                                                                                                       const LocalOrdinalViewType& Bcol2Ccol_dev,
                                                                                                                       const LocalOrdinalViewType& Icol2Ccol_dev,
                                                                                                                       CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>& C,
                                                                                                                       Teuchos::RCP<const Import<LocalOrdinal, GlobalOrdinal, Node>> /* Cimport */,
                                                                                                                       const std::string& label,
                                                                                                                       const Teuchos::RCP<Teuchos::ParameterList>& /* params */) {
  Tpetra::Details::ProfilingRegion MM2("TpetraExt: Jacobi: Reuse Serial Core");
  using Teuchos::RCP;
  using Teuchos::rcp;

  // Lots and lots of typedefs
  typedef typename Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::local_matrix_host_type KCRS;
  typedef typename KCRS::StaticCrsGraphType graph_t;
  typedef typename graph_t::row_map_type::const_type c_lno_view_t;
  typedef typename graph_t::entries_type::non_const_type lno_nnz_view_t;
  typedef typename KCRS::values_type::non_const_type scalar_view_t;
  typedef typename scalar_view_t::memory_space scalar_memory_space;

  typedef Scalar SC;
  typedef LocalOrdinal LO;
  typedef GlobalOrdinal GO;
  typedef Node NO;
  typedef Map<LO, GO, NO> map_type;
  const size_t ST_INVALID = Teuchos::OrdinalTraits<LO>::invalid();
  const LO LO_INVALID     = Teuchos::OrdinalTraits<LO>::invalid();
  const SC SC_ZERO        = Teuchos::ScalarTraits<Scalar>::zero();

  // Since this is being run on Cuda, we need to fence because the below host code will use UVM
  // KDDKDD typename graph_t::execution_space().fence();

  // KDDKDD UVM Without UVM, need to copy targetMap arrays to host.
  // KDDKDD UVM Ideally, this function would run on device and use
  // KDDKDD UVM KokkosKernels instead of this host implementation.
  auto targetMapToOrigRow =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                          targetMapToOrigRow_dev);
  auto targetMapToImportRow =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                          targetMapToImportRow_dev);
  auto Bcol2Ccol =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                          Bcol2Ccol_dev);
  auto Icol2Ccol =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                          Icol2Ccol_dev);

  // Sizes
  RCP<const map_type> Ccolmap = C.getColMap();
  size_t m                    = Aview.origMatrix->getLocalNumRows();
  size_t n                    = Ccolmap->getLocalNumElements();

  // Grab the  Kokkos::SparseCrsMatrices & inner stuff
  const KCRS Amat = Aview.origMatrix->getLocalMatrixHost();
  const KCRS Bmat = Bview.origMatrix->getLocalMatrixHost();
  const KCRS Cmat = C.getLocalMatrixHost();

  c_lno_view_t Arowptr = Amat.graph.row_map, Browptr = Bmat.graph.row_map, Crowptr = Cmat.graph.row_map;
  const lno_nnz_view_t Acolind = Amat.graph.entries, Bcolind = Bmat.graph.entries, Ccolind = Cmat.graph.entries;
  const scalar_view_t Avals = Amat.values, Bvals = Bmat.values;
  scalar_view_t Cvals = Cmat.values;

  c_lno_view_t Irowptr;
  lno_nnz_view_t Icolind;
  scalar_view_t Ivals;
  if (!Bview.importMatrix.is_null()) {
    auto lclB = Bview.importMatrix->getLocalMatrixHost();
    Irowptr   = lclB.graph.row_map;
    Icolind   = lclB.graph.entries;
    Ivals     = lclB.values;
  }

  // Jacobi-specific inner stuff
  auto Dvals =
      Dinv.template getLocalView<scalar_memory_space>(Access::ReadOnly);

  // The status array will contain the index into colind where this entry was last deposited.
  //   c_status[i] <  CSR_ip - not in the row yet
  //   c_status[i] >= CSR_ip - this is the entry where you can find the data
  // We start with this filled with INVALID's indicating that there are no entries yet.
  // Sadly, this complicates the code due to the fact that size_t's are unsigned.
  std::vector<size_t> c_status(n, ST_INVALID);

  // For each row of A/C
  size_t CSR_ip = 0, OLD_ip = 0;
  for (size_t i = 0; i < m; i++) {
    // First fill the c_status array w/ locations where we're allowed to
    // generate nonzeros for this row
    OLD_ip = Crowptr[i];
    CSR_ip = Crowptr[i + 1];
    for (size_t k = OLD_ip; k < CSR_ip; k++) {
      c_status[Ccolind[k]] = k;

      // Reset values in the row of C
      Cvals[k] = SC_ZERO;
    }

    SC minusOmegaDval = -omega * Dvals(i, 0);

    // Entries of B
    for (size_t j = Browptr[i]; j < Browptr[i + 1]; j++) {
      Scalar Bval = Bvals[j];
      if (Bval == SC_ZERO)
        continue;
      LO Bij = Bcolind[j];
      LO Cij = Bcol2Ccol[Bij];

      TEUCHOS_TEST_FOR_EXCEPTION(c_status[Cij] < OLD_ip || c_status[Cij] >= CSR_ip,
                                 std::runtime_error, "Trying to insert a new entry into a static graph");

      Cvals[c_status[Cij]] = Bvals[j];
    }

    // Entries of -omega * Dinv * A * B
    for (size_t k = Arowptr[i]; k < Arowptr[i + 1]; k++) {
      LO Aik        = Acolind[k];
      const SC Aval = Avals[k];
      if (Aval == SC_ZERO)
        continue;

      if (targetMapToOrigRow[Aik] != LO_INVALID) {
        // Local matrix
        size_t Bk = static_cast<size_t>(targetMapToOrigRow[Aik]);

        for (size_t j = Browptr[Bk]; j < Browptr[Bk + 1]; ++j) {
          LO Bkj = Bcolind[j];
          LO Cij = Bcol2Ccol[Bkj];

          TEUCHOS_TEST_FOR_EXCEPTION(c_status[Cij] < OLD_ip || c_status[Cij] >= CSR_ip,
                                     std::runtime_error, "Trying to insert a new entry into a static graph");

          Cvals[c_status[Cij]] += minusOmegaDval * Aval * Bvals[j];
        }

      } else {
        // Remote matrix
        size_t Ik = static_cast<size_t>(targetMapToImportRow[Aik]);
        for (size_t j = Irowptr[Ik]; j < Irowptr[Ik + 1]; ++j) {
          LO Ikj = Icolind[j];
          LO Cij = Icol2Ccol[Ikj];

          TEUCHOS_TEST_FOR_EXCEPTION(c_status[Cij] < OLD_ip || c_status[Cij] >= CSR_ip,
                                     std::runtime_error, "Trying to insert a new entry into a static graph");

          Cvals[c_status[Cij]] += minusOmegaDval * Aval * Ivals[j];
        }
      }
    }
  }

  {
    Tpetra::Details::ProfilingRegion MM3("TpetraExt: Jacobi: Reuse ESFC");
    C.fillComplete(C.getDomainMap(), C.getRangeMap());
  }
}

}  // namespace Tpetra::Tpetra::MMdetails

#endif
