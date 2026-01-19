// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
//
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef TPETRAEXT_MATRIXMATRIX_SERIAL_DEF_HPP
#define TPETRAEXT_MATRIXMATRIX_SERIAL_DEF_HPP

#ifdef HAVE_TPETRA_INST_SERIAL
namespace Tpetra {
namespace MMdetails {

/*********************************************************************************************************/
// MMM KernelWrappers for Partial Specialization to Serial
template <class Scalar,
          class LocalOrdinal,
          class GlobalOrdinal,
          class LocalOrdinalViewType>
struct KernelWrappers<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosSerialWrapperNode, LocalOrdinalViewType> {
  static void mult_A_B_newmatrix_kernel_wrapper(CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosSerialWrapperNode>& Aview,
                                                CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosSerialWrapperNode>& Bview,
                                                const LocalOrdinalViewType& Acol2Brow,
                                                const LocalOrdinalViewType& Acol2Irow,
                                                const LocalOrdinalViewType& Bcol2Ccol,
                                                const LocalOrdinalViewType& Icol2Ccol,
                                                CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosSerialWrapperNode>& C,
                                                Teuchos::RCP<const Import<LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosSerialWrapperNode>> Cimport,
                                                const std::string& label                           = std::string(),
                                                const Teuchos::RCP<Teuchos::ParameterList>& params = Teuchos::null);

  static void mult_A_B_reuse_kernel_wrapper(CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosSerialWrapperNode>& Aview,
                                            CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosSerialWrapperNode>& Bview,
                                            const LocalOrdinalViewType& Acol2Brow,
                                            const LocalOrdinalViewType& Acol2Irow,
                                            const LocalOrdinalViewType& Bcol2Ccol,
                                            const LocalOrdinalViewType& Icol2Ccol,
                                            CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosSerialWrapperNode>& C,
                                            Teuchos::RCP<const Import<LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosSerialWrapperNode>> Cimport,
                                            const std::string& label                           = std::string(),
                                            const Teuchos::RCP<Teuchos::ParameterList>& params = Teuchos::null) {
    mult_A_B_reuse_kernel_wrapper_fun(Aview, Bview, Acol2Brow, Acol2Irow, Bcol2Ccol, Icol2Ccol, C, Cimport, label, params);
  }

};

// Jacobi KernelWrappers for Partial Specialization to Serial
template <class Scalar,
          class LocalOrdinal,
          class GlobalOrdinal, class LocalOrdinalViewType>
struct KernelWrappers2<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosSerialWrapperNode, LocalOrdinalViewType> {
  static void jacobi_A_B_newmatrix_kernel_wrapper(Scalar omega,
                                                  const Vector<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosSerialWrapperNode>& Dinv,
                                                  CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosSerialWrapperNode>& Aview,
                                                  CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosSerialWrapperNode>& Bview,
                                                  const LocalOrdinalViewType& Acol2Brow,
                                                  const LocalOrdinalViewType& Acol2Irow,
                                                  const LocalOrdinalViewType& Bcol2Ccol,
                                                  const LocalOrdinalViewType& Icol2Ccol,
                                                  CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosSerialWrapperNode>& C,
                                                  Teuchos::RCP<const Import<LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosSerialWrapperNode>> Cimport,
                                                  const std::string& label                           = std::string(),
                                                  const Teuchos::RCP<Teuchos::ParameterList>& params = Teuchos::null);

  static void jacobi_A_B_reuse_kernel_wrapper(Scalar omega,
                                              const Vector<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosSerialWrapperNode>& Dinv,
                                              CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosSerialWrapperNode>& Aview,
                                              CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosSerialWrapperNode>& Bview,
                                              const LocalOrdinalViewType& Acol2Brow,
                                              const LocalOrdinalViewType& Acol2Irow,
                                              const LocalOrdinalViewType& Bcol2Ccol,
                                              const LocalOrdinalViewType& Icol2Ccol,
                                              CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosSerialWrapperNode>& C,
                                              Teuchos::RCP<const Import<LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosSerialWrapperNode>> Cimport,
                                              const std::string& label                           = std::string(),
                                              const Teuchos::RCP<Teuchos::ParameterList>& params = Teuchos::null) {
    jacobi_A_B_reuse_kernel_wrapper_fun(omega, Dinv, Aview, Bview, Acol2Brow, Acol2Irow, Bcol2Ccol, Icol2Ccol, C, Cimport, label, params);
  }

  static void jacobi_A_B_newmatrix_KokkosKernels(Scalar omega,
                                                 const Vector<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosSerialWrapperNode>& Dinv,
                                                 CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosSerialWrapperNode>& Aview,
                                                 CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosSerialWrapperNode>& Bview,
                                                 const LocalOrdinalViewType& Acol2Brow,
                                                 const LocalOrdinalViewType& Acol2Irow,
                                                 const LocalOrdinalViewType& Bcol2Ccol,
                                                 const LocalOrdinalViewType& Icol2Ccol,
                                                 CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosSerialWrapperNode>& C,
                                                 Teuchos::RCP<const Import<LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosSerialWrapperNode>> Cimport,
                                                 const std::string& label                           = std::string(),
                                                 const Teuchos::RCP<Teuchos::ParameterList>& params = Teuchos::null);
};

template <class Scalar,
          class LocalOrdinal,
          class GlobalOrdinal,
          class LocalOrdinalViewType>
void KernelWrappers<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosSerialWrapperNode, LocalOrdinalViewType>::mult_A_B_newmatrix_kernel_wrapper(CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosSerialWrapperNode>& Aview,
                                                                                                                                                                 CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosSerialWrapperNode>& Bview,
                                                                                                                        const LocalOrdinalViewType& targetMapToOrigRow,
                                                                                                                        const LocalOrdinalViewType& targetMapToImportRow,
                                                                                                                        const LocalOrdinalViewType& Bcol2Ccol,
                                                                                                                        const LocalOrdinalViewType& Icol2Ccol,
                                                                                                                        CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosSerialWrapperNode>& C,
                                                                                                                        Teuchos::RCP<const Import<LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosSerialWrapperNode>> Cimport,
                                                                                                                        const std::string& label,
                                                                                                                        const Teuchos::RCP<Teuchos::ParameterList>& params) {
  using Teuchos::Array;
  using Teuchos::ArrayRCP;
  using Teuchos::ArrayView;
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Node = Tpetra::KokkosCompat::KokkosSerialWrapperNode;

  Tpetra::Details::ProfilingRegion MM("TpetraExt: MMM: Newmatrix SerialCore");

  // Lots and lots of typedefs
  typedef typename Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::local_matrix_host_type KCRS;
  typedef typename KCRS::StaticCrsGraphType graph_t;
  typedef typename graph_t::row_map_type::const_type c_lno_view_t;
  typedef typename graph_t::row_map_type::non_const_type lno_view_t;
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

  // Sizes
  RCP<const map_type> Ccolmap = C.getColMap();
  size_t m                    = Aview.origMatrix->getLocalNumRows();
  size_t n                    = Ccolmap->getLocalNumElements();
  size_t b_max_nnz_per_row    = Bview.origMatrix->getLocalMaxNumRowEntries();

  // Grab the  Kokkos::SparseCrsMatrices & inner stuff
  const KCRS Amat = Aview.origMatrix->getLocalMatrixHost();
  const KCRS Bmat = Bview.origMatrix->getLocalMatrixHost();

  c_lno_view_t Arowptr = Amat.graph.row_map, Browptr = Bmat.graph.row_map;
  const lno_nnz_view_t Acolind = Amat.graph.entries, Bcolind = Bmat.graph.entries;
  const scalar_view_t Avals = Amat.values, Bvals = Bmat.values;

  c_lno_view_t Irowptr;
  lno_nnz_view_t Icolind;
  scalar_view_t Ivals;
  if (!Bview.importMatrix.is_null()) {
    auto lclB         = Bview.importMatrix->getLocalMatrixHost();
    Irowptr           = lclB.graph.row_map;
    Icolind           = lclB.graph.entries;
    Ivals             = lclB.values;
    b_max_nnz_per_row = std::max(b_max_nnz_per_row, Bview.importMatrix->getLocalMaxNumRowEntries());
  }

  // Classic csr assembly (low memory edition)
  //
  // mfh 27 Sep 2016: C_estimate_nnz does not promise an upper bound.
  // The method loops over rows of A, and may resize after processing
  // each row.  Chris Siefert says that this reflects experience in
  // ML; for the non-threaded case, ML found it faster to spend less
  // effort on estimation and risk an occasional reallocation.
  size_t CSR_alloc = std::max(C_estimate_nnz(*Aview.origMatrix, *Bview.origMatrix), n);
  lno_view_t Crowptr(Kokkos::ViewAllocateWithoutInitializing("Crowptr"), m + 1);
  lno_nnz_view_t Ccolind(Kokkos::ViewAllocateWithoutInitializing("Ccolind"), CSR_alloc);
  scalar_view_t Cvals(Kokkos::ViewAllocateWithoutInitializing("Cvals"), CSR_alloc);

  // mfh 27 Sep 2016: The c_status array is an implementation detail
  // of the local sparse matrix-matrix multiply routine.

  // The status array will contain the index into colind where this entry was last deposited.
  //   c_status[i] <  CSR_ip - not in the row yet
  //   c_status[i] >= CSR_ip - this is the entry where you can find the data
  // We start with this filled with INVALID's indicating that there are no entries yet.
  // Sadly, this complicates the code due to the fact that size_t's are unsigned.
  size_t INVALID = Teuchos::OrdinalTraits<size_t>::invalid();
  std::vector<size_t> c_status(n, ST_INVALID);

  // mfh 27 Sep 2016: Here is the local sparse matrix-matrix multiply
  // routine.  The routine computes C := A * (B_local + B_remote).
  //
  // For column index Aik in row i of A, targetMapToOrigRow[Aik] tells
  // you whether the corresponding row of B belongs to B_local
  // ("orig") or B_remote ("Import").

  // For each row of A/C
  size_t CSR_ip = 0, OLD_ip = 0;
  for (size_t i = 0; i < m; i++) {
    // mfh 27 Sep 2016: m is the number of rows in the input matrix A
    // on the calling process.
    Crowptr[i] = CSR_ip;

    // mfh 27 Sep 2016: For each entry of A in the current row of A
    for (size_t k = Arowptr[i]; k < Arowptr[i + 1]; k++) {
      LO Aik        = Acolind[k];  // local column index of current entry of A
      const SC Aval = Avals[k];    // value of current entry of A
      if (Aval == SC_ZERO)
        continue;  // skip explicitly stored zero values in A

      if (targetMapToOrigRow[Aik] != LO_INVALID) {
        // mfh 27 Sep 2016: If the entry of targetMapToOrigRow
        // corresponding to the current entry of A is populated, then
        // the corresponding row of B is in B_local (i.e., it lives on
        // the calling process).

        // Local matrix
        size_t Bk = static_cast<size_t>(targetMapToOrigRow[Aik]);

        // mfh 27 Sep 2016: Go through all entries in that row of B_local.
        for (size_t j = Browptr[Bk]; j < Browptr[Bk + 1]; ++j) {
          LO Bkj = Bcolind[j];
          LO Cij = Bcol2Ccol[Bkj];

          if (c_status[Cij] == INVALID || c_status[Cij] < OLD_ip) {
            // New entry
            c_status[Cij]   = CSR_ip;
            Ccolind[CSR_ip] = Cij;
            Cvals[CSR_ip]   = Aval * Bvals[j];
            CSR_ip++;

          } else {
            Cvals[c_status[Cij]] += Aval * Bvals[j];
          }
        }

      } else {
        // mfh 27 Sep 2016: If the entry of targetMapToOrigRow
        // corresponding to the current entry of A NOT populated (has
        // a flag "invalid" value), then the corresponding row of B is
        // in B_local (i.e., it lives on the calling process).

        // Remote matrix
        size_t Ik = static_cast<size_t>(targetMapToImportRow[Aik]);
        for (size_t j = Irowptr[Ik]; j < Irowptr[Ik + 1]; ++j) {
          LO Ikj = Icolind[j];
          LO Cij = Icol2Ccol[Ikj];

          if (c_status[Cij] == INVALID || c_status[Cij] < OLD_ip) {
            // New entry
            c_status[Cij]   = CSR_ip;
            Ccolind[CSR_ip] = Cij;
            Cvals[CSR_ip]   = Aval * Ivals[j];
            CSR_ip++;
          } else {
            Cvals[c_status[Cij]] += Aval * Ivals[j];
          }
        }
      }
    }

    // Resize for next pass if needed
    if (i + 1 < m && CSR_ip + std::min(n, (Arowptr[i + 2] - Arowptr[i + 1]) * b_max_nnz_per_row) > CSR_alloc) {
      CSR_alloc *= 2;
      Kokkos::resize(Ccolind, CSR_alloc);
      Kokkos::resize(Cvals, CSR_alloc);
    }
    OLD_ip = CSR_ip;
  }

  Crowptr[m] = CSR_ip;

  // Downward resize
  Kokkos::resize(Ccolind, CSR_ip);
  Kokkos::resize(Cvals, CSR_ip);

  {
    Tpetra::Details::ProfilingRegion MM3("TpetraExt: MMM: Newmatrix Final Sort");

    // Final sort & set of CRS arrays
    if (params.is_null() || params->get("sort entries", Details::MatrixTraits<Scalar, LocalOrdinal, GlobalOrdinal, Node>::spgemmNeedsSortedInputs()))
      Import_Util::sortCrsEntries(Crowptr, Ccolind, Cvals);
    C.setAllValues(Crowptr, Ccolind, Cvals);
  }

  Tpetra::Details::ProfilingRegion MM4("TpetraExt: MMM: Newmatrix ESCC");
  {
    // Final FillComplete
    //
    // mfh 27 Sep 2016: So-called "expert static fill complete" bypasses
    // Import (from domain Map to column Map) construction (which costs
    // lots of communication) by taking the previously constructed
    // Import object.  We should be able to do this without interfering
    // with the implementation of the local part of sparse matrix-matrix
    // multply above.
    RCP<Teuchos::ParameterList> labelList = rcp(new Teuchos::ParameterList);
    labelList->set("Timer Label", label);
    if (!params.is_null()) labelList->set("compute global constants", params->get("compute global constants", true));
    RCP<const Export<LO, GO, NO>> dummyExport;
    C.expertStaticFillComplete(Bview.origMatrix->getDomainMap(), Aview.origMatrix->getRangeMap(), Cimport, dummyExport, labelList);
  }
}

/*********************************************************************************************************/
template <class Scalar,
          class LocalOrdinal,
          class GlobalOrdinal,
          class LocalOrdinalViewType>
void KernelWrappers2<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosSerialWrapperNode, LocalOrdinalViewType>::jacobi_A_B_newmatrix_kernel_wrapper(Scalar omega,
                                                                                                                                                                    const Vector<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosSerialWrapperNode>& Dinv,
                                                                                                                                                                    CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosSerialWrapperNode>& Aview,
                                                                                                                                                                    CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosSerialWrapperNode>& Bview,
                                                                                                                                                                    const LocalOrdinalViewType& targetMapToOrigRow,
                                                                                                                                                                    const LocalOrdinalViewType& targetMapToImportRow,
                                                                                                                                                                    const LocalOrdinalViewType& Bcol2Ccol,
                                                                                                                                                                    const LocalOrdinalViewType& Icol2Ccol,
                                                                                                                                                                    CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosSerialWrapperNode>& C,
                                                                                                                                                                    Teuchos::RCP<const Import<LocalOrdinal, GlobalOrdinal, Tpetra::KokkosCompat::KokkosSerialWrapperNode>> Cimport,
                                                                                                                                                                    const std::string& label,
                                                                                                                                                                    const Teuchos::RCP<Teuchos::ParameterList>& params) {
  using Node = Tpetra::KokkosCompat::KokkosSerialWrapperNode;
  Tpetra::Details::ProfilingRegion MM("TpetraExt: Jacobi: Newmatrix SerialCore");

  using Teuchos::Array;
  using Teuchos::ArrayRCP;
  using Teuchos::ArrayView;
  using Teuchos::RCP;
  using Teuchos::rcp;

  // Lots and lots of typedefs
  typedef typename Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::local_matrix_host_type KCRS;
  typedef typename KCRS::StaticCrsGraphType graph_t;
  typedef typename graph_t::row_map_type::const_type c_lno_view_t;
  typedef typename graph_t::row_map_type::non_const_type lno_view_t;
  typedef typename graph_t::entries_type::non_const_type lno_nnz_view_t;
  typedef typename KCRS::values_type::non_const_type scalar_view_t;

  // Jacobi-specific
  typedef typename scalar_view_t::memory_space scalar_memory_space;

  typedef Scalar SC;
  typedef LocalOrdinal LO;
  typedef GlobalOrdinal GO;
  typedef Node NO;

  typedef Map<LO, GO, NO> map_type;
  size_t ST_INVALID = Teuchos::OrdinalTraits<LO>::invalid();
  LO LO_INVALID     = Teuchos::OrdinalTraits<LO>::invalid();

  // Sizes
  RCP<const map_type> Ccolmap = C.getColMap();
  size_t m                    = Aview.origMatrix->getLocalNumRows();
  size_t n                    = Ccolmap->getLocalNumElements();
  size_t b_max_nnz_per_row    = Bview.origMatrix->getLocalMaxNumRowEntries();

  // Grab the  Kokkos::SparseCrsMatrices & inner stuff
  const KCRS Amat = Aview.origMatrix->getLocalMatrixHost();
  const KCRS Bmat = Bview.origMatrix->getLocalMatrixHost();

  c_lno_view_t Arowptr = Amat.graph.row_map, Browptr = Bmat.graph.row_map;
  const lno_nnz_view_t Acolind = Amat.graph.entries, Bcolind = Bmat.graph.entries;
  const scalar_view_t Avals = Amat.values, Bvals = Bmat.values;

  c_lno_view_t Irowptr;
  lno_nnz_view_t Icolind;
  scalar_view_t Ivals;
  if (!Bview.importMatrix.is_null()) {
    auto lclB         = Bview.importMatrix->getLocalMatrixHost();
    Irowptr           = lclB.graph.row_map;
    Icolind           = lclB.graph.entries;
    Ivals             = lclB.values;
    b_max_nnz_per_row = std::max(b_max_nnz_per_row, Bview.importMatrix->getLocalMaxNumRowEntries());
  }

  // Jacobi-specific inner stuff
  auto Dvals =
      Dinv.template getLocalView<scalar_memory_space>(Access::ReadOnly);

  // Teuchos::ArrayView::operator[].
  // The status array will contain the index into colind where this entry was last deposited.
  // c_status[i] < CSR_ip - not in the row yet.
  // c_status[i] >= CSR_ip, this is the entry where you can find the data
  // We start with this filled with INVALID's indicating that there are no entries yet.
  // Sadly, this complicates the code due to the fact that size_t's are unsigned.
  size_t INVALID = Teuchos::OrdinalTraits<size_t>::invalid();
  Array<size_t> c_status(n, ST_INVALID);

  // Classic csr assembly (low memory edition)
  //
  // mfh 27 Sep 2016: C_estimate_nnz does not promise an upper bound.
  // The method loops over rows of A, and may resize after processing
  // each row.  Chris Siefert says that this reflects experience in
  // ML; for the non-threaded case, ML found it faster to spend less
  // effort on estimation and risk an occasional reallocation.
  size_t CSR_alloc = std::max(C_estimate_nnz(*Aview.origMatrix, *Bview.origMatrix), n);
  lno_view_t Crowptr(Kokkos::ViewAllocateWithoutInitializing("Crowptr"), m + 1);
  lno_nnz_view_t Ccolind(Kokkos::ViewAllocateWithoutInitializing("Ccolind"), CSR_alloc);
  scalar_view_t Cvals(Kokkos::ViewAllocateWithoutInitializing("Cvals"), CSR_alloc);
  size_t CSR_ip = 0, OLD_ip = 0;

  const SC SC_ZERO = Teuchos::ScalarTraits<Scalar>::zero();

  // mfh 27 Sep 2016: Here is the local sparse matrix-matrix multiply
  // routine.  The routine computes
  //
  // C := (I - omega * D^{-1} * A) * (B_local + B_remote)).
  //
  // This corresponds to one sweep of (weighted) Jacobi.
  //
  // For column index Aik in row i of A, targetMapToOrigRow[Aik] tells
  // you whether the corresponding row of B belongs to B_local
  // ("orig") or B_remote ("Import").

  // For each row of A/C
  for (size_t i = 0; i < m; i++) {
    // mfh 27 Sep 2016: m is the number of rows in the input matrix A
    // on the calling process.
    Crowptr[i]        = CSR_ip;
    SC minusOmegaDval = -omega * Dvals(i, 0);

    // Entries of B
    for (size_t j = Browptr[i]; j < Browptr[i + 1]; j++) {
      Scalar Bval = Bvals[j];
      if (Bval == SC_ZERO)
        continue;
      LO Bij = Bcolind[j];
      LO Cij = Bcol2Ccol[Bij];

      // Assume no repeated entries in B
      c_status[Cij]   = CSR_ip;
      Ccolind[CSR_ip] = Cij;
      Cvals[CSR_ip]   = Bvals[j];
      CSR_ip++;
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

          if (c_status[Cij] == INVALID || c_status[Cij] < OLD_ip) {
            // New entry
            c_status[Cij]   = CSR_ip;
            Ccolind[CSR_ip] = Cij;
            Cvals[CSR_ip]   = minusOmegaDval * Aval * Bvals[j];
            CSR_ip++;

          } else {
            Cvals[c_status[Cij]] += minusOmegaDval * Aval * Bvals[j];
          }
        }

      } else {
        // Remote matrix
        size_t Ik = static_cast<size_t>(targetMapToImportRow[Aik]);
        for (size_t j = Irowptr[Ik]; j < Irowptr[Ik + 1]; ++j) {
          LO Ikj = Icolind[j];
          LO Cij = Icol2Ccol[Ikj];

          if (c_status[Cij] == INVALID || c_status[Cij] < OLD_ip) {
            // New entry
            c_status[Cij]   = CSR_ip;
            Ccolind[CSR_ip] = Cij;
            Cvals[CSR_ip]   = minusOmegaDval * Aval * Ivals[j];
            CSR_ip++;
          } else {
            Cvals[c_status[Cij]] += minusOmegaDval * Aval * Ivals[j];
          }
        }
      }
    }

    // Resize for next pass if needed
    if (i + 1 < m && CSR_ip + std::min(n, (Arowptr[i + 2] - Arowptr[i + 1] + 1) * b_max_nnz_per_row) > CSR_alloc) {
      CSR_alloc *= 2;
      Kokkos::resize(Ccolind, CSR_alloc);
      Kokkos::resize(Cvals, CSR_alloc);
    }
    OLD_ip = CSR_ip;
  }
  Crowptr[m] = CSR_ip;

  // Downward resize
  Kokkos::resize(Ccolind, CSR_ip);
  Kokkos::resize(Cvals, CSR_ip);

  {
    Tpetra::Details::ProfilingRegion MM2("TpetraExt: Jacobi: Newmatrix Final Sort");

    // Replace the column map
    //
    // mfh 27 Sep 2016: We do this because C was originally created
    // without a column Map.  Now we have its column Map.
    C.replaceColMap(Ccolmap);

    // Final sort & set of CRS arrays
    //
    // TODO (mfh 27 Sep 2016) Will the thread-parallel "local" sparse
    // matrix-matrix multiply routine sort the entries for us?
    // Final sort & set of CRS arrays
    if (params.is_null() || params->get("sort entries", Details::MatrixTraits<Scalar, LocalOrdinal, GlobalOrdinal, Node>::spgemmNeedsSortedInputs()))
      Import_Util::sortCrsEntries(Crowptr, Ccolind, Cvals);
    C.setAllValues(Crowptr, Ccolind, Cvals);
  }
  {
    Tpetra::Details::ProfilingRegion MM3("TpetraExt: Jacobi: Newmatrix ESFC");

    // Final FillComplete
    //
    // mfh 27 Sep 2016: So-called "expert static fill complete" bypasses
    // Import (from domain Map to column Map) construction (which costs
    // lots of communication) by taking the previously constructed
    // Import object.  We should be able to do this without interfering
    // with the implementation of the local part of sparse matrix-matrix
    // multply above
    RCP<Teuchos::ParameterList> labelList = rcp(new Teuchos::ParameterList);
    labelList->set("Timer Label", label);
    if (!params.is_null()) labelList->set("compute global constants", params->get("compute global constants", true));
    RCP<const Export<LO, GO, NO>> dummyExport;
    C.expertStaticFillComplete(Bview.origMatrix->getDomainMap(), Aview.origMatrix->getRangeMap(), Cimport, dummyExport, labelList);
  }
}

}  // namespace MMdetails
}  // namespace Tpetra

#endif

#endif
