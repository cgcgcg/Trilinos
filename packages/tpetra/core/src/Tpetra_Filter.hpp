#ifndef TPETRA_FILTER_HPP
#define TPETRA_FILTER_HPP

#include "Kokkos_Core.hpp"
#include "Teuchos_RCP.hpp"

namespace Tpetra {

/// \brief Constructs a matrix consisting only of the entries for which the filter function evaluates to true.
///
/// This applies a filter function that takes as single argument the value of the matrix entry.
///
/// \param A [in] The source matrix that the filter will be applied to.
/// \param filter [in] A callable with signature (impl_scalar_type, ) -> bool.
/// \return The filtered matrix.
template <class matrix_type, class filter_type>
Teuchos::RCP<matrix_type> applyFilter_vals(const matrix_type& A, const filter_type& filter) {
  using local_ordinal_type = typename matrix_type::local_ordinal_type;
  using node_type          = typename matrix_type::node_type;
  using local_matrix_type  = typename matrix_type::local_matrix_device_type;
  using rowptr_type        = typename local_matrix_type::row_map_type::non_const_type;
  using colidx_type        = typename local_matrix_type::index_type::non_const_type;
  using values_type        = typename local_matrix_type::values_type::non_const_type;
  using exec_space         = typename node_type::execution_space;

  local_matrix_type newLclA;
  {
    rowptr_type rowptr;
    auto lclA = A.getLocalMatrixDevice();

    rowptr = rowptr_type("filtered_rowptr", lclA.numRows() + 1);
    local_ordinal_type NNZ;

    Kokkos::parallel_scan(
        "filterMatrix::rowptr_construction", Kokkos::RangePolicy<exec_space>(0, lclA.numRows()), KOKKOS_LAMBDA(const local_ordinal_type rlid, local_ordinal_type& nnz, const bool update) {
          auto row = lclA.rowConst(rlid);
          for (local_ordinal_type k = 0; k < row.length; ++k) {
            auto val = row.value(k);
            if (filter(val)) {
              ++nnz;
            }
          }
          if (update && (rlid + 2 < lclA.numRows() + 1))
            rowptr(rlid + 2) = nnz;
        },
        NNZ);

    auto colidx = colidx_type(Kokkos::ViewAllocateWithoutInitializing("filtered_colidx"), NNZ);
    auto values = values_type(Kokkos::ViewAllocateWithoutInitializing("filtered_values"), NNZ);

    Kokkos::parallel_for(
        "filterMatrix::matrix_fill", Kokkos::RangePolicy<exec_space>(0, lclA.numRows()), KOKKOS_LAMBDA(const local_ordinal_type rlid) {
          auto row = lclA.rowConst(rlid);
          for (local_ordinal_type k = 0; k < row.length; ++k) {
            auto val = row.value(k);
            if (filter(val)) {
              auto clid                = row.colidx(k);
              colidx(rowptr(rlid + 1)) = clid;
              values(rowptr(rlid + 1)) = val;
              ++rowptr(rlid + 1);
            }
          }
        });

    newLclA = local_matrix_type("filtered_matrix", lclA.numRows(), lclA.numCols(), NNZ, values, rowptr, colidx);
  }

  return Teuchos::rcp(new matrix_type(newLclA, A.getRowMap(), A.getColMap(), A.getDomainMap(), A.getRangeMap()));
}

/// \brief Constructs a graph consisting only of the entries for which the filter function evaluates to true.
///
/// This applies a filter function that takes as single argument the value of the matrix entry.
///
/// \param A [in] The source matrix that the filter will be applied to.
/// \param filter [in] A callable with signature (impl_scalar_type, ) -> bool.
/// \return The filtered graph.
template <class graph_type, class matrix_type, class filter_type>
Teuchos::RCP<graph_type> applyGraphFilter_vals(const matrix_type& A, const filter_type& filter) {
  using local_ordinal_type = typename matrix_type::local_ordinal_type;
  using node_type          = typename matrix_type::node_type;
  using local_graph_type   = typename graph_type::local_graph_device_type;
  using local_matrix_type  = typename matrix_type::local_matrix_device_type;
  using rowptr_type        = typename local_matrix_type::row_map_type::non_const_type;
  using colidx_type        = typename local_matrix_type::index_type::non_const_type;
  using values_type        = typename local_matrix_type::values_type::non_const_type;
  using exec_space         = typename node_type::execution_space;

  local_graph_type newLclA;
  {
    rowptr_type rowptr;
    auto lclA = A.getLocalMatrixDevice();

    rowptr = rowptr_type("filtered_rowptr", lclA.numRows() + 1);
    local_ordinal_type NNZ;

    Kokkos::parallel_scan(
        "filterMatrix::rowptr_construction", Kokkos::RangePolicy<exec_space>(0, lclA.numRows()), KOKKOS_LAMBDA(const local_ordinal_type rlid, local_ordinal_type& nnz, const bool update) {
          auto row = lclA.rowConst(rlid);
          for (local_ordinal_type k = 0; k < row.length; ++k) {
            auto val = row.value(k);
            if (filter(val)) {
              ++nnz;
            }
          }
          if (update && (rlid + 2 < lclA.numRows() + 1))
            rowptr(rlid + 2) = nnz;
        },
        NNZ);

    auto colidx = colidx_type(Kokkos::ViewAllocateWithoutInitializing("filtered_colidx"), NNZ);

    Kokkos::parallel_for(
        "filterMatrix::graph_fill", Kokkos::RangePolicy<exec_space>(0, lclA.numRows()), KOKKOS_LAMBDA(const local_ordinal_type rlid) {
          auto row = lclA.rowConst(rlid);
          for (local_ordinal_type k = 0; k < row.length; ++k) {
            auto val = row.value(k);
            if (filter(val)) {
              auto clid                = row.colidx(k);
              colidx(rowptr(rlid + 1)) = clid;
              ++rowptr(rlid + 1);
            }
          }
        });

    newLclA = local_graph_type(colidx, rowptr);
  }

  return Teuchos::rcp(new graph_type(newLclA, A.getRowMap(), A.getColMap(), A.getDomainMap(), A.getRangeMap()));
}

/// \brief Constructs a matrix consisting only of the entries for which the filter function evaluates to true.
///
/// This applies a filter function that takes as arguments the local row and column ids and the value of the matrix entry.
///
/// \param A [in] The source matrix that the filter will be applied to.
/// \param filter [in] A callable with signature (local_ordinal_type, local_ordinal_type, impl_scalar_type) -> bool.
/// \return The filtered matrix.
template <class matrix_type, class filter_type>
Teuchos::RCP<matrix_type> applyFilter_LID(const matrix_type& A, const filter_type& filter) {
  using local_ordinal_type = typename matrix_type::local_ordinal_type;
  using node_type          = typename matrix_type::node_type;
  using local_matrix_type  = typename matrix_type::local_matrix_device_type;
  using rowptr_type        = typename local_matrix_type::row_map_type::non_const_type;
  using colidx_type        = typename local_matrix_type::index_type::non_const_type;
  using values_type        = typename local_matrix_type::values_type::non_const_type;
  using exec_space         = typename node_type::execution_space;

  local_matrix_type newLclA;
  {
    rowptr_type rowptr;
    auto lclA = A.getLocalMatrixDevice();

    rowptr = rowptr_type("filtered_rowptr", lclA.numRows() + 1);
    local_ordinal_type NNZ;

    Kokkos::parallel_scan(
        "filterMatrix::rowptr_construction", Kokkos::RangePolicy<exec_space>(0, lclA.numRows()), KOKKOS_LAMBDA(const local_ordinal_type rlid, local_ordinal_type& nnz, const bool update) {
          auto row = lclA.rowConst(rlid);
          for (local_ordinal_type k = 0; k < row.length; ++k) {
            auto clid = row.colidx(k);
            auto val  = row.value(k);
            if (filter(rlid, clid, val)) {
              ++nnz;
            }
          }
          if (update && (rlid + 2 < lclA.numRows() + 1))
            rowptr(rlid + 2) = nnz;
        },
        NNZ);

    auto colidx = colidx_type(Kokkos::ViewAllocateWithoutInitializing("filtered_colidx"), NNZ);
    auto values = values_type(Kokkos::ViewAllocateWithoutInitializing("filtered_values"), NNZ);

    Kokkos::parallel_for(
        "filterMatrix::matrix_fill", Kokkos::RangePolicy<exec_space>(0, lclA.numRows()), KOKKOS_LAMBDA(const local_ordinal_type rlid) {
          auto row = lclA.rowConst(rlid);
          for (local_ordinal_type k = 0; k < row.length; ++k) {
            auto clid = row.colidx(k);
            auto val  = row.value(k);
            if (filter(rlid, clid, val)) {
              colidx(rowptr(rlid + 1)) = clid;
              values(rowptr(rlid + 1)) = val;
              ++rowptr(rlid + 1);
            }
          }
        });

    newLclA = local_matrix_type("filtered_matrix", lclA.numRows(), lclA.numCols(), NNZ, values, rowptr, colidx);
  }

  return Teuchos::rcp(new matrix_type(newLclA, A.getRowMap(), A.getColMap(), A.getDomainMap(), A.getRangeMap()));
}

/// \brief Constructs a graph consisting only of the entries for which the filter function evaluates to true.
///
/// This applies a filter function that takes as arguments the local row and column ids and the value of the matrix entry.
///
/// \param A [in] The source matrix that the filter will be applied to.
/// \param filter [in] A callable with signature (local_ordinal_type, local_ordinal_type, impl_scalar_type) -> bool.
/// \return The filtered graph.
template <class graph_type, class matrix_type, class filter_type>
Teuchos::RCP<graph_type> applyGraphFilter_LID(const matrix_type& A, const filter_type& filter) {
  using local_ordinal_type = typename matrix_type::local_ordinal_type;
  using node_type          = typename matrix_type::node_type;
  using local_graph_type   = typename graph_type::local_graph_device_type;
  using local_matrix_type  = typename matrix_type::local_matrix_device_type;
  using rowptr_type        = typename local_matrix_type::row_map_type::non_const_type;
  using colidx_type        = typename local_matrix_type::index_type::non_const_type;
  using values_type        = typename local_matrix_type::values_type::non_const_type;
  using exec_space         = typename node_type::execution_space;

  local_graph_type newLclA;
  {
    rowptr_type rowptr;
    auto lclA = A.getLocalMatrixDevice();

    rowptr = rowptr_type("filtered_rowptr", lclA.numRows() + 1);
    local_ordinal_type NNZ;

    Kokkos::parallel_scan(
        "filterMatrix::rowptr_construction", Kokkos::RangePolicy<exec_space>(0, lclA.numRows()), KOKKOS_LAMBDA(const local_ordinal_type rlid, local_ordinal_type& nnz, const bool update) {
          auto row = lclA.rowConst(rlid);
          for (local_ordinal_type k = 0; k < row.length; ++k) {
            auto clid = row.colidx(k);
            auto val  = row.value(k);
            if (filter(rlid, clid, val)) {
              ++nnz;
            }
          }
          if (update && (rlid + 2 < lclA.numRows() + 1))
            rowptr(rlid + 2) = nnz;
        },
        NNZ);

    auto colidx = colidx_type(Kokkos::ViewAllocateWithoutInitializing("filtered_colidx"), NNZ);

    Kokkos::parallel_for(
        "filterMatrix::graph_fill", Kokkos::RangePolicy<exec_space>(0, lclA.numRows()), KOKKOS_LAMBDA(const local_ordinal_type rlid) {
          auto row = lclA.rowConst(rlid);
          for (local_ordinal_type k = 0; k < row.length; ++k) {
            auto clid = row.colidx(k);
            auto val  = row.value(k);
            if (filter(rlid, clid, val)) {
              colidx(rowptr(rlid + 1)) = clid;
              ++rowptr(rlid + 1);
            }
          }
        });

    newLclA = local_graph_type(colidx, rowptr);
  }

  return Teuchos::rcp(new graph_type(newLclA, A.getRowMap(), A.getColMap(), A.getDomainMap(), A.getRangeMap()));
}

/// \brief Constructs a graph consisting only of the entries for which the filter function evaluates to true.
///
/// This applies a filter function that takes as arguments the local row and column ids and the value of the graph entry.
///
/// \param A [in] The source graph that the filter will be applied to.
/// \param filter [in] A callable with signature (local_ordinal_type, local_ordinal_type) -> bool.
/// \return The filtered graph.
template <class graph_type, class filter_type>
Teuchos::RCP<graph_type> applyGraphFilter_LID(const graph_type& G, const filter_type& filter) {
  using local_ordinal_type = typename graph_type::local_ordinal_type;
  using node_type          = typename graph_type::node_type;
  using local_graph_type   = typename graph_type::local_graph_device_type;
  using rowptr_type        = typename local_graph_type::row_map_type::non_const_type;
  using colidx_type        = typename local_graph_type::entries_type::non_const_type;
  using exec_space         = typename node_type::execution_space;

  local_graph_type newLclG;
  {
    rowptr_type rowptr;
    auto lclG = G.getLocalMatrixDevice();

    rowptr = rowptr_type("filtered_rowptr", lclG.numRows() + 1);
    local_ordinal_type NNZ;

    Kokkos::parallel_scan(
        "filterGraph::rowptr_construction", Kokkos::RangePolicy<exec_space>(0, lclG.numRows()), KOKKOS_LAMBDA(const local_ordinal_type rlid, local_ordinal_type& nnz, const bool update) {
          auto row = lclG.rowConst(rlid);
          for (local_ordinal_type k = 0; k < row.length; ++k) {
            auto clid = row.colidx(k);
            auto val  = row.value(k);
            if (filter(rlid, clid)) {
              ++nnz;
            }
          }
          if (update && (rlid + 2 < lclG.numRows() + 1))
            rowptr(rlid + 2) = nnz;
        },
        NNZ);

    auto colidx = colidx_type(Kokkos::ViewAllocateWithoutInitializing("filtered_colidx"), NNZ);

    Kokkos::parallel_for(
        "filterGraph::graph_fill", Kokkos::RangePolicy<exec_space>(0, lclG.numRows()), KOKKOS_LAMBDA(const local_ordinal_type rlid) {
          auto row = lclG.rowConst(rlid);
          for (local_ordinal_type k = 0; k < row.length; ++k) {
            auto clid = row.colidx(k);
            if (filter(rlid, clid)) {
              colidx(rowptr(rlid + 1)) = clid;
              ++rowptr(rlid + 1);
            }
          }
        });

    newLclG = local_graph_type(colidx, rowptr);
  }

  return Teuchos::rcp(new graph_type(newLclG, G.getRowMap(), G.getColMap(), G.getDomainMap(), G.getRangeMap()));
}

/// \brief Constructs a matrix consisting only of the entries for which the filter function evaluates to true.
///
/// This applies a filter function that takes as arguments the global row and column ids and the value of the matrix entry.
///
/// \param A [in] The source matrix that the filter will be applied to.
/// \param filter [in] A callable with signature (global_ordinal_type, global_ordinal_type, impl_scalar_type) -> bool.
/// \return The filtered matrix.
template <class matrix_type, class filter_type>
Teuchos::RCP<matrix_type> applyFilter_GID(const matrix_type& A, const filter_type& filter) {
  using local_ordinal_type = typename matrix_type::local_ordinal_type;
  using node_type          = typename matrix_type::node_type;
  using local_matrix_type  = typename matrix_type::local_matrix_device_type;
  using rowptr_type        = typename local_matrix_type::row_map_type::non_const_type;
  using colidx_type        = typename local_matrix_type::index_type::non_const_type;
  using values_type        = typename local_matrix_type::values_type::non_const_type;
  using exec_space         = typename node_type::execution_space;

  local_matrix_type newLclA;
  {
    rowptr_type rowptr;
    auto lclA = A.getLocalMatrixDevice();

    rowptr = rowptr_type("filtered_rowptr", lclA.numRows() + 1);
    local_ordinal_type NNZ;

    auto lclRowMap = A.getRowMap()->getLocalMap();
    auto lclColMap = A.getColMap()->getLocalMap();

    Kokkos::parallel_scan(
        "filterMatrix::rowptr_construction", Kokkos::RangePolicy<exec_space>(0, lclA.numRows()), KOKKOS_LAMBDA(const local_ordinal_type rlid, local_ordinal_type& nnz, const bool update) {
          auto row  = lclA.rowConst(rlid);
          auto rgid = lclRowMap.getGlobalElement(rlid);
          for (local_ordinal_type k = 0; k < row.length; ++k) {
            auto clid = row.colidx(k);
            auto cgid = lclRowMap.getGlobalElement(clid);
            auto val  = row.value(k);
            if (filter(rgid, cgid, val)) {
              ++nnz;
            }
          }
          if (update && (rlid + 2 < lclA.numRows() + 1))
            rowptr(rlid + 2) = nnz;
        },
        NNZ);

    auto colidx = colidx_type(Kokkos::ViewAllocateWithoutInitializing("filtered_colidx"), NNZ);
    auto values = values_type(Kokkos::ViewAllocateWithoutInitializing("filtered_values"), NNZ);

    Kokkos::parallel_for(
        "filterMatrix::matrix_fill", Kokkos::RangePolicy<exec_space>(0, lclA.numRows()), KOKKOS_LAMBDA(const local_ordinal_type rlid) {
          auto row  = lclA.rowConst(rlid);
          auto rgid = lclRowMap.getGlobalElement(rlid);
          for (local_ordinal_type k = 0; k < row.length; ++k) {
            auto clid = row.colidx(k);
            auto cgid = lclRowMap.getGlobalElement(clid);
            auto val  = row.value(k);
            if (filter(rgid, cgid, val)) {
              colidx(rowptr(rlid + 1)) = clid;
              values(rowptr(rlid + 1)) = val;
              ++rowptr(rlid + 1);
            }
          }
        });

    newLclA = local_matrix_type("filtered_matrix", lclA.numRows(), lclA.numCols(), NNZ, values, rowptr, colidx);
  }

  return Teuchos::rcp(new matrix_type(newLclA, A.getRowMap(), A.getColMap(), A.getDomainMap(), A.getRangeMap()));
}

/// \brief Constructs a graph consisting only of the entries for which the filter function evaluates to true.
///
/// This applies a filter function that takes as arguments the global row and column ids and the value of the matrix entry.
///
/// \param A [in] The source matrix that the filter will be applied to.
/// \param filter [in] A callable with signature (global_ordinal_type, global_ordinal_type, impl_scalar_type) -> bool.
/// \return The filtered graph.
template <class graph_type, class matrix_type, class filter_type>
Teuchos::RCP<graph_type> applyGraphFilter_GID(const matrix_type& A, const filter_type& filter) {
  using local_ordinal_type = typename matrix_type::local_ordinal_type;
  using node_type          = typename matrix_type::node_type;
  using local_graph_type   = typename graph_type::local_graph_device_type;
  using local_matrix_type  = typename matrix_type::local_matrix_device_type;
  using rowptr_type        = typename local_matrix_type::row_map_type::non_const_type;
  using colidx_type        = typename local_matrix_type::index_type::non_const_type;
  using values_type        = typename local_matrix_type::values_type::non_const_type;
  using exec_space         = typename node_type::execution_space;

  local_graph_type newLclA;
  {
    rowptr_type rowptr;
    auto lclA = A.getLocalMatrixDevice();

    rowptr = rowptr_type("filtered_rowptr", lclA.numRows() + 1);
    local_ordinal_type NNZ;

    auto lclRowMap = A.getRowMap()->getLocalMap();
    auto lclColMap = A.getColMap()->getLocalMap();

    Kokkos::parallel_scan(
        "filterMatrix::rowptr_construction", Kokkos::RangePolicy<exec_space>(0, lclA.numRows()), KOKKOS_LAMBDA(const local_ordinal_type rlid, local_ordinal_type& nnz, const bool update) {
          auto row  = lclA.rowConst(rlid);
          auto rgid = lclRowMap.getGlobalElement(rlid);
          for (local_ordinal_type k = 0; k < row.length; ++k) {
            auto clid = row.colidx(k);
            auto cgid = lclRowMap.getGlobalElement(clid);
            auto val  = row.value(k);
            if (filter(rgid, cgid, val)) {
              ++nnz;
            }
          }
          if (update && (rlid + 2 < lclA.numRows() + 1))
            rowptr(rlid + 2) = nnz;
        },
        NNZ);

    auto colidx = colidx_type(Kokkos::ViewAllocateWithoutInitializing("filtered_colidx"), NNZ);

    Kokkos::parallel_for(
        "filterMatrix::graph_fill", Kokkos::RangePolicy<exec_space>(0, lclA.numRows()), KOKKOS_LAMBDA(const local_ordinal_type rlid) {
          auto row  = lclA.rowConst(rlid);
          auto rgid = lclRowMap.getGlobalElement(rlid);
          for (local_ordinal_type k = 0; k < row.length; ++k) {
            auto clid = row.colidx(k);
            auto cgid = lclRowMap.getGlobalElement(clid);
            auto val  = row.value(k);
            if (filter(rgid, cgid, val)) {
              colidx(rowptr(rlid + 1)) = clid;
              ++rowptr(rlid + 1);
            }
          }
        });

    newLclA = local_graph_type(colidx, rowptr);
  }

  return Teuchos::rcp(new graph_type(newLclA, A.getRowMap(), A.getColMap(), A.getDomainMap(), A.getRangeMap()));
}

/// \brief Constructs a graph consisting only of the entries for which the filter function evaluates to true.
///
/// This applies a filter function that takes as arguments the global row and column ids and the value of the graph entry.
///
/// \param A [in] The source graph that the filter will be applied to.
/// \param filter [in] A callable with signature (global_ordinal_type, global_ordinal_type) -> bool.
/// \return The filtered graph.
template <class graph_type, class filter_type>
Teuchos::RCP<graph_type> applyGraphFilter_GID(const graph_type& G, const filter_type& filter) {
  using local_ordinal_type = typename graph_type::local_ordinal_type;
  using node_type          = typename graph_type::node_type;
  using local_graph_type   = typename graph_type::local_graph_device_type;
  using rowptr_type        = typename local_graph_type::row_map_type::non_const_type;
  using colidx_type        = typename local_graph_type::entries_type::non_const_type;
  using exec_space         = typename node_type::execution_space;

  local_graph_type newLclG;
  {
    rowptr_type rowptr;
    auto lclG = G.getLocalMatrixDevice();

    rowptr = rowptr_type("filtered_rowptr", lclG.numRows() + 1);
    local_ordinal_type NNZ;

    auto lclRowMap = G.getRowMap()->getLocalMap();
    auto lclColMap = G.getColMap()->getLocalMap();

    Kokkos::parallel_scan(
        "filterGraph::rowptr_construction", Kokkos::RangePolicy<exec_space>(0, lclG.numRows()), KOKKOS_LAMBDA(const local_ordinal_type rlid, local_ordinal_type& nnz, const bool update) {
          auto row  = lclG.rowConst(rlid);
          auto rgid = lclRowMap.getGlobalElement(rlid);
          for (local_ordinal_type k = 0; k < row.length; ++k) {
            auto clid = row.colidx(k);
            auto cgid = lclRowMap.getGlobalElement(clid);
            if (filter(rgid, cgid)) {
              ++nnz;
            }
          }
          if (update && (rlid + 2 < lclG.numRows() + 1))
            rowptr(rlid + 2) = nnz;
        },
        NNZ);

    auto colidx = colidx_type(Kokkos::ViewAllocateWithoutInitializing("filtered_colidx"), NNZ);

    Kokkos::parallel_for(
        "filterGraph::graph_fill", Kokkos::RangePolicy<exec_space>(0, lclG.numRows()), KOKKOS_LAMBDA(const local_ordinal_type rlid) {
          auto row  = lclG.rowConst(rlid);
          auto rgid = lclRowMap.getGlobalElement(rlid);
          for (local_ordinal_type k = 0; k < row.length; ++k) {
            auto clid = row.colidx(k);
            auto cgid = lclRowMap.getGlobalElement(clid);
            if (filter(rgid, cgid)) {
              colidx(rowptr(rlid + 1)) = clid;
              ++rowptr(rlid + 1);
            }
          }
        });

    newLclG = local_graph_type(colidx, rowptr);
  }

  return Teuchos::rcp(new graph_type(newLclG, G.getRowMap(), G.getColMap(), G.getDomainMap(), G.getRangeMap()));
}

template <class scalar_type, class local_ordinal_type>
struct AbsoluteMagnitudeFilter {
  using ATS              = KokkosKernels::ArithTraits<scalar_type>;
  using impl_scalar_type = typename ATS::val_type;
  using implATS          = KokkosKernels::ArithTraits<impl_scalar_type>;

  typename implATS::magnitudeType tol_;

  AbsoluteMagnitudeFilter(typename implATS::magnitudeType tol)
    : tol_(tol) {}

  KOKKOS_INLINE_FUNCTION
  bool operator()(impl_scalar_type val) const {
    return (implATS::magnitude(val) > tol_);
  }
};

}  // namespace Tpetra
#endif
