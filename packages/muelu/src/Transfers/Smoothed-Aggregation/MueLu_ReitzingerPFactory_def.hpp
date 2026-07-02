// @HEADER
// *****************************************************************************
//        MueLu: A package for multigrid based preconditioning
//
// Copyright 2012 NTESS and the MueLu contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef MUELU_REITZINGERPFACTORY_DEF_HPP
#define MUELU_REITZINGERPFACTORY_DEF_HPP

#include <Xpetra_MapFactory.hpp>
#include <Xpetra_Map.hpp>
#include <Xpetra_CrsMatrix.hpp>
#include <Xpetra_Matrix.hpp>
#include <Xpetra_MatrixMatrix.hpp>
#include <Xpetra_MultiVector.hpp>
#include <Xpetra_VectorFactory.hpp>
#include <Xpetra_CrsMatrixWrap.hpp>
// #include <Xpetra_IO.hpp>

#include "Kokkos_UnorderedMap.hpp"
#include "MueLu_ReitzingerPFactory_decl.hpp"

#include <Teuchos_ScalarTraits.hpp>

#include "MueLu_MasterList.hpp"
#include "MueLu_Monitor.hpp"
#include "MueLu_Utilities.hpp"

#include "MueLu_Behavior.hpp"
#include "Teuchos_Assert.hpp"
#include "Teuchos_TestForException.hpp"

namespace MueLu {

template <class LocalOrdinal, int dim>
struct my_tuple_type {
  LocalOrdinal data[dim];
  static constexpr int my_dim = dim;
};

template <class tuple_type, class RowType>
KOKKOS_FUNCTION
    tuple_type
    makeTuple(RowType& row) {
  tuple_type t;
  for (int k = 0; k < tuple_type::my_dim; ++k)
    t.data[k] = row.colidx(k);
  return t;
}

template <class rowptr_type, class colidx_type, class values_type,
          class map_type, class local_matrix_type, class local_ordinal_type>
class RegularCoarseningFunctor {
  using tuple_type = typename map_type::key_type;

 public:
  RegularCoarseningFunctor(map_type& map_, local_matrix_type& mat_, local_ordinal_type sizeCoarsenedEntity_) {
    map                 = map_;
    mat                 = mat_;
    sizeCoarsenedEntity = sizeCoarsenedEntity_;
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const local_ordinal_type rlid, local_ordinal_type& ne) const {
    auto row = mat.rowConst(rlid);
    if (row.length == sizeCoarsenedEntity) {
      auto entity = makeTuple<tuple_type>(row);
      if (!map.exists(entity)) {
        map.insert(entity, rlid);
        ++ne;
      }
    }
  }

  void setViews(rowptr_type& rowptr_, colidx_type& colidx_, values_type& values_) {
    rowptr = rowptr_;
    colidx = colidx_;
    values = values_;
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const uint32_t i, local_ordinal_type& ne, const bool update) const {
    if (map.valid_at(i)) {
      if (!update) {
        // First pass: figure out offsets for entries.
        ++ne;
      } else {
        // Second pass: enter entries
        // initialize
        if (ne == 0)
          rowptr(0) = 0;

        auto rlid = map.value_at(i);
        auto row  = mat.rowConst(rlid);
        for (int k = 0; k < sizeCoarsenedEntity; ++k) {
          colidx(sizeCoarsenedEntity * ne + k) = row.colidx(k);
          values(sizeCoarsenedEntity * ne + k) = row.value(k);
        }
        ++ne;
        rowptr(ne) = sizeCoarsenedEntity * ne;
      }
    }
  }

 private:
  map_type map;
  local_matrix_type mat;
  local_ordinal_type sizeCoarsenedEntity;

  rowptr_type rowptr;
  colidx_type colidx;
  values_type values;
};

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<const ParameterList> ReitzingerPFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::GetValidParameterList() const {
  RCP<ParameterList> validParamList = rcp(new ParameterList());

#define SET_VALID_ENTRY(name) validParamList->setEntry(name, MasterList::getEntry(name))
  SET_VALID_ENTRY("repartition: enable");
  SET_VALID_ENTRY("repartition: use subcommunicators");
  SET_VALID_ENTRY("tentative: calculate qr");
  SET_VALID_ENTRY("tentative: constant column sums");
#undef SET_VALID_ENTRY

  validParamList->set<RCP<const FactoryBase> >("D0", Teuchos::null, "Generating factory of the matrix D0");
  validParamList->set<RCP<const FactoryBase> >("NodeMatrix", Teuchos::null, "Generating factory of the matrix NodeMatrix");
  validParamList->set<RCP<const FactoryBase> >("Ptent_nodal", Teuchos::null, "Generating factory of the matrix P");

  // Make sure we don't recursively validate options for the matrixmatrix kernels
  ParameterList norecurse;
  norecurse.disableRecursiveValidation();
  validParamList->set<ParameterList>("matrixmatrix: kernel params", norecurse, "MatrixMatrix kernel parameters");

  return validParamList;
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void ReitzingerPFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::DeclareInput(Level& fineLevel, Level& coarseLevel) const {
  Input(fineLevel, "D0");
  Input(coarseLevel, "NodeMatrix");
  Input(coarseLevel, "Ptent_nodal");
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void ReitzingerPFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Build(Level& fineLevel, Level& coarseLevel) const {
  return BuildP(fineLevel, coarseLevel);
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void ReitzingerPFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::BuildP(Level& fineLevel, Level& coarseLevel) const {
  FactoryMonitor m(*this, "Build", coarseLevel);

  using XMM               = Xpetra::MatrixMatrix<SC, LO, GO, NO>;
  using local_matrix_type = typename Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::local_matrix_type;
  using rowptr_type       = typename local_matrix_type::row_map_type::non_const_type;
  using colidx_type       = typename local_matrix_type::index_type::non_const_type;
  using values_type       = typename local_matrix_type::values_type::non_const_type;

  using impl_scalar_type = typename Matrix::impl_scalar_type;
  using ATS              = KokkosKernels::ArithTraits<impl_scalar_type>;
  using mag_type         = typename KokkosKernels::ArithTraits<impl_scalar_type>::magnitudeType;
  using magATS           = KokkosKernels::ArithTraits<mag_type>;

  using execution_space = typename Node::execution_space;
  using memory_space    = typename Node::memory_space;

  const auto one_Scalar       = Teuchos::ScalarTraits<Scalar>::one();
  const auto one_impl_scalar  = ATS::one();
  const auto one_LO           = KokkosKernels::ArithTraits<LocalOrdinal>::one();
  const auto one_mag          = magATS::one();
  const auto eps_mag          = magATS::epsilon();
  const auto INVALID_GO       = Teuchos::OrdinalTraits<GlobalOrdinal>::invalid();
  const auto REGULAR_ENTITY   = KokkosKernels::ArithTraits<LocalOrdinal>::zero();
  const auto DIRICHLET_ENTITY = KokkosKernels::ArithTraits<LocalOrdinal>::one();

  // Using a prolongator P and the discrete differential operator matrix D, this factory constructs
  // a coarse discrete differential operator matrix Dc and an prolongator Pnext such that the commuting
  // relationship
  //
  //  D * P = Pnext * Dc
  //
  // holds.

  // We call the rows of D the "fine entities" and the rows of Dc the "coarse entities".
  // We call the columns of D the "fine preceding entities" and the columns of Dc the "coarse preceding entities".
  // When D is the discrete gradient, the entities are edges and the preceding entities are nodes (or nodal aggregates).

  // The construction of the coarse discrete differential operator Dc works as follows.
  // We form
  //
  //     T := D*P
  //
  // and get all unique sets of column indices of T of size sizeCoarsenedEntity, where sizeCoarsenedEntity is the maximum
  // number of entries per row of D. By "unique" we mean that we do not care about the ordering of the column indices.
  // We then form coarse entities (i.e. edges, faces or volumes) out of the unique index sets.

  // We also perform detection of boundary condtions and add additional entities to the coarse discrete operator.
  // We detect all entities in D that connect to fewer than sizeCoarsenedEntity preceding entities.

  Teuchos::FancyOStream& out0 = GetBlackHole();
  const ParameterList& pL     = GetParameterList();

  bool update_communicators = pL.get<bool>("repartition: enable") && pL.get<bool>("repartition: use subcommunicators");

  const std::string D_name = "D0";

  RCP<Matrix> D = Get<RCP<Matrix> >(fineLevel, D_name);
  RCP<Matrix> P = Get<RCP<Matrix> >(coarseLevel, "Ptent_nodal");

  // This needs to be an Operator because if NodeMatrix gets repartitioned away, we get an Operator on the level
  RCP<Operator> CoarseNodeMatrix = Get<RCP<Operator> >(coarseLevel, "NodeMatrix");

  // Matrix matrix params
  RCP<ParameterList> mm_params = rcp(new ParameterList);
  if (pL.isSublist("matrixmatrix: kernel params"))
    mm_params->sublist("matrixmatrix: kernel params") = pL.sublist("matrixmatrix: kernel params");

  if (Behavior::debug()) {  // Check that P is piecewise constant
    auto vec_ones = VectorFactory::Build(P->getDomainMap(), false);
    vec_ones->putScalar(one_Scalar);
    auto vec_rowsums = VectorFactory::Build(P->getRangeMap(), false);
    P->apply(*vec_ones, *vec_rowsums, Teuchos::NO_TRANS);

    auto lclP       = P->getLocalMatrixDevice();
    auto lclRowSums = vec_rowsums->getLocalViewDevice(Tpetra::Access::ReadOnly);

    bool all_entries_ok = true;
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<execution_space>(0, lclP.numRows()), KOKKOS_LAMBDA(const LocalOrdinal rlid, bool& entries_ok) {
      // rowsums are 1
      entries_ok = entries_ok && (ATS::magnitude(lclRowSums(rlid, 0) - one_impl_scalar) < eps_mag);

      // all nonzero entries are 1
      auto row = lclP.rowConst(rlid);
      for (LocalOrdinal k = 0; k < row.length; ++k) {
        entries_ok = entries_ok && (ATS::magnitude(row.value(k)-one_impl_scalar) < eps_mag);

      } }, Kokkos::LAnd<bool>(all_entries_ok));

    TEUCHOS_TEST_FOR_EXCEPTION(!all_entries_ok, std::runtime_error, "The prolongator needs to be piecewise constant and all entries need to be 1.");
  }

  RCP<Matrix> D_P;
  RCP<Matrix> Dc;
  LocalOrdinal numCoarseEntities          = 0;
  LocalOrdinal numCoarseRegularEntities   = 0;
  LocalOrdinal numCoarseDirichletEntities = 0;
  auto isDirichletFineEntity              = Xpetra::VectorFactory<LocalOrdinal, LocalOrdinal, GlobalOrdinal, Node>::Build(D->getRangeMap(), false);
  auto numFineEntities                    = D->getRangeMap()->getLocalNumElements();
  auto sizeCoarsenedEntity                = (LocalOrdinal)D->getLocalMaxNumRowEntries();
  {
    // Construct D*P
    RCP<Matrix> dummy;
    D_P = XMM::Multiply(*D, false, *P, false, dummy, GetOStream(Runtime0), true, true);

    auto rowMap    = D_P->getRowMap();
    auto colMap    = D_P->getColMap();
    auto lclRowMap = rowMap->getLocalMap();
    auto lclColMap = colMap->getLocalMap();

    // Mark as singleParents any D edges with only one node (so these are
    // edges that connect an interior node with a Dirichlet node).
    // isDirichletFineEdge is 1 for edges with a single endpoint and 0 otherwise.

    // Flag fine Dirichlet entities
    {
      isDirichletFineEntity->putScalar(DIRICHLET_ENTITY);
      auto lcl_isDirichletFineEntity = isDirichletFineEntity->getLocalViewDevice(Tpetra::Access::ReadWrite);
      auto lcl_D                     = D->getLocalMatrixDevice();
      Kokkos::parallel_for(
          Kokkos::RangePolicy<execution_space>(0, lcl_isDirichletFineEntity.extent(0)), KOKKOS_LAMBDA(const LocalOrdinal i) {
            auto row = lcl_D.rowConst(i);
            if (row.length == sizeCoarsenedEntity) {
              // This is a regular entity.
              lcl_isDirichletFineEntity(i, 0) = REGULAR_ENTITY;
            } else {
              // This is a Dirichlet entity.
              lcl_isDirichletFineEntity(i, 0) = DIRICHLET_ENTITY;
            }
          });
    }

    // Count the number of fine Dirichlet entities that are connected to every coarse preceding entity via
    // the graph of D*P.
    auto numberConnectedFineDirichletEdgesToCoarseNode = Xpetra::VectorFactory<LocalOrdinal, LocalOrdinal, GlobalOrdinal, Node>::Build(D_P->getDomainMap(), false);
    {
      using LOMatrix = Tpetra::CrsMatrix<LocalOrdinal, LocalOrdinal, GlobalOrdinal, Node>;
      auto abs_D_P   = LOMatrix(toTpetra(D_P->getCrsGraph()));
      abs_D_P.fillComplete(toTpetra(D_P->getDomainMap()), toTpetra(D_P->getRangeMap()));
      abs_D_P.setAllToScalar(one_LO);
      abs_D_P.apply(*toTpetra(isDirichletFineEntity), *toTpetra(numberConnectedFineDirichletEdgesToCoarseNode), Teuchos::TRANS);
    }

    // Count local Dirichlet coarse entities
    {
      auto lcl_numberConnectedFineDirichletEdgesToCoarseNode = numberConnectedFineDirichletEdgesToCoarseNode->getLocalViewDevice(Tpetra::Access::ReadOnly);

      Kokkos::parallel_reduce(
          Kokkos::RangePolicy<execution_space>(0, lcl_numberConnectedFineDirichletEdgesToCoarseNode.extent(0)),
          KOKKOS_LAMBDA(const LocalOrdinal i, LocalOrdinal& ne) {
            if (ATS::magnitude(lcl_numberConnectedFineDirichletEdgesToCoarseNode(i, 0)) > eps_mag) {
              ++ne;
            }
          },
          numCoarseDirichletEntities);
    }

    // Make sure that rows with identical column indices are sorted in the same way.
    TEUCHOS_ASSERT(toTpetra(D_P->getCrsGraph())->isSorted());
    auto lcl_D_P = D_P->getLocalMatrixDevice();

    // Count up how many coarse regular entities we are creating.
    auto numLocalRows = lcl_D_P.numRows();

    LocalOrdinal nnzRegular;
    LocalOrdinal nnzDirichlet;
    LocalOrdinal nnz;
    rowptr_type rowptr;
    colidx_type colidx;
    values_type values;
    if (sizeCoarsenedEntity == 2) {
      using tuple_type    = my_tuple_type<LocalOrdinal, 2>;
      using set_type      = Kokkos::UnorderedMap<tuple_type, LocalOrdinal, execution_space>;
      set_type Dc_entries = set_type(numLocalRows);

      RegularCoarseningFunctor<rowptr_type, colidx_type, values_type, set_type, decltype(lcl_D_P), LocalOrdinal> functor(Dc_entries, lcl_D_P, sizeCoarsenedEntity);

      Kokkos::parallel_reduce(Kokkos::RangePolicy<execution_space>(0, numLocalRows), functor, numCoarseRegularEntities);

      numCoarseEntities = numCoarseRegularEntities + numCoarseDirichletEntities;
      rowptr            = rowptr_type(Kokkos::ViewAllocateWithoutInitializing("rowptr Dc"), numCoarseEntities + 1);
      // sizeCoarsenedEntity entries per regular entity, 1 entry per Dirichlet edge
      nnzRegular   = sizeCoarsenedEntity * numCoarseRegularEntities;
      nnzDirichlet = numCoarseDirichletEntities;
      nnz          = nnzRegular + nnzDirichlet;
      colidx       = colidx_type(Kokkos::ViewAllocateWithoutInitializing("colidx Dc"), nnz);
      values       = values_type(Kokkos::ViewAllocateWithoutInitializing("values Dc"), nnz);

      functor.setViews(rowptr, colidx, values);

      // Fill regular entities
      Kokkos::parallel_scan(Kokkos::RangePolicy<execution_space>(0, Dc_entries.capacity()), functor);
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Not implemented");
    }

    // Fill Dirichlet entities
    // Create one coarse Dirichlet edge for every nodal aggregate that is connected to at least one fine Dirichlet edge.
    {
      auto lcl_numberConnectedFineDirichletEdgesToCoarseNode = numberConnectedFineDirichletEdgesToCoarseNode->getLocalViewDevice(Tpetra::Access::ReadOnly);
      Kokkos::parallel_scan(
          Kokkos::RangePolicy<execution_space>(0, lcl_numberConnectedFineDirichletEdgesToCoarseNode.extent(0)),
          KOKKOS_LAMBDA(const LocalOrdinal agg_lid, LocalOrdinal& ne, const bool update) {
            if (ATS::magnitude(lcl_numberConnectedFineDirichletEdgesToCoarseNode(agg_lid, 0)) > eps_mag) {
              if (!update) {
                // First pass: figure out offsets
                ++ne;
              } else {
                // Second pass: fill
                colidx(nnzRegular + ne) = agg_lid;
                values(nnzRegular + ne) = one_impl_scalar;
                ++ne;
                rowptr(numCoarseRegularEntities + ne) = nnzRegular + ne;
              }
            }
          });
    }

    auto Dc_rowmap = MapFactory::Build(rowMap->lib(), INVALID_GO, numCoarseEntities, 0, rowMap->getComm());
    auto lclDc     = local_matrix_type("Dc", numCoarseEntities, colMap->getLocalNumElements(), nnz, values, rowptr, colidx);

    // Construct distributed matrix
    Dc = MatrixFactory::Build(lclDc, Dc_rowmap, colMap, P->getDomainMap(), Dc_rowmap);

    if (IsPrint(Statistics0)) {
      LocalOrdinal numGlobalRegularEntities;
      LocalOrdinal numGlobalDirichletEntities;
      MueLu_sumAll(rowMap->getComm(), numCoarseRegularEntities, numGlobalRegularEntities);
      MueLu_sumAll(rowMap->getComm(), numCoarseDirichletEntities, numGlobalDirichletEntities);
      GetOStream(Statistics0) << "regular entities: " << numGlobalRegularEntities << ", Dirichlet entities: " << numGlobalDirichletEntities << std::endl;
    }
  }

  const bool needToBuildPnext = (coarseLevel.IsRequested("P", this) ||
                                 coarseLevel.IsRequested("Ptent", this));
  RCP<Matrix> Pnext;
  if (needToBuildPnext) {
    // We construct
    //                      D P Dc^T,
    // but then we only keep entries with value +-sizeCoarsenedEntity and divide by sizeCoarsenedEntity.
    // Here sizeCoarsenedEntity is the maximum number of entries per row in D. In the usual construction of a
    // prolongator for H(curl) we have that sizeCoarsenedEntity=2.

    RCP<Matrix> D_P_DcT;
    {
      SubFactoryMonitor m2(*this, "Generate Pnext (pre-fix)", coarseLevel);
      RCP<Matrix> dummy;
      D_P_DcT = XMM::Multiply(*D_P, false, *Dc, true, dummy, out0, true, true, "(D*P)*Dc^T");
    }

    {
      auto lcl_D_P                 = D_P->getLocalMatrixDevice();
      auto lcl_D_P_DcT             = D_P_DcT->getLocalMatrixDevice();
      auto lcl_isDirichletFineEdge = isDirichletFineEntity->getLocalViewDevice(Tpetra::Access::ReadOnly);

      auto lcl_colmap_D_P_DcT = D_P_DcT->getColMap()->getLocalMap();

      const auto normalization = one_impl_scalar / ((Scalar)sizeCoarsenedEntity);

      // overallocate by 1 to allow for easier counting
      rowptr_type Pnext_rowptr("Pnext_rowptr", numFineEntities + 2);

      // count entries per row
      Kokkos::parallel_for(
          "Pnext_count_entries", Kokkos::RangePolicy<execution_space>(0, numFineEntities), KOKKOS_LAMBDA(const LocalOrdinal fineEntity) {
            if (lcl_isDirichletFineEdge(fineEntity, 0) == REGULAR_ENTITY) {
              // regular fine entity
              auto row = lcl_D_P_DcT.rowConst(fineEntity);
              for (int k = 0; k < row.length; ++k) {
                auto val = row.value(k);
                // filter out entries that are not +-sizeCoarsenedEntity
                if (ATS::magnitude((ATS::magnitude(val) - sizeCoarsenedEntity)) < eps_mag) {
                  // add entry (fineEntity, clid) -> val * normalization.
                  ++Pnext_rowptr(fineEntity + 2);
                }
              }
            } else {
              // Dirichlet interior fine entity
              ++Pnext_rowptr(fineEntity + 2);
            }
          });

      // prefix sum
      LocalOrdinal Pnext_nnz;
      Kokkos::parallel_scan(
          "Pnext_prefix_sum", Kokkos::RangePolicy<execution_space>(0, numFineEntities + 2), KOKKOS_LAMBDA(const LocalOrdinal rlid, LocalOrdinal& nnz, const bool update) {
            nnz += Pnext_rowptr(rlid);
            if (update) {
              Pnext_rowptr(rlid) = nnz;
            }
          },
          Pnext_nnz);

      // allocate view for indices and values
      colidx_type Pnext_colidx("Pnext_colidx", Pnext_nnz);
      values_type Pnext_values("Pnext_values", Pnext_nnz);

      // We build the mapping from coarse nodes lids wrt column map of D_P to coarse edge gids.
      RCP<GOVector> map_coarseNodes_colMap_D_P_to_coarseEdges;
      {
        auto map_coarseEdges_rowMap_Dc_to_coarseEdges = Xpetra::VectorFactory<GlobalOrdinal, LocalOrdinal, GlobalOrdinal, Node>::Build(Dc->getRowMap());
        {
          auto lcl_map_coarseEdges_rowMap_Dc_to_coarseEdges = map_coarseEdges_rowMap_Dc_to_coarseEdges->getLocalViewDevice(Tpetra::Access::OverwriteAll);
          auto lclMap                                       = Dc->getRowMap()->getLocalMap();
          Kokkos::parallel_for(
              Kokkos::RangePolicy<execution_space>(numCoarseRegularEntities, numCoarseEntities), KOKKOS_LAMBDA(const LocalOrdinal coarseEntity) {
                lcl_map_coarseEdges_rowMap_Dc_to_coarseEdges(coarseEntity, 0) = lclMap.getGlobalElement(coarseEntity);
              });
        }
        auto map_coarseNodes_domainMap_D_P_to_coarseEdges = Xpetra::VectorFactory<GlobalOrdinal, LocalOrdinal, GlobalOrdinal, Node>::Build(Dc->getDomainMap());
        {
          // We want to do a transpose apply using Dc on vectors with Scalar=GlobalOrdinal.
          // Something like this could work and would not require any memory allocations, but it requires
          // "convert" to be ETI'd for all possible scalar types.

          // toTpetra(Dc)->template convert<GlobalOrdinal>()->apply(*toTpetra(map_coarseEdges_rowMap_Dc_to_coarseEdges), *toTpetra(map_coarseNodes_domainMap_D_P_to_coarseEdges), Teuchos::TRANS);

          using GOMatrix             = Tpetra::CrsMatrix<GlobalOrdinal, LocalOrdinal, GlobalOrdinal, Node>;
          using go_local_matrix_type = typename GOMatrix::local_matrix_device_type;

          auto lclGraph = Dc->getCrsGraph()->getLocalGraphDevice();
          typename go_local_matrix_type::values_type::non_const_type ones("ones_GlobalOrdinal", Dc->getLocalNumEntries());
          const auto one_GO = KokkosKernels::ArithTraits<typename go_local_matrix_type::values_type::value_type>::one();
          Kokkos::deep_copy(ones, one_GO);

          go_local_matrix_type lclMatrix("Dc_GlobalOrdinal", Dc->getLocalMatrixDevice().numCols(), ones, lclGraph);

          auto Dc_GlobalOrdinal = GOMatrix(lclMatrix, toTpetra(Dc->getRowMap()), toTpetra(Dc->getColMap()), toTpetra(Dc->getDomainMap()), toTpetra(Dc->getRangeMap()));
          Dc_GlobalOrdinal.apply(*toTpetra(map_coarseEdges_rowMap_Dc_to_coarseEdges), *toTpetra(map_coarseNodes_domainMap_D_P_to_coarseEdges), Teuchos::TRANS);
        }

        auto importer = D_P->getCrsGraph()->getImporter();
        if (!importer.is_null()) {
          map_coarseNodes_colMap_D_P_to_coarseEdges = Xpetra::VectorFactory<GlobalOrdinal, LocalOrdinal, GlobalOrdinal, Node>::Build(D_P->getColMap());
          map_coarseNodes_colMap_D_P_to_coarseEdges->doImport(*map_coarseNodes_domainMap_D_P_to_coarseEdges, *importer, Xpetra::INSERT);
        } else {
          map_coarseNodes_colMap_D_P_to_coarseEdges = map_coarseNodes_domainMap_D_P_to_coarseEdges;
        }
      }
      {
        auto lcl_map_coarseNodes_colMap_D_P_to_coarseEdges = map_coarseNodes_colMap_D_P_to_coarseEdges->getLocalViewDevice(Tpetra::Access::ReadOnly);

        // fill
        Kokkos::parallel_for(
            "Pnext_fill", Kokkos::RangePolicy<execution_space>(0, numFineEntities), KOKKOS_LAMBDA(const LocalOrdinal fineEntity_lid) {
              if (lcl_isDirichletFineEdge(fineEntity_lid, 0) == REGULAR_ENTITY) {
                // regular fine entity
                auto row = lcl_D_P_DcT.rowConst(fineEntity_lid);
                for (int k = 0; k < row.length; ++k) {
                  auto val = row.value(k);
                  // filter out entries that are not +-sizeCoarsenedEntity
                  if (ATS::magnitude((ATS::magnitude(val) - sizeCoarsenedEntity)) < eps_mag) {
                    auto clid = row.colidx(k);
                    // add entry (fineEntity_lid, clid) -> val*normalization.
                    auto offset          = Pnext_rowptr(fineEntity_lid + 1);
                    Pnext_colidx(offset) = clid;
                    Pnext_values(offset) = val * normalization;
                    ++Pnext_rowptr(fineEntity_lid + 1);
                  }
                }
              } else {
                // Dirichlet interior fine entity
                // Only one nonzero entry in row of D_P: (fineEntity_lid, coarseNode_lid_D_P) -> val_D_P
                for (auto offset_D_P = lcl_D_P.graph.row_map(fineEntity_lid); offset_D_P < lcl_D_P.graph.row_map(fineEntity_lid + 1); ++offset_D_P) {
                  LocalOrdinal coarseNode_lid_D_P = lcl_D_P.graph.entries(offset_D_P);
                  impl_scalar_type val_D_P        = lcl_D_P.values(offset_D_P);
                  if (ATS::magnitude(val_D_P) > eps_mag) {
                    GlobalOrdinal coarseEntity_gid = lcl_map_coarseNodes_colMap_D_P_to_coarseEdges(coarseNode_lid_D_P, 0);

                    auto coarseEntity_lid_D_P_DcT = lcl_colmap_D_P_DcT.getLocalElement(coarseEntity_gid);

                    // We rely on the fact that all coarse interior edges have been created with value 1.
                    const auto val_Dc = one_impl_scalar;

                    // add entry (fineEntity_lid, coarseEntity) -> val_D_P/val_Dc to next prolongator
                    auto offset_Pnext          = Pnext_rowptr(fineEntity_lid + 1);
                    Pnext_colidx(offset_Pnext) = coarseEntity_lid_D_P_DcT;
                    Pnext_values(offset_Pnext) = val_D_P / val_Dc;
                    ++Pnext_rowptr(fineEntity_lid + 1);
                    break;
                  }
                }
              }
            });
      }
      auto lclPnext = local_matrix_type("Pnext", numFineEntities, D_P_DcT->getColMap()->getLocalNumElements(), Pnext_nnz, Pnext_values, Kokkos::subview(Pnext_rowptr, Kokkos::make_pair((decltype(numFineEntities))0, numFineEntities + 1)), Pnext_colidx);

      // Construct distributed matrix
      Pnext = MatrixFactory::Build(lclPnext, D->getRowMap(), D_P_DcT->getColMap(), Dc->getRangeMap(), D->getRangeMap());
    }

    if (Behavior::debug()) {
      /* Check commuting property */
      CheckCommutingProperty(*Pnext, *Dc, *D, *P);
    }
  }

  /*  If we're repartitioning here, we need to cut down the communicators */
  // NOTE: We need to do this *after* checking the commuting property, since
  // that's going to need to fineLevel's communicators, not the repartitioned ones
  if (update_communicators) {
    // NOTE: We can only do D here.  We have to do Ke_coarse=(Rnext K_fine Pnext) in RebalanceAcFactory
    RCP<const Teuchos::Comm<int> > newComm;
    if (!CoarseNodeMatrix.is_null()) newComm = CoarseNodeMatrix->getDomainMap()->getComm();
    RCP<const Map> newMap = MapFactory::copyMapWithNewComm(Dc->getRowMap(), newComm);
    Dc->removeEmptyProcessesInPlace(newMap);

    // The "in place" still leaves a dummy matrix here.  That needs to go
    if (newMap.is_null()) Dc = Teuchos::null;

    Set(coarseLevel, "InPlaceMap", newMap);
  }

  /* Set output on the level */
  if (coarseLevel.IsRequested("P", this))
    Set(coarseLevel, "P", Pnext);
  if (coarseLevel.IsRequested("Ptent", this))
    Set(coarseLevel, "Ptent", Pnext);

  Set(coarseLevel, D_name, Dc);

  /* This needs to be kept for the smoothers */
  coarseLevel.Set(D_name, Dc, NoFactory::get());
  coarseLevel.AddKeepFlag(D_name, NoFactory::get(), MueLu::Final);
  coarseLevel.RemoveKeepFlag(D_name, NoFactory::get(), MueLu::UserData);

#if 0
  {
    int numProcs = Pnext->getRowMap()->getComm()->getSize();
    char fname[80];

    sprintf(fname, "Pnext_%d_%d.mat", numProcs, fineLevel.GetLevelID());
    Xpetra::IO<SC, LO, GO, NO>::Write(fname, *Pnext);
    sprintf(fname, "P_%d_%d.mat", numProcs, fineLevel.GetLevelID());
    Xpetra::IO<SC, LO, GO, NO>::Write(fname, *Pn);
    if (!D0H.is_null()) {
      sprintf(fname, "Dc_%d_%d.mat", numProcs, fineLevel.GetLevelID());
      Xpetra::IO<SC, LO, GO, NO>::Write(fname, *Dc);
    }
    sprintf(fname, "D_%d_%d.mat", numProcs, fineLevel.GetLevelID());
    Xpetra::IO<SC, LO, GO, NO>::Write(fname, *D);
  }
#endif

}  // end Build

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
typename Teuchos::ScalarTraits<Scalar>::magnitudeType ReitzingerPFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    ComputeCommutingPropertyResidualNorm(const Matrix& Pnext, const Matrix& Dc, const Matrix& D, const Matrix& P, Teuchos::FancyOStream& out) {
  using XMM = MatrixMatrix;
  auto one  = Teuchos::ScalarTraits<SC>::one();

  RCP<Matrix> dummy;
  RCP<Matrix> left  = XMM::Multiply(Pnext, false, Dc, false, dummy, out);
  RCP<Matrix> right = XMM::Multiply(D, false, P, false, dummy, out);

  RCP<Matrix> summation;
  XMM::TwoMatrixAdd(*left, false, one, *right, false, -one, summation, out);
  summation->fillComplete(left->getDomainMap(), left->getRangeMap());

  auto norm = summation->getFrobeniusNorm();

  return norm;
}  // end ComputeCommutingPropertyResidualNorm

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void ReitzingerPFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    CheckCommutingProperty(const Matrix& Pnext, const Matrix& Dc, const Matrix& D, const Matrix& P) const {
  if (IsPrint(Statistics0)) {
    auto norm = ComputeCommutingPropertyResidualNorm(Pnext, Dc, D, P, GetOStream(Runtime0));
    GetOStream(Statistics0) << "CheckCommutingProperty: || Pnext Dc - D P || = " << norm << std::endl;
  }

}  // end CheckCommutingProperty

}  // namespace MueLu

#define MUELU_REITZINGERPFACTORY_SHORT
#endif  // MUELU_REITZINGERPFACTORY_DEF_HPP
