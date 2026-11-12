// @HEADER
// *****************************************************************************
//        MueLu: A package for multigrid based preconditioning
//
// Copyright 2012 NTESS and the MueLu contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef MUELU_EDGEPROLONGATORPATTERNFACTORY_DEF_HPP
#define MUELU_EDGEPROLONGATORPATTERNFACTORY_DEF_HPP

#include <Xpetra_Matrix.hpp>
#include <Xpetra_MatrixMatrix.hpp>

#include "Kokkos_Abort.hpp"
#include "Kokkos_Printf.hpp"
#include "MueLu_EdgeProlongatorPatternFactory_decl.hpp"

#include "MueLu_Monitor.hpp"
#include "Teuchos_Assert.hpp"
#include "Teuchos_OrdinalTraits.hpp"
#include "Teuchos_ScalarTraits.hpp"
#include "MueLu_Behavior.hpp"

namespace MueLu {

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<const ParameterList> EdgeProlongatorPatternFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::GetValidParameterList() const {
  RCP<ParameterList> validParamList = rcp(new ParameterList());

  validParamList->set<RCP<const FactoryBase> >("FineD0", Teuchos::null, "Generating factory for the fine discrete gradient");
  validParamList->set<RCP<const FactoryBase> >("CoarseD0", Teuchos::null, "Generating factory for the coarse discrete gradient");
  validParamList->set<RCP<const FactoryBase> >("PnodalEmin", Teuchos::null, "Generating factory for the nodal prolongator");

  return validParamList;
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void EdgeProlongatorPatternFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::DeclareInput(Level& fineLevel, Level& coarseLevel) const {
  Input(fineLevel, "D0", "FineD0");
  Input(coarseLevel, "D0", "CoarseD0");
  Input(coarseLevel, "PnodalEmin");
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void EdgeProlongatorPatternFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Build(Level& fineLevel, Level& coarseLevel) const {
  FactoryMonitor m(*this, "EdgeProlongatorPattern", coarseLevel);

  auto D  = Get<RCP<Matrix> >(fineLevel, "D0", "FineD0");
  auto Dc = Get<RCP<Matrix> >(coarseLevel, "D0", "CoarseD0");
  auto Pn = Get<RCP<Matrix> >(coarseLevel, "PnodalEmin");

  const auto one     = Teuchos::ScalarTraits<Scalar>::one();
  const auto invalid = Teuchos::OrdinalTraits<Xpetra::global_size_t>::invalid();

  // |FineD| * |Pnodal| * |CoarseD^T|

  RCP<Matrix> absD_absPn_absDcT;
  RCP<Matrix> absDc;
  {
    SubFactoryMonitor m2(*this, "Matrix manipulations", coarseLevel);

    auto absD = MatrixFactory::BuildCopy(D);
    absD->setAllToScalar(one);

    auto absPn = MatrixFactory::BuildCopy(Pn);
    absPn->setAllToScalar(one);

    RCP<Matrix> absD_absPn = MatrixMatrix::Multiply(*absD, false, *absPn, false, GetOStream(Statistics2), true, true);
    absD_absPn->setAllToScalar(one);

    // If we rebalanced then Dc lives on a smaller communicator than D.
    // Since we need to perform matrix-matrix multiplications with Dc, we construct a version of it that lives on the same communicator.
    auto comm = absD_absPn->getRowMap()->getComm();
    if (Dc.is_null() || Dc->getRowMap()->getComm()->getSize() < comm->getSize()) {
      auto lib = absD_absPn->getRowMap()->lib();
      if (Dc.is_null()) {
        Kokkos::View<GlobalOrdinal*, typename Node::memory_space> dummy("", 0);
        auto big_coarse_nodal_map    = MapFactory::Build(lib, invalid, dummy, 0, comm);
        auto big_coarse_edge_map     = MapFactory::Build(lib, invalid, dummy, 0, comm);
        auto big_coarse_nodal_colmap = MapFactory::Build(lib, invalid, dummy, 0, comm);

        typename Matrix::local_matrix_device_type dummyLocalMatrix;
        Dc = MatrixFactory::Build(dummyLocalMatrix, big_coarse_edge_map, big_coarse_nodal_colmap, big_coarse_nodal_map, big_coarse_edge_map);

      } else {
        auto big_coarse_nodal_map    = MapFactory::Build(lib, invalid, Dc->getDomainMap()->getMyGlobalIndicesDevice(), 0, comm);
        auto big_coarse_edge_map     = MapFactory::Build(lib, invalid, Dc->getRangeMap()->getMyGlobalIndicesDevice(), 0, comm);
        auto big_coarse_nodal_colmap = MapFactory::Build(lib, invalid, Dc->getColMap()->getMyGlobalIndicesDevice(), 0, comm);

        Dc = MatrixFactory::Build(Dc->getLocalMatrixDevice(), big_coarse_edge_map, big_coarse_nodal_colmap, big_coarse_nodal_map, big_coarse_edge_map);
      }
    }
    absDc = MatrixFactory::BuildCopy(Dc);
    absDc->setAllToScalar(one);

    absD_absPn_absDcT = MatrixMatrix::Multiply(*absD_absPn, false, *absDc, true, GetOStream(Statistics2), true, true);
  }

  RCP<Matrix> filtered;
  {
    SubFactoryMonitor m2(*this, "Filtering", coarseLevel);
    using ATS           = KokkosKernels::ArithTraits<typename Matrix::impl_scalar_type>;
    using magnitudeType = typename ATS::magnitudeType;
    using magATS        = KokkosKernels::ArithTraits<magnitudeType>;
    auto eps            = magATS::epsilon();

    RCP<MultiVector> oneVec = MultiVectorFactory::Build(absDc->getDomainMap(), 1);
    oneVec->putScalar(one);
    RCP<MultiVector> singleParent = MultiVectorFactory::Build(absDc->getRowMap(), 1);
    absDc->apply(*oneVec, *singleParent, Teuchos::NO_TRANS);
    // ghost singleParent
    RCP<MultiVector> singleParentGhosted;
    auto importer = absD_absPn_absDcT->getCrsGraph()->getImporter();
    if (importer.is_null()) {
      singleParentGhosted = singleParent;
    } else {
      singleParentGhosted = MultiVectorFactory::Build(importer->getTargetMap(), 1);
      singleParentGhosted->doImport(*singleParent, *importer, Xpetra::INSERT);
    }

    auto lclSingleParent = singleParentGhosted->getLocalViewDevice(Tpetra::Access::ReadOnly);

    // Filter matrix using criterion
    filtered = Xpetra::applyFilter_LID(
        absD_absPn_absDcT,
        KOKKOS_LAMBDA(const LocalOrdinal /*row*/,
                      const LocalOrdinal col,
                      const typename Matrix::impl_scalar_type val) {
          return ((ATS::magnitude(val - 2.0) < eps) || ((lclSingleParent(col, 0) == 1.0) && (ATS::magnitude(val - 1.0) < eps)));
        });

    if (IsPrint(Statistics1)) {
      auto numEntriesBeforeFiltering = absD_absPn_absDcT->getGlobalNumEntries();
      auto numEntriesAfterFiltering  = filtered->getGlobalNumEntries();
      GetOStream(Statistics1) << "Number of kept entries in filtered pattern for P: " << numEntriesAfterFiltering << "/" << numEntriesBeforeFiltering << std::endl;
    }
  }

  auto Ppattern = filtered->getCrsGraph();

  if (Behavior::debug()) {
    CheckPattern(Dc, Ppattern);
  }

  Set(coarseLevel, "Ppattern", Ppattern);
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
bool EdgeProlongatorPatternFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::CheckPattern(RCP<Matrix>& Dc, RCP<const CrsGraph>& Ppattern) const {
  // Ppattern is the sparsity pattern for all admissible edge prolongators Pe.
  // The constraints on Pe are
  //  Pe*Dc = D*Pn.
  // The vectorized form of the equation is
  //  X*vec(P) = vec(D*Pn).
  // A solution candidate that we want to use is
  //  vec(P) := X^T * (X*X^T)^\dagger * vec(D*Pn).
  // X*X^T is a block diagonal matrix with blocks corresponding to rows of Ppattern*Dc (i.e. fine edges).
  // The block sizes correspond to the number of nonzero entries in each row of Ppattern*Dc (i.e. a set of coarse nodes).
  // On such a diagonal block, X*X^T corresponds to the restriction of Dc^T*Dc (a Laplacian-like operator) to the set of coarse nodes.
  // Generally, such a subblock is singular with the constant vector being in the nullspace.
  // What we do not want is that the nullspace has dimension > 1.
  // We check if the subblock has more than one connected component.
  // If we find more than one component for a fine edge e then we will need to make the pattern Ppattern in row e denser.
  // A denser row e implies more coarse nodes in Ppattern*Dc.

  using execution_space = typename Node::execution_space;
  // using memory_space    = typename Node::memory_space;
  using range_type = Kokkos::RangePolicy<LocalOrdinal, execution_space>;

  // Compute graph of Ppattern*Dc
  RCP<Matrix> absPe_absDc;
  {
    const auto one = Teuchos::ScalarTraits<Scalar>::one();
    auto absDc     = MatrixFactory::BuildCopy(Dc);
    absDc->setAllToScalar(one);

    auto absPe = MatrixFactory::Build(Ppattern);
    absPe->setAllToScalar(one);
    absPe->fillComplete();

    absPe_absDc = MatrixMatrix::Multiply(*absPe, false, *absDc, false, GetOStream(Statistics2), true, true);
  }

  // Dct_Dc ghosted to column map of absPe_absDc
  RCP<Matrix> ghosted_Dct_Dc;
  {
    // Compute Dc^T * Dc
    RCP<Matrix> Dct_Dc = MatrixMatrix::Multiply(*Dc, true, *Dc, false, GetOStream(Statistics2), true, true);

    auto importer = absPe_absDc->getCrsGraph()->getImporter();

    // ghost Dc^T * Dc
    if (!importer.is_null())
      ghosted_Dct_Dc = MatrixFactory::Build(Dct_Dc, *importer);
    else
      ghosted_Dct_Dc = Dct_Dc;
  }

  // Check that Dc^T*Dc has a single component on the column entries of each row of Ppattern*Dc.
  LocalOrdinal numBadFineEdges = 0;
  {
    auto lcl_Dct_Dc        = ghosted_Dct_Dc->getLocalMatrixDevice();
    auto lcl_absPe_absDc   = absPe_absDc->getLocalMatrixDevice();
    auto constraints_range = range_type(0, lcl_absPe_absDc.numRows());

    auto reachable = Kokkos::View<LocalOrdinal*>(Kokkos::ViewAllocateWithoutInitializing("reachable"), lcl_absPe_absDc.nnz());

    auto lcl_rowmap_Dct_Dc      = ghosted_Dct_Dc->getRowMap()->getLocalMap();
    auto lcl_colmap_Dct_Dc      = ghosted_Dct_Dc->getColMap()->getLocalMap();
    auto lcl_rowmap_absPe_absDc = absPe_absDc->getRowMap()->getLocalMap();
    auto lcl_colmap_absPe_absDc = absPe_absDc->getColMap()->getLocalMap();

    const auto LO_INVALID = Teuchos::OrdinalTraits<LocalOrdinal>::invalid();

    Kokkos::parallel_reduce(
        constraints_range, KOKKOS_LAMBDA(const LocalOrdinal m, LocalOrdinal& myNumBadFineEdges) {
          // get row of graph of Pe*Dc
          auto row_Pe_Dc = lcl_absPe_absDc.rowConst(m);
          if (row_Pe_Dc.length > 0) {
            auto offset    = lcl_absPe_absDc.graph.row_map(m);
            auto offsetEnd = lcl_absPe_absDc.graph.row_map(m + 1);
            // Create initial set of coarse nodes with an entry in row_Pe_Dc as single entry.
            {
              // Grab an entry (m, n) of Pe_Dc
              auto n_clid = row_Pe_Dc.colidx(0);
              auto n_gid  = lcl_colmap_absPe_absDc.getGlobalElement(n_clid);
              auto n_rlid = lcl_rowmap_Dct_Dc.getLocalElement(n_gid);
              TEUCHOS_ASSERT(n_rlid != LO_INVALID);
              reachable(offset) = n_rlid;
            }
            LocalOrdinal previouslyNumReachable = 0;
            LocalOrdinal numReachable           = 1;

            // continue looping until we find no connected entries anymore
            while (previouslyNumReachable < numReachable) {
              LocalOrdinal newlyFound = 0;
              // loop over current index set of coarse nodes
              for (LocalOrdinal l = 0; l < numReachable; ++l) {
                auto i_rlid     = reachable(offset + l);
                auto row_Dct_Dc = lcl_Dct_Dc.rowConst(i_rlid);
                // loop over entries (i, j) in row i of Dct_Dc
                for (int kk = 0; kk < row_Dct_Dc.length; ++kk) {
                  auto j_clid = row_Dct_Dc.colidx(kk);
                  auto j_gid  = lcl_colmap_Dct_Dc.getGlobalElement(j_clid);
                  auto j_rlid = lcl_rowmap_Dct_Dc.getLocalElement(j_gid);
                  TEUCHOS_ASSERT(j_rlid != LO_INVALID);

                  // Check if j is in row m of Pe_Dc
                  bool inRow = false;
                  for (LocalOrdinal k = 0; k < row_Pe_Dc.length; ++k) {
                    auto n_clid = row_Pe_Dc.colidx(k);
                    auto n_gid  = lcl_colmap_absPe_absDc.getGlobalElement(n_clid);
                    if (n_gid == j_gid) {
                      inRow = true;
                      break;
                    }
                  }
                  if (!inRow)
                    continue;

                  // Check if j is already in the index set. Add it if not.
                  bool found = false;
                  for (LocalOrdinal k = 0; k < numReachable + newlyFound; ++k) {
                    if (reachable(offset + k) == j_rlid) {
                      found = true;
                      break;
                    }
                  }
                  if (!found) {
#ifdef HAVE_MUELU_DEBUG
                    if (numReachable + newlyFound == row_Pe_Dc.length) {
                      // We should never be able to get here.
                      Kokkos::abort("Found too many entries in subblock");
                    }
#endif
                    reachable(offset + numReachable + newlyFound) = j_rlid;
                    ++newlyFound;
                  }
                }
              }
              previouslyNumReachable = numReachable;
              numReachable           = numReachable + newlyFound;
            }

            if (numReachable != row_Pe_Dc.length) {
              ++myNumBadFineEdges;
              Kokkos::printf("multiple components for row_Pe_Dc %i: %i < %i\n", m, numReachable, row_Pe_Dc.length);
            }
          }
        },
        numBadFineEdges);
  }

  GetOStream(Statistics0) << "CheckEdgeProlongatorPattern: number of constraints with multiple components = " << numBadFineEdges << std::endl;

  TEUCHOS_ASSERT(numBadFineEdges == 0);
  return (numBadFineEdges == 0);
}

}  // namespace MueLu

#endif  // MUELU_EDGEPROLONGATORPATTERNFACTORY_DEF_HPP
