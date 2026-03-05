// @HEADER
// *****************************************************************************
//        MueLu: A package for multigrid based preconditioning
//
// Copyright 2012 NTESS and the MueLu contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_DefaultComm.hpp>

#include <MueLu_TestHelpers.hpp>
#include <MueLu_Version.hpp>

#include <Xpetra_MultiVectorFactory.hpp>
#include <Xpetra_MatrixMatrix.hpp>
#include <Xpetra_IO.hpp>

#include <MueLu_AmalgamationFactory.hpp>
#include <MueLu_CoarseMapFactory.hpp>
#include <MueLu_CoalesceDropFactory.hpp>
#include <MueLu_ConstraintFactory.hpp>
#include <MueLu_Constraint.hpp>
#include <MueLu_DenseConstraint.hpp>
#include <MueLu_PatternFactory.hpp>
#include <MueLu_EdgeProlongatorPatternFactory.hpp>
#include <MueLu_EminPFactory.hpp>
#include <MueLu_UncoupledAggregationFactory.hpp>
#include <MueLu_TentativePFactory.hpp>
#include "MueLu_NoFactory.hpp"
#include "MueLu_SaPFactory.hpp"
#include "MueLu_FilteredAFactory.hpp"
#include "MueLu_ReitzingerPFactory_decl.hpp"
#include "Teuchos_ScalarTraitsDecl.hpp"
#include "Teuchos_VerbosityLevel.hpp"
#include <algorithm>

// Garbage sorting code that gives us a permutation array that we can use to reorder
// related entities. We use it in this test to sort all the  abs( An_ij)/sqrt(A_ii A_jj)
// nonzeros. We then use the permuation array to then reorder the nonzeros of An, 
// of the rows and of the columns.
// I'm sure there is a kokkos way of doing this, but I didn't find it immediately
// so I grabbed this off the internet.
template <typename T>
std::vector<unsigned> getSortPermutation(const std::vector<T>& v) {
    std::vector<unsigned> order(v.size());
    std::iota(order.begin(), order.end(), 0); // Fills with 0, 1, 2, ...

    std::sort(order.begin(), order.end(), [&](unsigned i, unsigned j){
        return v[i] > v[j]; // Compares elements in 'v' using indices
    });

    return order;
}
template <typename T>
void applyPermutation(const std::vector<unsigned>& order, std::vector<T>& t) {
    // Assert that the sizes match
    assert(order.size() == t.size());
    
    std::vector<T> temp(t.size());
    for (unsigned i = 0; i < t.size(); ++i) {
      temp[i] = t[order[i]]; // Apply the permutation
    }
    t = temp; // Overwrite the original vector
}
namespace MueLuTests {

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void testNullspaceConstraint(const std::string &matrixType, Teuchos::FancyOStream &out, bool &success) {
#include "MueLu_UseShortNames.hpp"

  using TST                   = Teuchos::ScalarTraits<SC>;
  using magnitude_type        = typename TST::magnitudeType;
  using TMT                   = Teuchos::ScalarTraits<magnitude_type>;
  using real                  = typename TST::coordinateType;
  using RealValuedMultiVector = Xpetra::MultiVector<real, LO, GO, NO>;
  using test_factory          = TestHelpers::TestFactory<SC, LO, GO, NO>;

  out << "version: " << MueLu::Version() << std::endl;

  out << "\n\n==================================================\nTesting " << matrixType << "\n\n";

  Level fineLevel;
  Level coarseLevel;
  test_factory::createTwoLevelHierarchy(fineLevel, coarseLevel);
  fineLevel.SetFactoryManager(Teuchos::null);  // factory manager is not used on this test
  coarseLevel.SetFactoryManager(Teuchos::null);

  GlobalOrdinal nx, ny = 1, nz = 1;
  if (matrixType == "Laplace1D") {
    nx = 200;
  } else if ((matrixType == "Laplace2D") || (matrixType == "Brick2D") || (matrixType == "Elasticity2D")) {
    nx = 20;
    ny = 20;
  } else if ((matrixType == "Laplace3D") || (matrixType == "Brick3D")) {
    nx = 20;
    ny = 20;
    nz = 20;
  } else if (matrixType == "Elasticity3D") {
    nx = 10;
    ny = 10;
    nz = 10;
  }

  Teuchos::ParameterList galeriList;
  galeriList.set("matrixType", matrixType);
  galeriList.set("nx", nx);
  galeriList.set("ny", ny);
  galeriList.set("nz", nz);

  auto [A, coordinates, nullSpace, DofsPerNode] = test_factory::BuildMatrixCoordsNullspace(galeriList);

  fineLevel.Request("A");
  fineLevel.Set("A", A);
  fineLevel.Set("Coordinates", coordinates);
  fineLevel.Set("Nullspace", nullSpace);
  fineLevel.Set("DofsPerNode", DofsPerNode);

  RCP<AmalgamationFactory> amalgFact = rcp(new AmalgamationFactory());
  RCP<CoalesceDropFactory> dropFact  = rcp(new CoalesceDropFactory());
  dropFact->SetFactory("UnAmalgamationInfo", amalgFact);
  RCP<UncoupledAggregationFactory> UncoupledAggFact = rcp(new UncoupledAggregationFactory());
  UncoupledAggFact->SetFactory("Graph", dropFact);
  UncoupledAggFact->SetMinNodesPerAggregate(3);
  UncoupledAggFact->SetMaxNeighAlreadySelected(0);
  UncoupledAggFact->SetOrdering("natural");

  RCP<CoarseMapFactory> coarseMapFact = rcp(new CoarseMapFactory());
  coarseMapFact->SetFactory("Aggregates", UncoupledAggFact);
  RCP<TentativePFactory> TentativePFact = rcp(new TentativePFactory());
  TentativePFact->SetFactory("Aggregates", UncoupledAggFact);
  TentativePFact->SetFactory("UnAmalgamationInfo", amalgFact);
  TentativePFact->SetFactory("CoarseMap", coarseMapFact);

  RCP<PatternFactory> patternFact = rcp(new PatternFactory());
  patternFact->SetFactory("P", TentativePFact);

  RCP<ConstraintFactory> constraintFact = rcp(new ConstraintFactory());
  constraintFact->SetFactory("CoarseNullspace", TentativePFact);
  constraintFact->SetFactory("Ppattern", patternFact);

  RCP<EminPFactory> eminFact = rcp(new EminPFactory());
  eminFact->SetFactory("P", TentativePFact);
  eminFact->SetFactory("Constraint", constraintFact);

  coarseLevel.Request("P", TentativePFact.get());  // request Ptent
  coarseLevel.Request("P", eminFact.get());        // request P
  coarseLevel.Request("Nullspace", TentativePFact.get());
  coarseLevel.Request("Constraint", constraintFact.get());
  coarseLevel.Request(*eminFact);
  TentativePFact->Build(fineLevel, coarseLevel);

  RCP<Matrix> Ptent, P;
  coarseLevel.Get("P", Ptent, TentativePFact.get());
  coarseLevel.Get("P", P, eminFact.get());

  RCP<Constraint> constraint;
  coarseLevel.Get("Constraint", constraint, constraintFact.get());

  using Magnitude = typename Teuchos::ScalarTraits<Scalar>::magnitudeType;
  const auto eps  = Teuchos::ScalarTraits<Magnitude>::eps();

  // Test that Ptent satisfies the constraint.
  TEST_COMPARE(constraint->ResidualNorm(Ptent), <, 400 * eps);

  // Test that both Ptent satisfies the constraint after converting it to a vector and back to a matrix.
  auto vecP = MultiVectorFactory::Build(constraint->getDomainMap(), 1);
  constraint->AssignMatrixEntriesToVector(*Ptent, *vecP);
  auto Ptent2 = constraint->GetMatrixWithEntriesFromVector(*vecP);
  TEST_COMPARE(constraint->ResidualNorm(Ptent2), <, 400 * eps);

  // Teuchos::rcp_const_cast<CrsGraph>(Ptent->getCrsGraph())->computeGlobalConstants();
  // Teuchos::rcp_const_cast<CrsGraph>(Ptent2->getCrsGraph())->computeGlobalConstants();
  // Ptent->describe(out, Teuchos::VERB_EXTREME);
  // Ptent2->describe(out, Teuchos::VERB_EXTREME);

  // Test that P satisfies the constraint.
  TEST_COMPARE(constraint->ResidualNorm(P), <, 20000 * eps);

  // Test that P has lower energy norm than Ptent.
  auto energyNormPtent = EminPFactory::ComputeProlongatorEnergyNorm(A, Ptent, out);
  auto energyNormP     = EminPFactory::ComputeProlongatorEnergyNorm(A, P, out);
  TEST_COMPARE(energyNormP, <, energyNormPtent);
}

TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(EminPFactory, NullspaceConstraint_Laplace1D, Scalar, LocalOrdinal, GlobalOrdinal, Node) {
#include "MueLu_UseShortNames.hpp"
  MUELU_TESTING_SET_OSTREAM;
  MUELU_TESTING_LIMIT_SCOPE(Scalar, GlobalOrdinal, Node);
  testNullspaceConstraint<Scalar, LocalOrdinal, GlobalOrdinal, Node>("Laplace1D", out, success);
}

TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(EminPFactory, NullspaceConstraint_Laplace2D, Scalar, LocalOrdinal, GlobalOrdinal, Node) {
#include "MueLu_UseShortNames.hpp"
  MUELU_TESTING_SET_OSTREAM;
  MUELU_TESTING_LIMIT_SCOPE(Scalar, GlobalOrdinal, Node);
  testNullspaceConstraint<Scalar, LocalOrdinal, GlobalOrdinal, Node>("Laplace2D", out, success);
}

TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(EminPFactory, NullspaceConstraint_Laplace3D, Scalar, LocalOrdinal, GlobalOrdinal, Node) {
#include "MueLu_UseShortNames.hpp"
  MUELU_TESTING_SET_OSTREAM;
  MUELU_TESTING_LIMIT_SCOPE(Scalar, GlobalOrdinal, Node);
  testNullspaceConstraint<Scalar, LocalOrdinal, GlobalOrdinal, Node>("Laplace3D", out, success);
}

TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(EminPFactory, NullspaceConstraint_Brick3D, Scalar, LocalOrdinal, GlobalOrdinal, Node) {
#include "MueLu_UseShortNames.hpp"
  MUELU_TESTING_SET_OSTREAM;
  MUELU_TESTING_LIMIT_SCOPE(Scalar, GlobalOrdinal, Node);
  testNullspaceConstraint<Scalar, LocalOrdinal, GlobalOrdinal, Node>("Brick3D", out, success);
}

TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(EminPFactory, NullspaceConstraint_Elasticity2D, Scalar, LocalOrdinal, GlobalOrdinal, Node) {
#include "MueLu_UseShortNames.hpp"
  MUELU_TESTING_SET_OSTREAM;
  MUELU_TESTING_LIMIT_SCOPE(Scalar, GlobalOrdinal, Node);
  testNullspaceConstraint<Scalar, LocalOrdinal, GlobalOrdinal, Node>("Elasticity2D", out, success);
}

TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(EminPFactory, NullspaceConstraint_Elasticity3D, Scalar, LocalOrdinal, GlobalOrdinal, Node) {
#include "MueLu_UseShortNames.hpp"
  MUELU_TESTING_SET_OSTREAM;
  MUELU_TESTING_LIMIT_SCOPE(Scalar, GlobalOrdinal, Node);
  testNullspaceConstraint<Scalar, LocalOrdinal, GlobalOrdinal, Node>("Elasticity3D", out, success);
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void testMaxwellConstraint(const std::string &inputDir,
                           const bool readNodalProlongators,
                           Teuchos::FancyOStream &out, bool &success) {
#include "MueLu_UseShortNames.hpp"

  using TST                   = Teuchos::ScalarTraits<SC>;
  using magnitude_type        = typename TST::magnitudeType;
  using TMT                   = Teuchos::ScalarTraits<magnitude_type>;
  using real                  = typename TST::coordinateType;
  using RealValuedMultiVector = Xpetra::MultiVector<real, LO, GO, NO>;
  using test_factory          = TestHelpers::TestFactory<SC, LO, GO, NO>;

  out << "version: " << MueLu::Version() << std::endl;

  RCP<const Teuchos::Comm<int>> comm = TestHelpers::Parameters::getDefaultComm();

  std::string scalarName = Teuchos::ScalarTraits<Scalar>::name();
  out << "scalar type = " << scalarName << std::endl;
  if (scalarName.find("complex") != std::string::npos) {
    out << "Skipping Test for SC=complex" << std::endl;
    return;
  }

  auto lib = TestHelpers::Parameters::getLib();

  auto A = Xpetra::IO<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Read(inputDir + "A.dat", lib, comm);
  auto D = Xpetra::IO<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Read(inputDir + "D0h.dat", lib, comm);

  auto fine_edge_map  = A->getDomainMap();
  auto fine_nodal_map = D->getDomainMap();
  TEUCHOS_ASSERT(fine_edge_map->isSameAs(*D->getRangeMap()));

  // Auxiliary nodal hierarchy
  RCP<Matrix> NodeAggMatrix, NodeAggMatrixCoarse, Pnodal, Ptentnodal;
  // Edge hierarchy
  RCP<Matrix> P0, P;
  const bool useExternalP0 = false;
  const bool GrindEmin     = false;

  {
//  uncomment these two defines to run the miniBadSubGraph test
//#define BadSubGraph
//#define writeResult
#define randomlyPerturbAn
#ifdef  BadSubGraph
    NodeAggMatrix = Xpetra::IO<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Read(inputDir + "An.dat", lib, comm);
#else
    auto A_D0     = Xpetra::MatrixMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Multiply(*A, false, *D, false, out, true, true);
    NodeAggMatrix = Xpetra::MatrixMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Multiply(*D, true, *A_D0, false, out, true, true);
#endif
#ifdef randomlyPerturbAn
    Xpetra::CrsMatrixWrap<Scalar, LocalOrdinal, GlobalOrdinal, Node>& crsOp =
                     dynamic_cast<Xpetra::CrsMatrixWrap<Scalar, LocalOrdinal, GlobalOrdinal, Node>&>(*NodeAggMatrix);
    RCP<Xpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>> tmp_CrsMtx = crsOp.getCrsMatrix();
    RCP<Xpetra::TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>>   tmp_TCrsMtx =
       Teuchos::rcp_dynamic_cast<Xpetra::TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>>(tmp_CrsMtx);
#if KOKKOS_VERSION >= 40799
   using ATS              = KokkosKernels::ArithTraits<Scalar>;
   using impl_scalar_type = typename ATS::val_type;
   using implATS          = KokkosKernels::ArithTraits<impl_scalar_type>;
   using mag_type         = typename KokkosKernels::ArithTraits<impl_scalar_type>::magnitudeType;
   using magATS           = KokkosKernels::ArithTraits<mag_type>;
 #else
   using ATS              = Kokkos::ArithTraits<Scalar>;
   using impl_scalar_type = typename ATS::val_type;
   using implATS          = Kokkos::ArithTraits<impl_scalar_type>;
   using mag_type         = typename Kokkos::ArithTraits<impl_scalar_type>::magnitudeType;
   using magATS           = Kokkos::ArithTraits<mag_type>;
 #endif
    Scalar percentOffdiagRetained = .3;
    LO lowestNnzPerRow = 7;
    if (tmp_TCrsMtx != Teuchos::null) {
      RCP<Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>> Ahat = tmp_TCrsMtx->getTpetra_CrsMatrixNonConst();
      auto view    = Ahat->getLocalValuesDevice(Tpetra::Access::ReadWrite);
      auto offs    = Ahat->getLocalRowPtrsDevice();
      auto inds    = Ahat->getLocalIndicesDevice();
      auto lclMap  = Ahat->getRowMap()->getLocalMap();
      auto lclCMap = Ahat->getColMap()->getLocalMap();
      std::vector<bool>  isDirichlet(Ahat->getLocalNumRows(), false);
      GO nDirichlet = 0;
      const auto eps_mag          = magATS::epsilon();
      size_t nnz = offs[Ahat->getLocalNumRows()];
      std::vector<LO> cols(nnz);
      std::vector<LO> rows(nnz);
      std::vector<Scalar> nonzeroVals(nnz);
      // put random numbers into the off-diagonals while setting the
      // diagonal temporarily to 0.0. At the same time record
      // the Dirichlet BC rows
      for (LO i = 0; i < (LO) Ahat->getLocalNumRows(); ++i) {
        LO nz_inRow = 0;

        GO gid = lclMap.getGlobalElement(i);
        for (size_t j = offs[i]; j < offs[i + 1]; j++) {
          GO gidc = lclCMap.getGlobalElement(inds[j]);
          if ( ATS::magnitude(view[j])  > eps_mag) {
            nz_inRow++;
            if (gid != gidc ) {
              unsigned int mySeed = (size_t) 135 + gid + gidc + gid*gidc; //do this to get a symmetric pattern
              Teuchos::ScalarTraits<double>::seedrandom(mySeed);
              view[j] = -1.0 + .5*Teuchos::ScalarTraits<double>::random();
            }
            else {
              view[j] = 0.0;  // set to zero so that the row sum below just includes off diagonals
            }
          }
        }
        if (nz_inRow  < 2)  { isDirichlet[i] = true; nDirichlet++;}
      }

      // set matrix diagonal so that row sums are zero
    
      RCP<MultiVector> onesVector = MultiVectorFactory::Build(NodeAggMatrix->getRowMap(), 1);
      onesVector->putScalar(Teuchos::ScalarTraits<Scalar>::one());
      RCP<MultiVector> rowSumVector = MultiVectorFactory::Build(onesVector->getMap(), 1);
      NodeAggMatrix->apply(*onesVector, *rowSumVector);
      RCP<Vector> ghostedDiag    = Xpetra::VectorFactory<SC, LO, GO, Node>::Build(NodeAggMatrix->getColMap(), true);
      RCP<const Xpetra::Import<LO, GO, Node>> importer;
      importer = NodeAggMatrix->getCrsGraph()->getImporter();
      if (importer == Teuchos::null) {
        importer = Xpetra::ImportFactory<LO, GO, Node>::Build(NodeAggMatrix->getRowMap(), NodeAggMatrix->getColMap());
      }
      auto v0 = rowSumVector->getVector(0);
      ghostedDiag->doImport(*v0, *(importer), Xpetra::INSERT);
      const ArrayRCP<const Scalar> diagData          = ghostedDiag->getData(0);

      // make a copy of the nonzeros but change the values so that are scaled symmetrically 
      // using the diagonal. At the same time, show the diagonal into An.
      for (LO i = 0; i < (LO) Ahat->getLocalNumRows(); ++i) {
        GO gid = lclMap.getGlobalElement(i);
        for (size_t j = offs[i]; j < offs[i + 1]; j++) {
          GO gidc = lclCMap.getGlobalElement(inds[j]);
          rows[j] = (LO) i;
          cols[j] = (LO) inds[j];
          if (gid != gidc ) {
             nonzeroVals[j] = ATS::magnitude(view[j])/( sqrt(ATS::magnitude(diagData[i]))*sqrt(ATS::magnitude(diagData[inds[j]])));
          }
          else { nonzeroVals[j] = 1.0; view[j] = -diagData[i]; }
        }
      }
      // sort the nonzeros in descending order

      auto perm_vec = getSortPermutation(nonzeroVals);
      applyPermutation(perm_vec, nonzeroVals);
      applyPermutation(perm_vec, rows);
      applyPermutation(perm_vec, cols);

      // Let's decide how many nonzeros we should retain based on a user-supplied percentOffdiagRetained
      //
      GO keepLength = (GO) ceil(percentOffdiagRetained* (nnz - Ahat->getLocalNumRows()) + Ahat->getLocalNumRows());
      Scalar tolTarget = nonzeroVals[keepLength-1] - 1.e-5;
      printf("recommending a tolerance of %e to retain %e of the off-diagonal entries\n", (double) tolTarget, (double) percentOffdiagRetained);

      // let's check that this tolerance does not reduce the number of nonzeros in
      // a single row below a user-supplied  lowestNnzPerRow. If the tolerance is
      // to large, adjust it.

      std::vector<LO>  newRowCounts(Ahat->getLocalNumRows(), 0);
      GO nViolations = Ahat->getLocalNumRows() - nDirichlet;
      for (size_t i = 0; i < (size_t) keepLength; i++) {
        if (isDirichlet[rows[i]])
          newRowCounts[rows[i]] = 10000;
        else {
          newRowCounts[rows[i]]++;
          if (newRowCounts[rows[i]] == lowestNnzPerRow)  nViolations--;
        }  
      }
      if ( nViolations > 0) printf("must adjust the tolerance because we would have %d rows with less than %d nonzero\n",(int) nViolations, (int) lowestNnzPerRow);
      while (nViolations > 0) {
        keepLength++; 
        TEUCHOS_TEST_FOR_EXCEPTION(keepLength > (GO) nnz, MueLu::Exceptions::RuntimeError,
                               "The smallest number of nonzeros per row in the undropped matrix is below the user-requested nnzs per row");
        size_t i = keepLength-1;
        newRowCounts[rows[i]]++;
        if (newRowCounts[rows[i]] == lowestNnzPerRow)  nViolations--;
      }
      tolTarget = nonzeroVals[keepLength-1] - 1.e-5;
      printf("the revised suggested tolerance is %e \n", (double) tolTarget);
    }
#endif
  }
  if (readNodalProlongators) {
    Ptentnodal = Xpetra::IO<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Read(inputDir + "Ptent.dat", lib, comm);
    Pnodal     = Xpetra::IO<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Read(inputDir + "Pn.dat", lib, comm);

    TEUCHOS_ASSERT(fine_nodal_map->isSameAs(*Ptentnodal->getRangeMap()));
    TEUCHOS_ASSERT(fine_nodal_map->isSameAs(*Pnodal->getRangeMap()));

    if (useExternalP0) {
      P0 = Xpetra::IO<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Read(inputDir + "Pe.dat", lib, comm);
      TEUCHOS_ASSERT(fine_edge_map->isSameAs(*P0->getRangeMap()));
    }

    auto NodeAggMatrix_P = Xpetra::MatrixMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Multiply(*NodeAggMatrix, false, *Pnodal, false, out, true, true);
    NodeAggMatrixCoarse  = Xpetra::MatrixMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Multiply(*Pnodal, true, *NodeAggMatrix_P, false, out, true, true);
  } else {
    RCP<Hierarchy> H = rcp(new Hierarchy("Nodal Hierarchy"));
    H->setDefaultVerbLevel(Teuchos::VERB_HIGH);

    RCP<Level> fineLevelNodal = H->GetLevel();
    fineLevelNodal->Set("A", NodeAggMatrix);
    FactoryManager M;
    M.SetKokkosRefactor(false);
    RCP<TentativePFactory> Ptentfact = rcp(new TentativePFactory());
    Ptentfact->SetParameter("sa: keep tentative prolongator", Teuchos::ParameterEntry(true));
    Ptentfact->SetParameter("tentative: calculate qr", Teuchos::ParameterEntry(false));
    Ptentfact->SetParameter("tentative: constant column sums", Teuchos::ParameterEntry(false));
    RCP<AmalgamationFactory> amalgFact = rcp(new AmalgamationFactory());
    RCP<CoalesceDropFactory> dropFact  = rcp(new CoalesceDropFactory());
Scalar droptol = 0.0;; 
#ifdef BadSubGraph
    droptol = .01;
#endif
#ifdef randomlyPerturbAn
    printf("enter the drop tolerance\n"); scanf("%lf",&droptol); // I love printf
#endif
    dropFact->SetParameter("aggregation: drop tol",  Teuchos::ParameterEntry(droptol));
    dropFact->SetFactory("UnAmalgamationInfo", amalgFact);
    RCP<UncoupledAggregationFactory> UncoupledAggFact = rcp(new UncoupledAggregationFactory());
    UncoupledAggFact->SetFactory("Graph", dropFact);
    UncoupledAggFact->SetParameter("aggregation: max selected neighbors", Teuchos::ParameterEntry(2));
    RCP<CoarseMapFactory> coarseMapFact = rcp(new CoarseMapFactory());
    coarseMapFact->SetFactory("Aggregates", UncoupledAggFact);
    Ptentfact->SetFactory("Aggregates", UncoupledAggFact);
    Ptentfact->SetFactory("UnAmalgamationInfo", amalgFact);
    Ptentfact->SetFactory("CoarseMap", coarseMapFact);
    Teuchos::ParameterList Pparams;
    Pparams.set("sa: damping factor", 1.33333);
    RCP<SaPFactory> Pnfact = rcp(new SaPFactory);
    Pnfact->SetParameterList(Pparams);
    Pnfact->SetParameter("sa: damping factor", Teuchos::ParameterEntry(1.33333));
    Pnfact->SetFactory("P", Ptentfact);
    RCP<Factory> filterFactory = rcp(new FilteredAFactory());
    filterFactory->SetFactory("Graph", dropFact); // manager.GetFactory("Graph"));
    filterFactory->SetFactory("Filtering", dropFact); //manager.GetFactory("Graph"));
    filterFactory->SetFactory("Aggregates", UncoupledAggFact); // manager.GetFactory("Aggregates"));
    filterFactory->SetFactory("UnAmalgamationInfo", amalgFact); // manager.GetFactory("UnAmalgamationInfo"));
    Pnfact->SetFactory("A", filterFactory);
    M.SetFactory("P", Pnfact);
    M.SetFactory("Ptent", Ptentfact);
    H->SetMaxCoarseSize(1);
    H->Setup(M, 0, 2);

    RCP<Level> coarseLevelNodal = H->GetLevel(1);
    Ptentnodal                  = coarseLevelNodal->Get<RCP<Matrix>>("Ptent");
    Pnodal                      = coarseLevelNodal->Get<RCP<Matrix>>("P");
    Pnodal->RemoveView("stridedMaps");
    Teuchos::rcp_const_cast<CrsGraph>(Pnodal->getCrsGraph())->computeGlobalConstants();
    NodeAggMatrixCoarse         = coarseLevelNodal->Get<RCP<Matrix>>("A");
#ifdef writeResult
    Xpetra::IO<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Write("Ptent.code", *Ptentnodal);
    Xpetra::IO<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Write("Pn.code", *Pnodal);
#endif
  }

  RCP<Constraint> constraint;
  {
    Level fineLevel, coarseLevel;
    if (P0 != Teuchos::null) coarseLevel.Set("P0", P0);
    test_factory::createTwoLevelHierarchy(fineLevel, coarseLevel);
    fineLevel.SetFactoryManager(Teuchos::null);  // factory manager is not used on this test
    coarseLevel.SetFactoryManager(Teuchos::null);

    fineLevel.Set("A", A);
    fineLevel.Set("D0", D);

    // Ptentnodal is used to construct coarse D0
    coarseLevel.Set("Pnodal", Ptentnodal);

    RCP<ReitzingerPFactory> reitzingerFact = rcp(new ReitzingerPFactory());

    // PnodalEmin is used to construct pattern for P
    coarseLevel.Set("PnodalEmin", Pnodal);

    fineLevel.Set("NodeAggMatrix", NodeAggMatrix);
    coarseLevel.Set("NodeAggMatrix", NodeAggMatrixCoarse);

    RCP<EdgeProlongatorPatternFactory> patternFact = rcp(new EdgeProlongatorPatternFactory());
    patternFact->SetFactory("CoarseD0", reitzingerFact);

    RCP<ConstraintFactory> constraintFact = rcp(new ConstraintFactory());
    constraintFact->SetFactory("Ppattern", patternFact);
    constraintFact->SetParameter("emin: constraint type", Teuchos::ParameterEntry(std::string("maxwell")));
    constraintFact->SetFactory("CoarseD0", reitzingerFact);

    RCP<EminPFactory> eminFact = rcp(new EminPFactory());
    eminFact->SetFactory("Constraint", constraintFact);
    eminFact->SetFactory("P", constraintFact);
    if (GrindEmin)
      eminFact->SetParameter("emin: num iterations", Teuchos::ParameterEntry(110));
#ifdef  BadSubGraph
    else
      eminFact->SetParameter("emin: num iterations", Teuchos::ParameterEntry(0));
#endif

    coarseLevel.Request("Constraint", constraintFact.get());
    coarseLevel.Request("P", constraintFact.get());
    coarseLevel.Request("P", eminFact.get());
    coarseLevel.Request(*eminFact);

    coarseLevel.Get("Constraint", constraint, constraintFact.get());

    // The initial guess used for emin starts up being call P

    if (coarseLevel.IsAvailable("P0")) {
      coarseLevel.Get("P0", P0);
    } else {
      coarseLevel.Get("P", P0, constraintFact.get());
    }

    // This is the result after running the minimization.
    coarseLevel.Get("P", P, eminFact.get());
#ifdef writeResult
    RCP<Matrix> D0H;
    D0H = coarseLevel.Get<RCP<Matrix>>("D0");
    GO nFineEdges  =   P->getRowMap()->getGlobalNumElements();
    GO nCoarEdges  = D0H->getRowMap()->getGlobalNumElements();
    std::string suffix = inputDir;
    std::replace(suffix.begin(), suffix.end(), '/', '_');
    std::string sub = "emin_matrices";
    size_t pos = suffix.find(sub);
    if (pos != std::string::npos) suffix.erase(pos, sub.length());

    Xpetra::IO<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Write("Pe_" + std::to_string(nFineEdges) + suffix + ".code", *P);
    Xpetra::IO<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Write("DH_" + std::to_string(nCoarEdges) + suffix +".code", *D0H);
#endif
  }

  const auto eps = Teuchos::ScalarTraits<magnitude_type>::eps();

  // Test that P0 satisfies the constraint.
  TEST_COMPARE(constraint->ResidualNorm(P0), <, 400000 * eps);

  // Test that P satisfies the constraint.
  TEST_COMPARE(constraint->ResidualNorm(P), <, 40000000000000 * eps);

  // Test that P has lower energy norm than P0.
  auto energyNormP0 = EminPFactory::ComputeProlongatorEnergyNorm(A, P0, out);
  auto energyNormP  = EminPFactory::ComputeProlongatorEnergyNorm(A, P, out);
  TEST_COMPARE(energyNormP, <, energyNormP0);
}

TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(EminPFactory, MaxwellConstraint_1, Scalar, LocalOrdinal, GlobalOrdinal, Node) {
#include "MueLu_UseShortNames.hpp"
  MUELU_TESTING_SET_OSTREAM;
  MUELU_TESTING_LIMIT_SCOPE(Scalar, GlobalOrdinal, Node);
  testMaxwellConstraint<Scalar, LocalOrdinal, GlobalOrdinal, Node>(/*inputDir=*/"emin_matrices/",
                                                                   /*readNodalProlongators=*/false,
                                                                   out, success);
}

TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(EminPFactory, MaxwellConstraint_Tris, Scalar, LocalOrdinal, GlobalOrdinal, Node) {
#include "MueLu_UseShortNames.hpp"
  MUELU_TESTING_SET_OSTREAM;
  MUELU_TESTING_LIMIT_SCOPE(Scalar, GlobalOrdinal, Node);
  testMaxwellConstraint<Scalar, LocalOrdinal, GlobalOrdinal, Node>(/*inputDir=*/"emin_matrices/tris/",
                                                                   /*readNodalProlongators=*/true,
                                                                   out, success);
}

TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(EminPFactory, MaxwellConstraint_TrisWithDir, Scalar, LocalOrdinal, GlobalOrdinal, Node) {
#include "MueLu_UseShortNames.hpp"
  MUELU_TESTING_SET_OSTREAM;
  MUELU_TESTING_LIMIT_SCOPE(Scalar, GlobalOrdinal, Node);
  testMaxwellConstraint<Scalar, LocalOrdinal, GlobalOrdinal, Node>(/*inputDir=*/"emin_matrices/tris/withDir/",
                                                                   /*readNodalProlongators=*/true,
                                                                   out, success);
}

TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(EminPFactory, MaxwellConstraint_Quads, Scalar, LocalOrdinal, GlobalOrdinal, Node) {
#include "MueLu_UseShortNames.hpp"
  MUELU_TESTING_SET_OSTREAM;
  MUELU_TESTING_LIMIT_SCOPE(Scalar, GlobalOrdinal, Node);
  testMaxwellConstraint<Scalar, LocalOrdinal, GlobalOrdinal, Node>(/*inputDir=*/"emin_matrices/quads/",
                                                                   /*readNodalProlongators=*/true,
                                                                   out, success);
}

TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(EminPFactory, MaxwellConstraint_QuadsWithDir, Scalar, LocalOrdinal, GlobalOrdinal, Node) {
#include "MueLu_UseShortNames.hpp"
  MUELU_TESTING_SET_OSTREAM;
  MUELU_TESTING_LIMIT_SCOPE(Scalar, GlobalOrdinal, Node);
  testMaxwellConstraint<Scalar, LocalOrdinal, GlobalOrdinal, Node>(/*inputDir=*/"emin_matrices/quads/withDir/",
                                                                   /*readNodalProlongators=*/true,
                                                                   out, success);
}

TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(EminPFactory, MaxwellConstraint_Tets, Scalar, LocalOrdinal, GlobalOrdinal, Node) {
#include "MueLu_UseShortNames.hpp"
  MUELU_TESTING_SET_OSTREAM;
  MUELU_TESTING_LIMIT_SCOPE(Scalar, GlobalOrdinal, Node);
  testMaxwellConstraint<Scalar, LocalOrdinal, GlobalOrdinal, Node>(/*inputDir=*/"emin_matrices/tets/",
                                                                   /*readNodalProlongators=*/true,
                                                                   out, success);
}

TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(EminPFactory, MaxwellConstraint_TetsWithDir, Scalar, LocalOrdinal, GlobalOrdinal, Node) {
#include "MueLu_UseShortNames.hpp"
  MUELU_TESTING_SET_OSTREAM;
  MUELU_TESTING_LIMIT_SCOPE(Scalar, GlobalOrdinal, Node);
  testMaxwellConstraint<Scalar, LocalOrdinal, GlobalOrdinal, Node>(/*inputDir=*/"emin_matrices/tets/withDir/",
                                                                   /*readNodalProlongators=*/true,
                                                                   out, success);
}

TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(EminPFactory, MaxwellConstraint_Hexes, Scalar, LocalOrdinal, GlobalOrdinal, Node) {
#include "MueLu_UseShortNames.hpp"
  MUELU_TESTING_SET_OSTREAM;
  MUELU_TESTING_LIMIT_SCOPE(Scalar, GlobalOrdinal, Node);
  testMaxwellConstraint<Scalar, LocalOrdinal, GlobalOrdinal, Node>(/*inputDir=*/"emin_matrices/hexes/",
                                                                   /*readNodalProlongators=*/true,
                                                                   out, success);
}

TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(EminPFactory, MaxwellRandPerubAn_Hexes, Scalar, LocalOrdinal, GlobalOrdinal, Node) {
#include "MueLu_UseShortNames.hpp"
  MUELU_TESTING_SET_OSTREAM;
  MUELU_TESTING_LIMIT_SCOPE(Scalar, GlobalOrdinal, Node);
  // I believe this data is on balda.
  testMaxwellConstraint<Scalar, LocalOrdinal, GlobalOrdinal, Node>(/*inputDir=*/"emin_matrices/hexes/stretch/squishZgood/lev0/",
                                                                   /*readNodalProlongators=*/false,
                                                                   out, success);
}

TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(EminPFactory, MaxwellConstraint_HexesWithDir, Scalar, LocalOrdinal, GlobalOrdinal, Node) {
#include "MueLu_UseShortNames.hpp"
  MUELU_TESTING_SET_OSTREAM;
  MUELU_TESTING_LIMIT_SCOPE(Scalar, GlobalOrdinal, Node);
  testMaxwellConstraint<Scalar, LocalOrdinal, GlobalOrdinal, Node>(/*inputDir=*/"emin_matrices/hexes/withDir/",
                                                                   /*readNodalProlongators=*/true,
                                                                   out, success);
}
TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(EminPFactory, miniBadSubGraph,  Scalar, LocalOrdinal, GlobalOrdinal, Node) {
#include "MueLu_UseShortNames.hpp"
  MUELU_TESTING_SET_OSTREAM;
  MUELU_TESTING_LIMIT_SCOPE(Scalar, GlobalOrdinal, Node);
  testMaxwellConstraint<Scalar, LocalOrdinal, GlobalOrdinal, Node>(/*inputDir=*/"emin_matrices/graphDisconnect/",
                                                                   /*readNodalProlongators=*/false, out, success);
}

#define MUELU_ETI_GROUP(SC, LO, GO, Node)                                                                \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(EminPFactory, NullspaceConstraint_Laplace1D, SC, LO, GO, Node)    \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(EminPFactory, NullspaceConstraint_Laplace2D, SC, LO, GO, Node)    \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(EminPFactory, NullspaceConstraint_Laplace3D, SC, LO, GO, Node)    \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(EminPFactory, NullspaceConstraint_Brick3D, SC, LO, GO, Node)      \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(EminPFactory, NullspaceConstraint_Elasticity2D, SC, LO, GO, Node) \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(EminPFactory, NullspaceConstraint_Elasticity3D, SC, LO, GO, Node) \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(EminPFactory, MaxwellConstraint_1, SC, LO, GO, Node)              \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(EminPFactory, MaxwellConstraint_Tris, SC, LO, GO, Node)           \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(EminPFactory, MaxwellConstraint_TrisWithDir, SC, LO, GO, Node)    \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(EminPFactory, MaxwellConstraint_Quads, SC, LO, GO, Node)          \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(EminPFactory, MaxwellConstraint_QuadsWithDir, SC, LO, GO, Node)   \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(EminPFactory, MaxwellConstraint_Tets, SC, LO, GO, Node)           \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(EminPFactory, MaxwellConstraint_TetsWithDir, SC, LO, GO, Node)    \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(EminPFactory, MaxwellConstraint_Hexes, SC, LO, GO, Node)          \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(EminPFactory, MaxwellConstraint_HexesWithDir, SC, LO, GO, Node)   \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(EminPFactory, MaxwellRandPerubAn_Hexes, SC, LO, GO, Node)         \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(EminPFactory, miniBadSubGraph, SC, LO, GO, Node) 

#include <MueLu_ETI_4arg.hpp>

}  // namespace MueLuTests
