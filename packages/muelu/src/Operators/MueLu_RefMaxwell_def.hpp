// @HEADER
//
// ***********************************************************************
//
//        MueLu: A package for multigrid based preconditioning
//                  Copyright 2012 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact
//                    Jonathan Hu       (jhu@sandia.gov)
//                    Andrey Prokopenko (aprokop@sandia.gov)
//                    Ray Tuminaro      (rstumin@sandia.gov)
//
// ***********************************************************************
//
// @HEADER
#ifndef MUELU_REFMAXWELL_DEF_HPP
#define MUELU_REFMAXWELL_DEF_HPP

#include <sstream>

#include "MueLu_ConfigDefs.hpp"

#include "Xpetra_Map.hpp"
#include "Xpetra_MatrixMatrix.hpp"
#include "Xpetra_TripleMatrixMultiply.hpp"
#include "Xpetra_CrsMatrixUtils.hpp"
#include "Xpetra_MatrixUtils.hpp"

#include "MueLu_RefMaxwell_decl.hpp"

#include "MueLu_AmalgamationFactory.hpp"
#include "MueLu_RAPFactory.hpp"
#include "MueLu_SmootherFactory.hpp"

#include "MueLu_CoalesceDropFactory.hpp"
#include "MueLu_CoarseMapFactory.hpp"
#include "MueLu_CoordinatesTransferFactory.hpp"
#include "MueLu_UncoupledAggregationFactory.hpp"
#include "MueLu_TentativePFactory.hpp"
#include "MueLu_SaPFactory.hpp"
#include "MueLu_AggregationExportFactory.hpp"
#include "MueLu_Utilities.hpp"
#include "MueLu_Maxwell_Utils.hpp"

#include "MueLu_CoalesceDropFactory_kokkos.hpp"
#include "MueLu_UncoupledAggregationFactory_kokkos.hpp"
#include "MueLu_TentativePFactory_kokkos.hpp"
#include "MueLu_SaPFactory_kokkos.hpp"
#include <Kokkos_Core.hpp>
#include <KokkosSparse_CrsMatrix.hpp>

#include "MueLu_ZoltanInterface.hpp"
#include "MueLu_Zoltan2Interface.hpp"
#include "MueLu_RepartitionHeuristicFactory.hpp"
#include "MueLu_RepartitionFactory.hpp"
#include "MueLu_RebalanceAcFactory.hpp"
#include "MueLu_RebalanceTransferFactory.hpp"

#include "MueLu_VerbosityLevel.hpp"

#include <MueLu_CreateXpetraPreconditioner.hpp>
#include <MueLu_ML2MueLuParameterTranslator.hpp>

#ifdef HAVE_MUELU_CUDA
#include "cuda_profiler_api.h"
#endif

// Stratimikos
#if defined(HAVE_MUELU_STRATIMIKOS) && defined(HAVE_MUELU_THYRA)
#include <Xpetra_ThyraLinearOp.hpp>
#endif


namespace MueLu {

  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  Teuchos::RCP<Xpetra::CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >
  Matrix2CrsMatrix(Teuchos::RCP<Xpetra::CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> > &matrix) {
    return Teuchos::rcp_dynamic_cast<Xpetra::CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node> >(matrix, true)->getCrsMatrix();
  }


  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  Teuchos::RCP<const Xpetra::Map<LocalOrdinal,GlobalOrdinal,Node> > RefMaxwell<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getDomainMap() const {
    return SM_Matrix_->getDomainMap();
  }


  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  Teuchos::RCP<const Xpetra::Map<LocalOrdinal,GlobalOrdinal,Node> > RefMaxwell<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getRangeMap() const {
    return SM_Matrix_->getRangeMap();
  }


  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  Teuchos::RCP<Teuchos::ParameterList>
  RefMaxwell<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
  getValidParamterList() {

    bool useKokkosDefault = false;
#ifdef HAVE_MUELU_SERIAL
    if (typeid(Node).name() == typeid(Tpetra::KokkosCompat::KokkosSerialWrapperNode).name())
      useKokkosDefault = false;
#endif
#ifdef HAVE_MUELU_OPENMP
    if (typeid(Node).name() == typeid(Tpetra::KokkosCompat::KokkosOpenMPWrapperNode).name())
      useKokkosDefault = true;
#endif
#ifdef HAVE_MUELU_CUDA
    if (typeid(Node).name() == typeid(Tpetra::KokkosCompat::KokkosCudaWrapperNode).name())
      useKokkosDefault = true;
#endif
#ifdef HAVE_MUELU_HIP
    if (typeid(Node).name() == typeid(Tpetra::KokkosCompat::KokkosHIPWrapperNode).name())
      useKokkosDefault = true;
#endif
#ifdef HAVE_MUELU_SYCL
    if (typeid(Node).name() == typeid(Tpetra::KokkosCompat::KokkosSYCLWrapperNode).name())
      useKokkosDefault = true;
#endif

    RCP<ParameterList> params = rcp(new ParameterList("RefMaxwell"));

    params->set<RCP<Matrix> >("Dk_1", Teuchos::null);
    params->set<RCP<Matrix> >("Dk_2", Teuchos::null);
    params->set<RCP<Matrix> >("D0", Teuchos::null);

    params->set<RCP<Matrix> >("M1_beta", Teuchos::null);
    params->set<RCP<Matrix> >("M1_alpha", Teuchos::null);
    params->set<RCP<Matrix> >("Ms", Teuchos::null);

    params->set<RCP<Matrix> >("Mk_one", Teuchos::null);
    params->set<RCP<Matrix> >("Mk_1_one", Teuchos::null);
    params->set<RCP<Matrix> >("M1", Teuchos::null);

    params->set<RCP<Matrix> >("invMk_1_invBeta", Teuchos::null);
    params->set<RCP<Matrix> >("invMk_2_invAlpha", Teuchos::null);
    params->set<RCP<Matrix> >("M0inv", Teuchos::null);

    params->set<RCP<MultiVector> >("Nullspace", Teuchos::null);
    params->set<RCP<RealValuedMultiVector> >("Coordinates", Teuchos::null);

    params->set("refmaxwell: space number", 1);
    params->set("verbosity", MasterList::getDefault<std::string>("verbosity"));
    params->set("use kokkos refactor", useKokkosDefault);
    params->set("half precision", false);
    params->set("parameterlist: syntax", MasterList::getDefault<std::string>("parameterlist: syntax"));
    params->set("output filename", MasterList::getDefault<std::string>("output filename"));
    params->set("print initial parameters", MasterList::getDefault<bool>("print initial parameters"));
    params->set("refmaxwell: disable addon", MasterList::getDefault<bool>("refmaxwell: disable addon"));
    params->set("refmaxwell: disable addon 22", true);
    params->set("refmaxwell: mode", MasterList::getDefault<std::string>("refmaxwell: mode"));
    params->set("refmaxwell: use as preconditioner", MasterList::getDefault<bool>("refmaxwell: use as preconditioner"));
    params->set("refmaxwell: dump matrices", MasterList::getDefault<bool>("refmaxwell: dump matrices"));
    params->set("refmaxwell: enable reuse", MasterList::getDefault<bool>("refmaxwell: enable reuse"));
    params->set("refmaxwell: skip first (1,1) level", MasterList::getDefault<bool>("refmaxwell: skip first (1,1) level"));
    params->set("transpose: use implicit", MasterList::getDefault<bool>("transpose: use implicit"));
    params->set("fuse prolongation and update", MasterList::getDefault<bool>("fuse prolongation and update"));
    params->set("refmaxwell: subsolves on subcommunicators", MasterList::getDefault<bool>("refmaxwell: subsolves on subcommunicators"));
    params->set("refmaxwell: subsolves striding", 1);
    params->set("refmaxwell: row sum drop tol (1,1)", MasterList::getDefault<double>("aggregation: row sum drop tol"));
    params->set("sync timers", false);
    params->set("refmaxwell: num iters coarse 11", 1);
    params->set("refmaxwell: num iters 22", 1);
    params->set("refmaxwell: apply BCs to Anodal",    false);
    params->set("refmaxwell: apply BCs to coarse 11", true);
    params->set("refmaxwell: apply BCs to 22",        true);

    ParameterList & precList11 = params->sublist("refmaxwell: 11list");
    precList11.disableRecursiveValidation();
    ParameterList & precList22 = params->sublist("refmaxwell: 22list");
    precList22.disableRecursiveValidation();

    params->set("smoother: type", "CHEBYSHEV");
    ParameterList & smootherList = params->sublist("smoother: params");
    smootherList.disableRecursiveValidation();
    params->set("smoother: pre type", "NONE");
    ParameterList & preSmootherList = params->sublist("smoother: pre params");
    preSmootherList.disableRecursiveValidation();
    params->set("smoother: post type", "NONE");
    ParameterList & postSmootherList = params->sublist("smoother: post params");
    postSmootherList.disableRecursiveValidation();

    ParameterList & matvecParams = params->sublist("matvec params");
    matvecParams.disableRecursiveValidation();

    params->set("multigrid algorithm", "unsmoothed");
    params->set("aggregation: drop tol", MasterList::getDefault<double>("aggregation: drop tol"));
    params->set("aggregation: drop scheme", MasterList::getDefault<std::string>("aggregation: drop scheme"));
    params->set("aggregation: distance laplacian algo", MasterList::getDefault<std::string>("aggregation: distance laplacian algo"));
    params->set("aggregation: min agg size", MasterList::getDefault<int>("aggregation: min agg size"));
    params->set("aggregation: max agg size", MasterList::getDefault<int>("aggregation: max agg size"));
    params->set("aggregation: export visualization data", MasterList::getDefault<bool>("aggregation: export visualization data"));

    return params;
  }


  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void RefMaxwell<Scalar,LocalOrdinal,GlobalOrdinal,Node>::setParameters(Teuchos::ParameterList& list) {

    if (list.isType<std::string>("parameterlist: syntax") && list.get<std::string>("parameterlist: syntax") == "ml") {
      Teuchos::ParameterList newList = *Teuchos::getParametersFromXmlString(MueLu::ML2MueLuParameterTranslator::translate(list,"refmaxwell"));
      if(list.isSublist("refmaxwell: 11list") && list.sublist("refmaxwell: 11list").isSublist("edge matrix free: coarse"))
        newList.sublist("refmaxwell: 11list") = *Teuchos::getParametersFromXmlString(MueLu::ML2MueLuParameterTranslator::translate(list.sublist("refmaxwell: 11list").sublist("edge matrix free: coarse"),"SA"));
      if(list.isSublist("refmaxwell: 22list"))
        newList.sublist("refmaxwell: 22list") = *Teuchos::getParametersFromXmlString(MueLu::ML2MueLuParameterTranslator::translate(list.sublist("refmaxwell: 22list"),"SA"));
      list = newList;
    }

    parameterList_             = list;
    parameterList_.validateParametersAndSetDefaults(*getValidParamterList());
    std::string verbosityLevel = parameterList_.get<std::string>("verbosity");
    VerboseObject::SetDefaultVerbLevel(toVerbLevel(verbosityLevel));
    std::string outputFilename = parameterList_.get<std::string>("output filename");
    if (outputFilename != "")
      VerboseObject::SetMueLuOFileStream(outputFilename);
    if (parameterList_.isType<Teuchos::RCP<Teuchos::FancyOStream> >("output stream"))
      VerboseObject::SetMueLuOStream(parameterList_.get<Teuchos::RCP<Teuchos::FancyOStream> >("output stream"));

    if (parameterList_.get<bool>("print initial parameters"))
      GetOStream(static_cast<MsgType>(Runtime1), 0) << parameterList_ << std::endl;
    disable_addon_             = parameterList_.get<bool>("refmaxwell: disable addon");
    disable_addon_22_          = parameterList_.get<bool>("refmaxwell: disable addon 22");
    mode_                      = parameterList_.get<std::string>("refmaxwell: mode");
    use_as_preconditioner_     = parameterList_.get<bool>("refmaxwell: use as preconditioner");
    dump_matrices_             = parameterList_.get<bool>("refmaxwell: dump matrices");
    enable_reuse_              = parameterList_.get<bool>("refmaxwell: enable reuse");
    implicitTranspose_         = parameterList_.get<bool>("transpose: use implicit");
    fuseProlongationAndUpdate_ = parameterList_.get<bool>("fuse prolongation and update");
    skipFirstLevel_            = parameterList_.get<bool>("refmaxwell: skip first (1,1) level");
    syncTimers_                = parameterList_.get<bool>("sync timers");
    useKokkos_                 = parameterList_.get<bool>("use kokkos refactor");
    numItersCoarse11_          = parameterList_.get<int>("refmaxwell: num iters coarse 11");
    numIters22_                = parameterList_.get<int>("refmaxwell: num iters 22");
    applyBCsToAnodal_          = parameterList_.get<bool>("refmaxwell: apply BCs to Anodal");
    applyBCsToCoarse11_        = parameterList_.get<bool>("refmaxwell: apply BCs to coarse 11");
    applyBCsTo22_              = parameterList_.get<bool>("refmaxwell: apply BCs to 22");

    precList11_     =  parameterList_.sublist("refmaxwell: 11list");
    if(!precList11_.isType<std::string>("Preconditioner Type") &&
       !precList11_.isType<std::string>("smoother: type") &&
       !precList11_.isType<std::string>("smoother: pre type") &&
       !precList11_.isType<std::string>("smoother: post type")) {
      precList11_.set("smoother: type", "CHEBYSHEV");
      precList11_.sublist("smoother: params").set("chebyshev: degree",2);
      precList11_.sublist("smoother: params").set("chebyshev: ratio eigenvalue",5.4);
      precList11_.sublist("smoother: params").set("chebyshev: eigenvalue max iterations",30);
    }

    precList22_     =  parameterList_.sublist("refmaxwell: 22list");
    if(!precList22_.isType<std::string>("Preconditioner Type") &&
       !precList22_.isType<std::string>("smoother: type") &&
       !precList22_.isType<std::string>("smoother: pre type") &&
       !precList22_.isType<std::string>("smoother: post type")) {
      precList22_.set("smoother: type", "CHEBYSHEV");
      precList22_.sublist("smoother: params").set("chebyshev: degree",2);
      precList22_.sublist("smoother: params").set("chebyshev: ratio eigenvalue",7.0);
      precList22_.sublist("smoother: params").set("chebyshev: eigenvalue max iterations",30);
    }

    if(!parameterList_.isType<std::string>("smoother: type") && !parameterList_.isType<std::string>("smoother: pre type") && !parameterList_.isType<std::string>("smoother: post type")) {
      list.set("smoother: type", "CHEBYSHEV");
      list.sublist("smoother: params").set("chebyshev: degree",2);
      list.sublist("smoother: params").set("chebyshev: ratio eigenvalue",20.0);
      list.sublist("smoother: params").set("chebyshev: eigenvalue max iterations",30);
    }

    smootherList_ = parameterList_.sublist("smoother: params");

    if (enable_reuse_ &&
        !precList11_.isType<std::string>("Preconditioner Type") &&
        !precList11_.isParameter("reuse: type"))
      precList11_.set("reuse: type", "full");
    if (enable_reuse_ &&
        !precList22_.isType<std::string>("Preconditioner Type") &&
        !precList22_.isParameter("reuse: type"))
      precList22_.set("reuse: type", "full");

    // This should be taken out again as soon as
    // CoalesceDropFactory_kokkos supports BlockSize > 1 and
    // drop tol != 0.0
    if (useKokkos_ && precList11_.isParameter("aggregation: drop tol") && precList11_.get<double>("aggregation: drop tol") != 0.0) {
      GetOStream(Warnings0) << "RefMaxwell::compute(): Setting \"aggregation: drop tol\". to 0.0, since CoalesceDropFactory_kokkos does not "
                            << "support BlockSize > 1 and drop tol != 0.0" << std::endl;
      precList11_.set("aggregation: drop tol", 0.0);
    }
  }


  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void RefMaxwell<Scalar,LocalOrdinal,GlobalOrdinal,Node>::compute(bool reuse) {

#ifdef HAVE_MUELU_CUDA
    if (parameterList_.get<bool>("refmaxwell: cuda profile setup", false)) cudaProfilerStart();
#endif

    std::string timerLabel;
    if (reuse)
      timerLabel = "MueLu RefMaxwell: compute (reuse)";
    else
      timerLabel = "MueLu RefMaxwell: compute";
    RCP<Teuchos::TimeMonitor> tmCompute = getTimer(timerLabel);

    ////////////////////////////////////////////////////////////////////////////////
    // Remove explicit zeros from matrices
    Maxwell_Utils<SC,LO,GO,NO>::removeExplicitZeros(parameterList_,D0_,SM_Matrix_,Mk_one_,M1_beta_);

    if (IsPrint(Statistics2)) {
      RCP<ParameterList> params = rcp(new ParameterList());;
      params->set("printLoadBalancingInfo", true);
      params->set("printCommInfo",          true);
      GetOStream(Statistics2) << PerfUtils::PrintMatrixInfo(*SM_Matrix_, "SM_Matrix", params);
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Detect Dirichlet boundary conditions
    if (!reuse) {
      magnitudeType rowSumTol = parameterList_.get<double>("refmaxwell: row sum drop tol (1,1)");
      Maxwell_Utils<SC,LO,GO,NO>::detectBoundaryConditionsSM(SM_Matrix_,Dk_1_,rowSumTol,
                                                             BCrows11_,BCcols22_,BCdomain22_,
                                                             globalNumberBoundaryUnknowns11_,
                                                             globalNumberBoundaryUnknowns22_,
                                                             onlyBoundary11_,onlyBoundary22_);
      if (IsPrint(Statistics2)) {
        GetOStream(Statistics2) << "MueLu::RefMaxwell::compute(): Detected " << globalNumberBoundaryUnknowns11_ << " BC rows and " << globalNumberBoundaryUnknowns22_ << " BC columns." << std::endl;
      }
    }

    if (onlyBoundary11_) {
      // All unknowns of the (1,1) block have been detected as boundary unknowns.
      // Do not attempt to construct sub-hierarchies, but just set up a single level preconditioner.
      GetOStream(Warnings0) << "All unknowns of the (1,1) block have been detected as boundary unknowns!" << std::endl;
      mode_ = "none";
      setFineLevelSmoother();
      return;
    }

    ////////////////////////////////////////////////////////////////////////////////
    // build nullspace (if necessary)
    if (!reuse)
      buildNullspace();

    ////////////////////////////////////////////////////////////////////////////////
    // build special prolongators
    if (!reuse) {

      {
        // build special prolongator for (1,1)-block
        RCP<Matrix> A11_nodal;
        if (skipFirstLevel_) {
          // Form A11_nodal = D0^T * M1_beta * D0  (aka TMT_agg)
          std::string label("D0^T*M1_beta*D0");
          A11_nodal = Maxwell_Utils<SC,LO,GO,NO>::PtAPWrapper(M1_beta_,D0_,parameterList_,label);

          if (applyBCsToAnodal_) {
            // Apply boundary conditions to A11_nodal
            Utilities::ApplyOAZToMatrixRows(A11_nodal,BCdomain22_);
          }
          dump(*A11_nodal, "A11_nodal.m");
        }

        // build special prolongator
        GetOStream(Runtime0) << "RefMaxwell::compute(): building special prolongator" << std::endl;
        if (spaceNumber_ == 1)
          buildEdgeProlongator(A11_nodal, P11_, NullspaceCoarse11_, CoordsCoarse11_);
        else if (spaceNumber_ == 2)
          buildFaceProlongator(A11_nodal, P11_, NullspaceCoarse11_, CoordsCoarse11_);
        else
          throw;
        TEUCHOS_ASSERT(NullspaceCoarse11_.is_null() == !skipFirstLevel_);
      }

      // build special prolongator for (2,2)-block
      if (spaceNumber_ >= 2) {
        // Form A22_nodal = D0^T * M1_alpha * D0
        std::string label("D0^T*M1_alpha*D0");
        RCP<Matrix> A22_nodal = Maxwell_Utils<SC,LO,GO,NO>::PtAPWrapper(M1_alpha_,D0_,parameterList_,label);

        if (applyBCsToAnodal_) {
          // Apply boundary conditions to A22_nodal
          Utilities::ApplyOAZToMatrixRows(A22_nodal,BCdomain22_);
        }
        dump(*A22_nodal, "A22_nodal.m");

        if (spaceNumber_ == 2)
          buildEdgeProlongator(A22_nodal, P22_, CoarseNullspace22_, CoordsCoarse22_);
        else
          throw;
      }
    }

    ////////////////////////////////////////////////////////////////////////////////
    // build coarse grid operator for (1,1)-block
    buildCoarse11Matrix();

    ////////////////////////////////////////////////////////////////////////////////
    // determine the communicator sizes for (1,1)- and (2,2)-blocks
    bool doRebalancing;
    int rebalanceStriding, numProcsCoarseA11, numProcsA22;
    if (!reuse)
      this->determineSubHierarchyCommSizes(doRebalancing, rebalanceStriding, numProcsCoarseA11, numProcsA22);
    else
      doRebalancing = false;

    // rebalance the coarse A11 matrix, as well as P11, CoordsCoarse11 and Addon_Matrix
    if (!reuse && doRebalancing)
      rebalanceCoarse11Matrix(rebalanceStriding, numProcsCoarseA11);
    if (!coarseA11_.is_null()) {
      dump(*coarseA11_, "coarseA11.m");
      if (!reuse) {
        dumpCoords(*CoordsCoarse11_, "CoordsCoarse11.m");
        if (!NullspaceCoarse11_.is_null())
          dump(*NullspaceCoarse11_, "NullspaceCoarse11.m");
      }
    }

    if (!reuse) {
      dump(*P11_, "P11.m");
      if (!implicitTranspose_) {
        R11_ = Utilities::Transpose(*P11_);
        dump(*R11_, "R11.m");
      }
    }

    ////////////////////////////////////////////////////////////////////////////////
    // build multigrid for coarse (1,1)-block
    if (!coarseA11_.is_null()) {
      VerbLevel verbosityLevel = VerboseObject::GetDefaultVerbLevel();
      std::string label("coarseA11");
      setupSubSolve(HierarchyCoarse11_, thyraPrecOpH_, coarseA11_, NullspaceCoarse11_, CoordsCoarse11_, precList11_, label, reuse);
      VerboseObject::SetDefaultVerbLevel(verbosityLevel);
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Apply BCs to columns of Dk_1
    if(!reuse && applyBCsTo22_) {
      GetOStream(Runtime0) << "RefMaxwell::compute(): nuking BC nodes of Dk_1" << std::endl;

      Dk_1_->resumeFill();
      Scalar replaceWith = (Dk_1_->getRowMap()->lib() == Xpetra::UseEpetra) ? Teuchos::ScalarTraits<SC>::eps() : Teuchos::ScalarTraits<SC>::zero();
      Utilities::ZeroDirichletCols(Dk_1_,BCcols22_,replaceWith);
      Dk_1_->fillComplete(Dk_1_->getDomainMap(),Dk_1_->getRangeMap());
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Build A22 = Dk_1^T SM Dk_1 and hierarchy for A22
    if (!onlyBoundary22_) {
      GetOStream(Runtime0) << "RefMaxwell::compute(): building MG for (2,2)-block" << std::endl;

      // Build A22 = Dk_1^T * SM * Dk_1 and rebalance it, as well as Dk_1 and Coords_
      build22Matrix(reuse, doRebalancing, rebalanceStriding, numProcsA22);

      if (!reuse && !implicitTranspose_)
        Dk_1_T_ = Utilities::Transpose(*Dk_1_);

      if (!A22_.is_null()) {
        VerbLevel verbosityLevel = VerboseObject::GetDefaultVerbLevel();
        std::string label("A22");
        setupSubSolve(Hierarchy22_, thyraPrecOp22_, A22_, Teuchos::null, Coords_, precList22_, label, reuse, /*isSingular=*/globalNumberBoundaryUnknowns11_ == 0);
        VerboseObject::SetDefaultVerbLevel(verbosityLevel);
      }
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Apply BCs to rows of Dk_1
    if(!reuse && !onlyBoundary22_ && applyBCsTo22_) {
      GetOStream(Runtime0) << "RefMaxwell::compute(): nuking BC edges of Dk_1" << std::endl;

      Dk_1_->resumeFill();
      Scalar replaceWith = (Dk_1_->getRowMap()->lib() == Xpetra::UseEpetra) ? Teuchos::ScalarTraits<SC>::eps() : Teuchos::ScalarTraits<SC>::zero();
      Utilities::ZeroDirichletRows(Dk_1_,BCrows11_,replaceWith);
      Dk_1_->fillComplete(Dk_1_->getDomainMap(),Dk_1_->getRangeMap());
      dump(*Dk_1_, "Dk_1_nuked.m");
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Set up the smoother on the finest level
    setFineLevelSmoother();

    if (!reuse) {
      if (!ImporterCoarse11_.is_null()) {
        RCP<const Import> ImporterP11 = ImportFactory::Build(ImporterCoarse11_->getTargetMap(),P11_->getColMap());
        rcp_dynamic_cast<CrsMatrixWrap>(P11_)->getCrsMatrix()->replaceDomainMapAndImporter(ImporterCoarse11_->getTargetMap(), ImporterP11);
      }

      if (!Importer22_.is_null()) {
        if (enable_reuse_) {
          DorigDomainMap_ = Dk_1_->getDomainMap();
          DorigImporter_ = rcp_dynamic_cast<CrsMatrixWrap>(Dk_1_)->getCrsMatrix()->getCrsGraph()->getImporter();
        }
        RCP<const Import> ImporterD = ImportFactory::Build(Importer22_->getTargetMap(),Dk_1_->getColMap());
        rcp_dynamic_cast<CrsMatrixWrap>(Dk_1_)->getCrsMatrix()->replaceDomainMapAndImporter(Importer22_->getTargetMap(), ImporterD);
      }

#ifdef HAVE_MUELU_TPETRA
      if ((!Dk_1_T_.is_null()) &&
          (!R11_.is_null()) &&
          (!rcp_dynamic_cast<CrsMatrixWrap>(Dk_1_T_)->getCrsMatrix()->getCrsGraph()->getImporter().is_null()) &&
          (!rcp_dynamic_cast<CrsMatrixWrap>(R11_)->getCrsMatrix()->getCrsGraph()->getImporter().is_null()) &&
          (Dk_1_T_->getColMap()->lib() == Xpetra::UseTpetra) &&
          (R11_->getColMap()->lib() == Xpetra::UseTpetra))
        Dk_1_T_R11_colMapsMatch_ = Dk_1_T_->getColMap()->isSameAs(*R11_->getColMap());
      else
#endif
        Dk_1_T_R11_colMapsMatch_ = false;
      if (Dk_1_T_R11_colMapsMatch_)
        GetOStream(Runtime0) << "RefMaxwell::compute(): D_T and R11 have matching colMaps" << std::endl;

      // Allocate MultiVectors for solve
      allocateMemory(1);

      // apply matvec params
      if (parameterList_.isSublist("matvec params"))
        {
          RCP<ParameterList> matvecParams = rcpFromRef(parameterList_.sublist("matvec params"));
          Maxwell_Utils<SC,LO,GO,NO>::setMatvecParams(*SM_Matrix_, matvecParams);
          Maxwell_Utils<SC,LO,GO,NO>::setMatvecParams(*Dk_1_, matvecParams);
          Maxwell_Utils<SC,LO,GO,NO>::setMatvecParams(*P11_, matvecParams);
          if (!Dk_1_T_.is_null()) Maxwell_Utils<SC,LO,GO,NO>::setMatvecParams(*Dk_1_T_, matvecParams);
          if (!R11_.is_null())         Maxwell_Utils<SC,LO,GO,NO>::setMatvecParams(*R11_, matvecParams);
          if (!ImporterCoarse11_.is_null())   ImporterCoarse11_->setDistributorParameters(matvecParams);
          if (!Importer22_.is_null())  Importer22_->setDistributorParameters(matvecParams);
        }
      if (!ImporterCoarse11_.is_null() && parameterList_.isSublist("refmaxwell: ImporterCoarse11 params")){
        RCP<ParameterList> importerParams = rcpFromRef(parameterList_.sublist("refmaxwell: ImporterCoarse11 params"));
        ImporterCoarse11_->setDistributorParameters(importerParams);
      }
      if (!Importer22_.is_null() && parameterList_.isSublist("refmaxwell: Importer22 params")){
        RCP<ParameterList> importerParams = rcpFromRef(parameterList_.sublist("refmaxwell: Importer22 params"));
        Importer22_->setDistributorParameters(importerParams);
      }
    }

    describe(GetOStream(Runtime0));

#ifdef HAVE_MUELU_CUDA
    if (parameterList_.get<bool>("refmaxwell: cuda profile setup", false)) cudaProfilerStop();
#endif
  }


  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void RefMaxwell<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
  buildNullspace() {
    if(Nullspace_ != null) {
      // no need to do anything - nullspace is built
      TEUCHOS_ASSERT(Nullspace_->getMap()->isCompatible(*(SM_Matrix_->getRowMap())));
    }
    else if(Nullspace_ == null && Coords_ != null) {
      RCP<MultiVector> CoordsSC;
      CoordsSC = Utilities::RealValuedToScalarMultiVector(Coords_);
      Nullspace_ = MultiVectorFactory::Build(SM_Matrix_->getRowMap(),Coords_->getNumVectors());
      Dk_1_->apply(*CoordsSC,*Nullspace_);

      bool normalize = parameterList_.get<bool>("refmaxwell: normalize nullspace", MasterList::getDefault<bool>("refmaxwell: normalize nullspace"));

      coordinateType minLen, maxLen, meanLen;
      if (IsPrint(Statistics2) || normalize){
        // compute edge lengths
        ArrayRCP<ArrayRCP<const Scalar> > localNullspace(Nullspace_->getNumVectors());
        for (size_t i = 0; i < Nullspace_->getNumVectors(); i++)
          localNullspace[i] = Nullspace_->getData(i);
        coordinateType localMinLen = Teuchos::ScalarTraits<coordinateType>::rmax();
        coordinateType localMeanLen = Teuchos::ScalarTraits<coordinateType>::zero();
        coordinateType localMaxLen = Teuchos::ScalarTraits<coordinateType>::zero();
        for (size_t j=0; j < Nullspace_->getMap()->getLocalNumElements(); j++) {
          Scalar lenSC = Teuchos::ScalarTraits<Scalar>::zero();
          for (size_t i=0; i < Nullspace_->getNumVectors(); i++)
            lenSC += localNullspace[i][j]*localNullspace[i][j];
          coordinateType len = Teuchos::as<coordinateType>(Teuchos::ScalarTraits<Scalar>::real(Teuchos::ScalarTraits<Scalar>::squareroot(lenSC)));
          localMinLen = std::min(localMinLen, len);
          localMaxLen = std::max(localMaxLen, len);
          localMeanLen += len;
        }

        RCP<const Teuchos::Comm<int> > comm = Nullspace_->getMap()->getComm();
        MueLu_minAll(comm, localMinLen,  minLen);
        MueLu_sumAll(comm, localMeanLen, meanLen);
        MueLu_maxAll(comm, localMaxLen,  maxLen);
        meanLen /= Nullspace_->getMap()->getGlobalNumElements();
      }

      if (IsPrint(Statistics2)) {
        GetOStream(Statistics0) << "Edge length (min/mean/max): " << minLen << " / " << meanLen << " / " << maxLen << std::endl;
      }

      if (normalize) {
        // normalize the nullspace
        GetOStream(Runtime0) << "RefMaxwell::compute(): normalizing nullspace" << std::endl;

        const Scalar one = Teuchos::ScalarTraits<Scalar>::one();

        Array<Scalar> normsSC(Coords_->getNumVectors(), one / Teuchos::as<Scalar>(meanLen));
        Nullspace_->scale(normsSC());
      }
    }
    else {
      GetOStream(Errors) << "MueLu::RefMaxwell::compute(): either the nullspace or the nodal coordinates must be provided." << std::endl;
    }

    if (skipFirstLevel_) {
      // Nuke the BC edges in nullspace
      Utilities::ZeroDirichletRows(Nullspace_,BCrows11_);
      dump(*Nullspace_, "nullspace.m");
    }
  }


  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void RefMaxwell<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
  determineSubHierarchyCommSizes(bool &doRebalancing, int &rebalanceStriding, int &numProcsCoarseA11, int &numProcsA22) {

    doRebalancing = parameterList_.get<bool>("refmaxwell: subsolves on subcommunicators");
    rebalanceStriding = parameterList_.get<int>("refmaxwell: subsolves striding", -1);
    int numProcs = SM_Matrix_->getDomainMap()->getComm()->getSize();
    if (numProcs == 1) {
      doRebalancing = false;
      return;
    }

#ifdef HAVE_MPI
    if (doRebalancing) {

      {
        // decide on number of ranks for coarse (1, 1) problem

        Level level;
        level.SetFactoryManager(null);
        level.SetLevelID(0);
        level.Set("A",coarseA11_);

        auto repartheurFactory = rcp(new RepartitionHeuristicFactory());
        ParameterList repartheurParams;
        repartheurParams.set("repartition: start level",            0);
        // Setting min == target on purpose.
        int defaultTargetRows = 10000;
        repartheurParams.set("repartition: min rows per proc",      precList11_.get<int>("repartition: target rows per proc", defaultTargetRows));
        repartheurParams.set("repartition: target rows per proc",   precList11_.get<int>("repartition: target rows per proc", defaultTargetRows));
        repartheurParams.set("repartition: min rows per thread",    precList11_.get<int>("repartition: target rows per thread", defaultTargetRows));
        repartheurParams.set("repartition: target rows per thread", precList11_.get<int>("repartition: target rows per thread", defaultTargetRows));
        repartheurParams.set("repartition: max imbalance",          precList11_.get<double>("repartition: max imbalance", 1.1));
        repartheurFactory->SetParameterList(repartheurParams);

        level.Request("number of partitions", repartheurFactory.get());
        repartheurFactory->Build(level);
        numProcsCoarseA11 = level.Get<int>("number of partitions", repartheurFactory.get());
        numProcsCoarseA11 = std::min(numProcsCoarseA11,numProcs);
      }

      {
        // decide on number of ranks for (2, 2) problem

        Level level;
        level.SetFactoryManager(null);
        level.SetLevelID(0);

        level.Set("Map",Dk_1_->getDomainMap());

        auto repartheurFactory = rcp(new RepartitionHeuristicFactory());
        ParameterList repartheurParams;
        repartheurParams.set("repartition: start level",            0);
        repartheurParams.set("repartition: use map",                true);
        // Setting min == target on purpose.
        int defaultTargetRows = 10000;
        repartheurParams.set("repartition: min rows per proc",      precList22_.get<int>("repartition: target rows per proc", defaultTargetRows));
        repartheurParams.set("repartition: target rows per proc",   precList22_.get<int>("repartition: target rows per proc", defaultTargetRows));
        repartheurParams.set("repartition: min rows per thread",    precList22_.get<int>("repartition: target rows per thread", defaultTargetRows));
        repartheurParams.set("repartition: target rows per thread", precList22_.get<int>("repartition: target rows per thread", defaultTargetRows));
        // repartheurParams.set("repartition: max imbalance",        precList22_.get<double>("repartition: max imbalance", 1.1));
        repartheurFactory->SetParameterList(repartheurParams);

        level.Request("number of partitions", repartheurFactory.get());
        repartheurFactory->Build(level);
        numProcsA22 = level.Get<int>("number of partitions", repartheurFactory.get());
        numProcsA22 = std::min(numProcsA22,numProcs);
      }

      if (rebalanceStriding >= 1) {
        TEUCHOS_ASSERT(rebalanceStriding*numProcsCoarseA11<=numProcs);
        TEUCHOS_ASSERT(rebalanceStriding*numProcsA22<=numProcs);
        if (rebalanceStriding*(numProcsCoarseA11+numProcsA22)>numProcs) {
          GetOStream(Warnings0) << "RefMaxwell::compute(): Disabling striding = " << rebalanceStriding << ", since coareA11 needs " << numProcsCoarseA11
                                << " procs and A22 needs " << numProcsA22 << " procs."<< std::endl;
          rebalanceStriding = -1;
        }
      }

      if ((numProcsCoarseA11 < 0) || (numProcsA22 < 0) || (numProcsCoarseA11 + numProcsA22 > numProcs)) {
        GetOStream(Warnings0) << "RefMaxwell::compute(): Disabling rebalancing of subsolves, since partition heuristic resulted "
                              << "in undesirable number of partitions: " << numProcsCoarseA11 << ", " << numProcsA22 << std::endl;
        doRebalancing = false;
      }
    }
#else
    doRebalancing = false;
#endif // HAVE_MPI
  }


  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void RefMaxwell<Scalar,LocalOrdinal,GlobalOrdinal,Node>::buildCoarse11Matrix() {
    RCP<Teuchos::TimeMonitor> tm = getTimer("MueLu RefMaxwell: Build coarse (1,1) matrix");

    const Scalar one = Teuchos::ScalarTraits<Scalar>::one();

    // coarse matrix for P11* (M1 + D1* M2 D1) P11
    RCP<Matrix> temp;
    temp = Xpetra::MatrixMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Multiply(*SM_Matrix_,false,*P11_,false,temp,GetOStream(Runtime0),true,true);
    if (ImporterCoarse11_.is_null())
      coarseA11_ = Xpetra::MatrixMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Multiply(*P11_,true,*temp,false,coarseA11_,GetOStream(Runtime0),true,true);
    else {

      RCP<Matrix> temp2;
      temp2 = Xpetra::MatrixMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Multiply(*P11_,true,*temp,false,temp2,GetOStream(Runtime0),true,true);

      RCP<const Map> map = ImporterCoarse11_->getTargetMap()->removeEmptyProcesses();
      temp2->removeEmptyProcessesInPlace(map);
      if (!temp2.is_null() && temp2->getRowMap().is_null())
        temp2 = Teuchos::null;
      coarseA11_ = temp2;
    }

    if (!disable_addon_) {

      RCP<Matrix> addon;

      if (!coarseA11_.is_null() && Addon_Matrix_.is_null()) {
        // construct addon
        RCP<Teuchos::TimeMonitor> tmAddon = getTimer("MueLu RefMaxwell: Build coarse addon matrix");
        // catch a failure
        TEUCHOS_TEST_FOR_EXCEPTION(invMk_1_invBeta_==Teuchos::null,std::invalid_argument,
                                   "MueLu::RefMaxwell::buildCoarse11Matrix(): Inverse of "
                                   "lumped mass matrix required for add-on (i.e. M0inv_Matrix is null)");

        // coarse matrix for add-on, i.e P11* (M1 D M0inv D^T M1) P11
        RCP<Matrix> Zaux, Z;

        // construct Zaux = M1 P11
        Zaux = Xpetra::MatrixMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Multiply(*Mk_one_,false,*P11_,false,Zaux,GetOStream(Runtime0),true,true);
        // construct Z = D* M1 P11 = D^T Zaux
        Z = Xpetra::MatrixMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Multiply(*Dk_1_,true,*Zaux,false,Z,GetOStream(Runtime0),true,true);

        // construct Z* M0inv Z
        if (invMk_1_invBeta_->getGlobalMaxNumRowEntries()<=1) {
          // We assume that if M0inv has at most one entry per row then
          // these are all diagonal entries.
          RCP<Vector> diag = VectorFactory::Build(invMk_1_invBeta_->getRowMap());
          invMk_1_invBeta_->getLocalDiagCopy(*diag);
	  {
	    ArrayRCP<Scalar> diagVals = diag->getDataNonConst(0);
	    for (size_t j=0; j < diag->getMap()->getLocalNumElements(); j++) {
	      diagVals[j] = Teuchos::ScalarTraits<Scalar>::squareroot(diagVals[j]);
	    }
	  }
          if (Z->getRowMap()->isSameAs(*(diag->getMap())))
            Z->leftScale(*diag);
          else {
            RCP<Import> importer = ImportFactory::Build(diag->getMap(),Z->getRowMap());
            RCP<Vector> diag2 = VectorFactory::Build(Z->getRowMap());
            diag2->doImport(*diag,*importer,Xpetra::INSERT);
            Z->leftScale(*diag2);
          }
          addon = Xpetra::MatrixMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Multiply(*Z,true,*Z,false,addon,GetOStream(Runtime0),true,true);
        } else if (parameterList_.get<bool>("rap: triple product", false) == false) {
          RCP<Matrix> C2;
          // construct C2 = M0inv Z
          C2 = Xpetra::MatrixMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Multiply(*invMk_1_invBeta_,false,*Z,false,C2,GetOStream(Runtime0),true,true);
          // construct Matrix2 = Z* M0inv Z = Z* C2
          addon = Xpetra::MatrixMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Multiply(*Z,true,*C2,false,addon,GetOStream(Runtime0),true,true);
        } else {
          addon = MatrixFactory::Build(Z->getDomainMap());
          // construct Matrix2 = Z* M0inv Z
          Xpetra::TripleMatrixMultiply<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
            MultiplyRAP(*Z, true, *invMk_1_invBeta_, false, *Z, false, *addon, true, true);
        }
        // Should we keep the addon for next setup?
        if (enable_reuse_)
          Addon_Matrix_ = addon;
      } else
        addon = Addon_Matrix_;

      if (!coarseA11_.is_null()) {
        // add matrices together
        RCP<Matrix> newCoarseA11;
        Xpetra::MatrixMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::TwoMatrixAdd(*coarseA11_,false,one,*addon,false,one,newCoarseA11,GetOStream(Runtime0));
        newCoarseA11->fillComplete();
        coarseA11_ = newCoarseA11;
      }
    }

    if (!coarseA11_.is_null() && !skipFirstLevel_) {
      ArrayRCP<bool> coarseA11BCrows;
      coarseA11BCrows.resize(coarseA11_->getRowMap()->getLocalNumElements());
      size_t dim = Nullspace_->getNumVectors();
      for (size_t i = 0; i < BCdomain22_.size(); i++)
        for (size_t k = 0; k < dim; k++)
          coarseA11BCrows[i*dim+k] = BCdomain22_(i);
      magnitudeType rowSumTol = parameterList_.get<double>("refmaxwell: row sum drop tol (1,1)");
      if (rowSumTol > 0.)
        Utilities::ApplyRowSumCriterion(*coarseA11_, rowSumTol, coarseA11BCrows);
      if (applyBCsToCoarse11_)
        Utilities::ApplyOAZToMatrixRows(coarseA11_, coarseA11BCrows);
    }

    if (!coarseA11_.is_null()) {
      // If we already applied BCs to A_nodal, we likely do not need
      // to fix up coarseA11.
      // If we did not apply BCs to A_nodal, we now need to correct
      // the zero diagonals of coarseA11, since we did nuke the nullspace.

      bool fixZeroDiagonal = !applyBCsToAnodal_;
      if (precList11_.isParameter("rap: fix zero diagonals"))
        fixZeroDiagonal = precList11_.get<bool>("rap: fix zero diagonals");

      if (fixZeroDiagonal) {
        magnitudeType threshold = 1e-16;
        Scalar replacement = 1.0;
        if (precList11_.isType<magnitudeType>("rap: fix zero diagonals threshold"))
          threshold = precList11_.get<magnitudeType>("rap: fix zero diagonals threshold");
        else if (precList11_.isType<double>("rap: fix zero diagonals threshold"))
          threshold = Teuchos::as<magnitudeType>(precList11_.get<double>("rap: fix zero diagonals threshold"));
        if (precList11_.isType<double>("rap: fix zero diagonals replacement"))
          replacement = Teuchos::as<Scalar>(precList11_.get<double>("rap: fix zero diagonals replacement"));
        Xpetra::MatrixUtils<SC,LO,GO,NO>::CheckRepairMainDiagonal(coarseA11_, true, GetOStream(Warnings1), threshold, replacement);
      }

      // Set block size
      size_t dim = Nullspace_->getNumVectors();
      coarseA11_->SetFixedBlockSize(dim);
      coarseA11_->setObjectLabel("RefMaxwell coarse (1,1)");
    }
  }


  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void RefMaxwell<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
  rebalanceCoarse11Matrix (const int rebalanceStriding, const int numProcsCoarseA11) {
#ifdef HAVE_MPI
    // rebalance coarseA11
    RCP<Teuchos::TimeMonitor> tm = getTimer("MueLu RefMaxwell: Rebalance coarseA11");

    Level fineLevel, coarseLevel;
    fineLevel.SetFactoryManager(null);
    coarseLevel.SetFactoryManager(null);
    coarseLevel.SetPreviousLevel(rcpFromRef(fineLevel));
    fineLevel.SetLevelID(0);
    coarseLevel.SetLevelID(1);
    coarseLevel.Set("A",coarseA11_);
    coarseLevel.Set("P",P11_);
    coarseLevel.Set("Coordinates",CoordsCoarse11_);
    if (!NullspaceCoarse11_.is_null())
      coarseLevel.Set("Nullspace",NullspaceCoarse11_);
    coarseLevel.Set("number of partitions", numProcsCoarseA11);
    coarseLevel.Set("repartition: heuristic target rows per process", 1000);

    coarseLevel.setlib(coarseA11_->getDomainMap()->lib());
    fineLevel.setlib(coarseA11_->getDomainMap()->lib());
    coarseLevel.setObjectLabel("RefMaxwell coarse (1,1)");
    fineLevel.setObjectLabel("RefMaxwell coarse (1,1)");

    std::string partName = precList11_.get<std::string>("repartition: partitioner", "zoltan2");
    RCP<Factory> partitioner;
    if (partName == "zoltan") {
#ifdef HAVE_MUELU_ZOLTAN
      partitioner = rcp(new ZoltanInterface());
      // NOTE: ZoltanInteface ("zoltan") does not support external parameters through ParameterList
      // partitioner->SetFactory("number of partitions", repartheurFactory);
#else
      throw Exceptions::RuntimeError("Zoltan interface is not available");
#endif
    } else if (partName == "zoltan2") {
#ifdef HAVE_MUELU_ZOLTAN2
      partitioner = rcp(new Zoltan2Interface());
      ParameterList partParams;
      RCP<const ParameterList> partpartParams = rcp(new ParameterList(precList11_.sublist("repartition: params", false)));
      partParams.set("ParameterList", partpartParams);
      partitioner->SetParameterList(partParams);
      // partitioner->SetFactory("number of partitions", repartheurFactory);
#else
      throw Exceptions::RuntimeError("Zoltan2 interface is not available");
#endif
    }

    auto repartFactory = rcp(new RepartitionFactory());
    ParameterList repartParams;
    repartParams.set("repartition: print partition distribution", precList11_.get<bool>("repartition: print partition distribution", false));
    repartParams.set("repartition: remap parts", precList11_.get<bool>("repartition: remap parts", true));
    if (rebalanceStriding >= 1) {
      bool acceptPart = (SM_Matrix_->getDomainMap()->getComm()->getRank() % rebalanceStriding) == 0;
      if (SM_Matrix_->getDomainMap()->getComm()->getRank() >= numProcsCoarseA11*rebalanceStriding)
        acceptPart = false;
      repartParams.set("repartition: remap accept partition", acceptPart);
    }
    repartFactory->SetParameterList(repartParams);
    // repartFactory->SetFactory("number of partitions", repartheurFactory);
    repartFactory->SetFactory("Partition", partitioner);

    auto newP = rcp(new RebalanceTransferFactory());
    ParameterList newPparams;
    newPparams.set("type", "Interpolation");
    newPparams.set("repartition: rebalance P and R", precList11_.get<bool>("repartition: rebalance P and R", false));
    newPparams.set("repartition: use subcommunicators", true);
    newPparams.set("repartition: rebalance Nullspace", !NullspaceCoarse11_.is_null());
    newP->SetFactory("Coordinates", NoFactory::getRCP());
    if (!NullspaceCoarse11_.is_null())
      newP->SetFactory("Nullspace", NoFactory::getRCP());
    newP->SetParameterList(newPparams);
    newP->SetFactory("Importer", repartFactory);

    auto newA = rcp(new RebalanceAcFactory());
    ParameterList rebAcParams;
    rebAcParams.set("repartition: use subcommunicators", true);
    newA->SetParameterList(rebAcParams);
    newA->SetFactory("Importer", repartFactory);

    coarseLevel.Request("P", newP.get());
    coarseLevel.Request("Importer", repartFactory.get());
    coarseLevel.Request("A", newA.get());
    coarseLevel.Request("Coordinates", newP.get());
    if (!NullspaceCoarse11_.is_null())
      coarseLevel.Request("Nullspace", newP.get());
    repartFactory->Build(coarseLevel);

    if (!precList11_.get<bool>("repartition: rebalance P and R", false))
      ImporterCoarse11_ = coarseLevel.Get< RCP<const Import> >("Importer", repartFactory.get());
    P11_ = coarseLevel.Get< RCP<Matrix> >("P", newP.get());
    coarseA11_ = coarseLevel.Get< RCP<Matrix> >("A", newA.get());
    CoordsCoarse11_ = coarseLevel.Get< RCP<RealValuedMultiVector> >("Coordinates", newP.get());
    if (!NullspaceCoarse11_.is_null())
      NullspaceCoarse11_ = coarseLevel.Get< RCP<MultiVector> >("Nullspace", newP.get());

    if (!coarseA11_.is_null()) {
      // Set block size
      size_t dim = Nullspace_->getNumVectors();
      coarseA11_->SetFixedBlockSize(dim);
      coarseA11_->setObjectLabel("RefMaxwell coarse (1,1)");
    }

    coarseA11_AP_reuse_data_ = Teuchos::null;
    coarseA11_RAP_reuse_data_ = Teuchos::null;

    if (!disable_addon_ && enable_reuse_) {
      // Rebalance the addon for next setup
      RCP<const Import> ImporterCoarse11 = coarseLevel.Get< RCP<const Import> >("Importer", repartFactory.get());
      RCP<const Map> targetMap = ImporterCoarse11->getTargetMap();
      ParameterList XpetraList;
      XpetraList.set("Restrict Communicator",true);
      Addon_Matrix_ = MatrixFactory::Build(Addon_Matrix_, *ImporterCoarse11, *ImporterCoarse11, targetMap, targetMap, rcp(&XpetraList,false));
    }
#endif
  }

  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void RefMaxwell<Scalar,LocalOrdinal,GlobalOrdinal,Node>::build22Matrix (const bool reuse, const bool doRebalancing, const int rebalanceStriding, const int numProcsA22) {
        
    if (!reuse) { // build fine grid operator for (2,2)-block, D^T SM D  (aka TMT)
      RCP<Teuchos::TimeMonitor> tm = getTimer("MueLu RefMaxwell: Build A22");

      Level fineLevel, coarseLevel;
      fineLevel.SetFactoryManager(null);
      coarseLevel.SetFactoryManager(null);
      coarseLevel.SetPreviousLevel(rcpFromRef(fineLevel));
      fineLevel.SetLevelID(0);
      coarseLevel.SetLevelID(1);
      fineLevel.Set("A",SM_Matrix_);
      coarseLevel.Set("P",Dk_1_);
      coarseLevel.Set("Coordinates",Coords_);

      coarseLevel.setlib(SM_Matrix_->getDomainMap()->lib());
      fineLevel.setlib(SM_Matrix_->getDomainMap()->lib());
      coarseLevel.setObjectLabel("RefMaxwell (2,2)");
      fineLevel.setObjectLabel("RefMaxwell (2,2)");

      RCP<RAPFactory> rapFact = rcp(new RAPFactory());
      ParameterList rapList = *(rapFact->GetValidParameterList());
      rapList.set("transpose: use implicit", true);
      rapList.set("rap: fix zero diagonals", parameterList_.get<bool>("rap: fix zero diagonals", true));
      rapList.set("rap: fix zero diagonals threshold", parameterList_.get<double>("rap: fix zero diagonals threshold", Teuchos::ScalarTraits<double>::eps()));
      rapList.set("rap: triple product", parameterList_.get<bool>("rap: triple product", false));
      rapFact->SetParameterList(rapList);

      if (!A22_AP_reuse_data_.is_null()) {
        coarseLevel.AddKeepFlag("AP reuse data", rapFact.get());
        coarseLevel.Set<Teuchos::RCP<Teuchos::ParameterList> >("AP reuse data", A22_AP_reuse_data_, rapFact.get());
      }
      if (!A22_RAP_reuse_data_.is_null()) {
        coarseLevel.AddKeepFlag("RAP reuse data", rapFact.get());
        coarseLevel.Set<Teuchos::RCP<Teuchos::ParameterList> >("RAP reuse data", A22_RAP_reuse_data_, rapFact.get());
      }

#ifdef HAVE_MPI
      if (doRebalancing) {

        coarseLevel.Set("number of partitions", numProcsA22);
        coarseLevel.Set("repartition: heuristic target rows per process", 1000);

        std::string partName = precList22_.get<std::string>("repartition: partitioner", "zoltan2");
        RCP<Factory> partitioner;
        if (partName == "zoltan") {
#ifdef HAVE_MUELU_ZOLTAN
          partitioner = rcp(new ZoltanInterface());
          partitioner->SetFactory("A", rapFact);
          // partitioner->SetFactory("number of partitions", repartheurFactory);
          // NOTE: ZoltanInteface ("zoltan") does not support external parameters through ParameterList
#else
          throw Exceptions::RuntimeError("Zoltan interface is not available");
#endif
        } else if (partName == "zoltan2") {
#ifdef HAVE_MUELU_ZOLTAN2
          partitioner = rcp(new Zoltan2Interface());
          ParameterList partParams;
          RCP<const ParameterList> partpartParams = rcp(new ParameterList(precList22_.sublist("repartition: params", false)));
          partParams.set("ParameterList", partpartParams);
          partitioner->SetParameterList(partParams);
          partitioner->SetFactory("A", rapFact);
          // partitioner->SetFactory("number of partitions", repartheurFactory);
#else
          throw Exceptions::RuntimeError("Zoltan2 interface is not available");
#endif
        }

        auto repartFactory = rcp(new RepartitionFactory());
        ParameterList repartParams;
        repartParams.set("repartition: print partition distribution", precList22_.get<bool>("repartition: print partition distribution", false));
        repartParams.set("repartition: remap parts", precList22_.get<bool>("repartition: remap parts", true));
        if (rebalanceStriding >= 1) {
          bool acceptPart = ((SM_Matrix_->getDomainMap()->getComm()->getSize()-1-SM_Matrix_->getDomainMap()->getComm()->getRank()) % rebalanceStriding) == 0;
          if (SM_Matrix_->getDomainMap()->getComm()->getSize()-1-SM_Matrix_->getDomainMap()->getComm()->getRank() >= numProcsA22*rebalanceStriding)
            acceptPart = false;
          if (acceptPart)
            TEUCHOS_ASSERT(coarseA11_.is_null());
          repartParams.set("repartition: remap accept partition", acceptPart);
        } else
          repartParams.set("repartition: remap accept partition", coarseA11_.is_null());
        repartFactory->SetParameterList(repartParams);
        repartFactory->SetFactory("A", rapFact);
        // repartFactory->SetFactory("number of partitions", repartheurFactory);
        repartFactory->SetFactory("Partition", partitioner);

        auto newP = rcp(new RebalanceTransferFactory());
        ParameterList newPparams;
        newPparams.set("type", "Interpolation");
        newPparams.set("repartition: rebalance P and R", precList22_.get<bool>("repartition: rebalance P and R", false));
        newPparams.set("repartition: use subcommunicators", true);
        newPparams.set("repartition: rebalance Nullspace", false);
        newP->SetFactory("Coordinates", NoFactory::getRCP());
        newP->SetParameterList(newPparams);
        newP->SetFactory("Importer", repartFactory);

        auto newA = rcp(new RebalanceAcFactory());
        ParameterList rebAcParams;
        rebAcParams.set("repartition: use subcommunicators", true);
        newA->SetParameterList(rebAcParams);
        newA->SetFactory("A", rapFact);
        newA->SetFactory("Importer", repartFactory);

        coarseLevel.Request("P", newP.get());
        coarseLevel.Request("Importer", repartFactory.get());
        coarseLevel.Request("A", newA.get());
        coarseLevel.Request("Coordinates", newP.get());
        rapFact->Build(fineLevel,coarseLevel);
        repartFactory->Build(coarseLevel);

        if (!precList22_.get<bool>("repartition: rebalance P and R", false))
          Importer22_ = coarseLevel.Get< RCP<const Import> >("Importer", repartFactory.get());
        Dk_1_ = coarseLevel.Get< RCP<Matrix> >("P", newP.get());
        A22_ = coarseLevel.Get< RCP<Matrix> >("A", newA.get());
        Coords_ = coarseLevel.Get< RCP<RealValuedMultiVector> >("Coordinates", newP.get());

      } else
#endif // HAVE_MPI
        {
          coarseLevel.Request("A", rapFact.get());
          if (enable_reuse_) {
            coarseLevel.Request("AP reuse data", rapFact.get());
            coarseLevel.Request("RAP reuse data", rapFact.get());
          }

          A22_ = coarseLevel.Get< RCP<Matrix> >("A", rapFact.get());

          if (enable_reuse_) {
            if (coarseLevel.IsAvailable("AP reuse data", rapFact.get()))
              A22_AP_reuse_data_ = coarseLevel.Get< RCP<ParameterList> >("AP reuse data", rapFact.get());
            if (coarseLevel.IsAvailable("RAP reuse data", rapFact.get()))
              A22_RAP_reuse_data_ = coarseLevel.Get< RCP<ParameterList> >("RAP reuse data", rapFact.get());
          }
        }
    } else {
      RCP<Teuchos::TimeMonitor> tm = getTimer("MueLu RefMaxwell: Build A22");
      if (Importer22_.is_null()) {
        RCP<Matrix> temp;
        temp = Xpetra::MatrixMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Multiply(*SM_Matrix_,false,*Dk_1_,false,temp,GetOStream(Runtime0),true,true);
        if (!implicitTranspose_)
          A22_ = Xpetra::MatrixMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Multiply(*Dk_1_T_,false,*temp,false,A22_,GetOStream(Runtime0),true,true);
        else
          A22_ = Xpetra::MatrixMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Multiply(*Dk_1_,true,*temp,false,A22_,GetOStream(Runtime0),true,true);
      } else {
        // we replaced domain map and importer on D, reverse that
        RCP<const Import> Dimporter = rcp_dynamic_cast<CrsMatrixWrap>(Dk_1_)->getCrsMatrix()->getCrsGraph()->getImporter();
        rcp_dynamic_cast<CrsMatrixWrap>(Dk_1_)->getCrsMatrix()->replaceDomainMapAndImporter(DorigDomainMap_, DorigImporter_);

        RCP<Matrix> temp, temp2;
        temp = Xpetra::MatrixMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Multiply(*SM_Matrix_,false,*Dk_1_,false,temp,GetOStream(Runtime0),true,true);
        if (!implicitTranspose_)
          temp2 = Xpetra::MatrixMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Multiply(*Dk_1_T_,false,*temp,false,temp2,GetOStream(Runtime0),true,true);
        else
          temp2 = Xpetra::MatrixMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Multiply(*Dk_1_,true,*temp,false,temp2,GetOStream(Runtime0),true,true);

        // and back again
        rcp_dynamic_cast<CrsMatrixWrap>(Dk_1_)->getCrsMatrix()->replaceDomainMapAndImporter(Importer22_->getTargetMap(), Dimporter);

        ParameterList XpetraList;
        XpetraList.set("Restrict Communicator",true);
        XpetraList.set("Timer Label","MueLu::RebalanceA22");
        RCP<const Map> targetMap = Importer22_->getTargetMap();
        A22_ = MatrixFactory::Build(temp2, *Importer22_, *Importer22_, targetMap, targetMap, rcp(&XpetraList,false));
      }
    }

    if (!A22_.is_null()) {
      dump(*A22_, "A22.m");
      A22_->setObjectLabel("RefMaxwell (2,2)");
    }
  }


  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void RefMaxwell<Scalar,LocalOrdinal,GlobalOrdinal,Node>::setFineLevelSmoother() {

    Level level;
    RCP<MueLu::FactoryManagerBase> factoryHandler = rcp(new FactoryManager());
    level.SetFactoryManager(factoryHandler);
    level.SetLevelID(0);
    level.setObjectLabel("RefMaxwell (1,1)");
    level.Set("A",SM_Matrix_);
    level.setlib(SM_Matrix_->getDomainMap()->lib());
    // For Hiptmair
    level.Set("NodeMatrix", A22_);
    level.Set("D0", Dk_1_);

    if ((parameterList_.get<std::string>("smoother: pre type") != "NONE") && (parameterList_.get<std::string>("smoother: post type") != "NONE")) {
      std::string preSmootherType = parameterList_.get<std::string>("smoother: pre type");
      std::string postSmootherType = parameterList_.get<std::string>("smoother: post type");

      ParameterList preSmootherList, postSmootherList;
      if (parameterList_.isSublist("smoother: pre params"))
        preSmootherList = parameterList_.sublist("smoother: pre params");
      if (parameterList_.isSublist("smoother: post params"))
        postSmootherList = parameterList_.sublist("smoother: post params");

      RCP<SmootherPrototype> preSmootherPrototype = rcp(new TrilinosSmoother(preSmootherType, preSmootherList));
      RCP<SmootherPrototype> postSmootherPrototype = rcp(new TrilinosSmoother(postSmootherType, postSmootherList));
      RCP<SmootherFactory> smootherFact = rcp(new SmootherFactory(preSmootherPrototype, postSmootherPrototype));

      level.Request("PreSmoother",smootherFact.get());
      level.Request("PostSmoother",smootherFact.get());
      if (enable_reuse_) {
        ParameterList smootherFactoryParams;
        smootherFactoryParams.set("keep smoother data", true);
        smootherFact->SetParameterList(smootherFactoryParams);
        level.Request("PreSmoother data", smootherFact.get());
        level.Request("PostSmoother data", smootherFact.get());
        if (!PreSmootherData_.is_null())
          level.Set("PreSmoother data", PreSmootherData_, smootherFact.get());
        if (!PostSmootherData_.is_null())
          level.Set("PostSmoother data", PostSmootherData_, smootherFact.get());
      }
      smootherFact->Build(level);
      PreSmoother_ = level.Get<RCP<SmootherBase> >("PreSmoother",smootherFact.get());
      PostSmoother_ = level.Get<RCP<SmootherBase> >("PostSmoother",smootherFact.get());
      if (enable_reuse_) {
        PreSmootherData_ = level.Get<RCP<SmootherPrototype> >("PreSmoother data",smootherFact.get());
        PostSmootherData_ = level.Get<RCP<SmootherPrototype> >("PostSmoother data",smootherFact.get());
      }
    } else {
      std::string smootherType = parameterList_.get<std::string>("smoother: type");

      RCP<SmootherPrototype> smootherPrototype = rcp(new TrilinosSmoother(smootherType, smootherList_));
      RCP<SmootherFactory> smootherFact = rcp(new SmootherFactory(smootherPrototype));
      level.Request("PreSmoother",smootherFact.get());
      if (enable_reuse_) {
        ParameterList smootherFactoryParams;
        smootherFactoryParams.set("keep smoother data", true);
        smootherFact->SetParameterList(smootherFactoryParams);
        level.Request("PreSmoother data", smootherFact.get());
        if (!PreSmootherData_.is_null())
          level.Set("PreSmoother data", PreSmootherData_, smootherFact.get());
      }
      smootherFact->Build(level);
      PreSmoother_ = level.Get<RCP<SmootherBase> >("PreSmoother",smootherFact.get());
      PostSmoother_ = PreSmoother_;
      if (enable_reuse_)
        PreSmootherData_ = level.Get<RCP<SmootherPrototype> >("PreSmoother data",smootherFact.get());
    }

  }


  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void RefMaxwell<Scalar,LocalOrdinal,GlobalOrdinal,Node>::allocateMemory(int numVectors) const {

    RCP<Teuchos::TimeMonitor> tmAlloc = getTimer("MueLu RefMaxwell: Allocate MVs");

    if (!R11_.is_null())
      P11res_    = MultiVectorFactory::Build(R11_->getRangeMap(), numVectors);
    else
      P11res_    = MultiVectorFactory::Build(P11_->getDomainMap(), numVectors);
    P11res_->setObjectLabel("P11res");
    if (Dk_1_T_R11_colMapsMatch_) {
      DTR11Tmp_ = MultiVectorFactory::Build(R11_->getColMap(), numVectors);
      DTR11Tmp_->setObjectLabel("DTR11Tmp");
    }
    if (!ImporterCoarse11_.is_null()) {
      P11resTmp_ = MultiVectorFactory::Build(ImporterCoarse11_->getTargetMap(), numVectors);
      P11resTmp_->setObjectLabel("P11resTmp");
      P11x_      = MultiVectorFactory::Build(ImporterCoarse11_->getTargetMap(), numVectors);
    } else
      P11x_      = MultiVectorFactory::Build(P11_->getDomainMap(), numVectors);
    P11x_->setObjectLabel("P11x");
    if (!Dk_1_T_.is_null())
      Dres_     = MultiVectorFactory::Build(Dk_1_T_->getRangeMap(), numVectors);
    else
      Dres_     = MultiVectorFactory::Build(Dk_1_->getDomainMap(), numVectors);
    Dres_->setObjectLabel("Dres");
    if (!Importer22_.is_null()) {
      DresTmp_ = MultiVectorFactory::Build(Importer22_->getTargetMap(), numVectors);
      DresTmp_->setObjectLabel("DresTmp");
      Dx_      = MultiVectorFactory::Build(Importer22_->getTargetMap(), numVectors);
    } else
      Dx_      = MultiVectorFactory::Build(Dk_1_->getDomainMap(), numVectors);
    Dx_->setObjectLabel("Dx");
    if (!coarseA11_.is_null()) {
      if (!ImporterCoarse11_.is_null() && !implicitTranspose_)
        P11resSubComm_ = MultiVectorFactory::Build(P11resTmp_, Teuchos::View);
      else
        P11resSubComm_ = MultiVectorFactory::Build(P11res_, Teuchos::View);
      P11resSubComm_->replaceMap(coarseA11_->getRangeMap());
      P11resSubComm_->setObjectLabel("P11resSubComm");

      P11xSubComm_ = MultiVectorFactory::Build(P11x_, Teuchos::View);
      P11xSubComm_->replaceMap(coarseA11_->getDomainMap());
      P11xSubComm_->setObjectLabel("P11xSubComm");
    }
    if (!A22_.is_null()) {
      if (!Importer22_.is_null() && !implicitTranspose_)
        DresSubComm_ = MultiVectorFactory::Build(DresTmp_, Teuchos::View);
      else
        DresSubComm_ = MultiVectorFactory::Build(Dres_, Teuchos::View);
      DresSubComm_->replaceMap(A22_->getRangeMap());
      DresSubComm_->setObjectLabel("DresSubComm");

      DxSubComm_ = MultiVectorFactory::Build(Dx_, Teuchos::View);
      DxSubComm_->replaceMap(A22_->getDomainMap());
      DxSubComm_->setObjectLabel("DxSubComm");
    }
    residual_  = MultiVectorFactory::Build(SM_Matrix_->getDomainMap(), numVectors);
    residual_->setObjectLabel("residual");
  }


  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void RefMaxwell<Scalar,LocalOrdinal,GlobalOrdinal,Node>::dump(const Matrix& A, std::string name) const {
    if (dump_matrices_) {
      GetOStream(Runtime0) << "Dumping to " << name << std::endl;
      Xpetra::IO<SC, LO, GO, NO>::Write(name, A);
    }
  }


  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void RefMaxwell<Scalar,LocalOrdinal,GlobalOrdinal,Node>::dump(const MultiVector& X, std::string name) const {
    if (dump_matrices_) {
      GetOStream(Runtime0) << "Dumping to " << name << std::endl;
      Xpetra::IO<SC, LO, GO, NO>::Write(name, X);
    }
  }


  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void RefMaxwell<Scalar,LocalOrdinal,GlobalOrdinal,Node>::dumpCoords(const RealValuedMultiVector& X, std::string name) const {
    if (dump_matrices_) {
      GetOStream(Runtime0) << "Dumping to " << name << std::endl;
      Xpetra::IO<coordinateType, LO, GO, NO>::Write(name, X);
    }
  }


  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void RefMaxwell<Scalar,LocalOrdinal,GlobalOrdinal,Node>::dump(const Teuchos::ArrayRCP<bool>& v, std::string name) const {
    if (dump_matrices_) {
      GetOStream(Runtime0) << "Dumping to " << name << std::endl;
      std::ofstream out(name);
      for (size_t i = 0; i < Teuchos::as<size_t>(v.size()); i++)
        out << v[i] << "\n";
    }
  }


  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void RefMaxwell<Scalar,LocalOrdinal,GlobalOrdinal,Node>::dump(const Kokkos::View<bool*, typename Node::device_type>& v, std::string name) const {
    if (dump_matrices_) {
      GetOStream(Runtime0) << "Dumping to " << name << std::endl;
      std::ofstream out(name);
      auto vH = Kokkos::create_mirror_view (v);
          Kokkos::deep_copy(vH , v);
          for (size_t i = 0; i < vH.size(); i++)
            out << vH[i] << "\n";
    }
  }


  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  Teuchos::RCP<Teuchos::TimeMonitor> RefMaxwell<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getTimer(std::string name, RCP<const Teuchos::Comm<int> > comm) const {
    if (IsPrint(Timings)) {
      if (!syncTimers_)
        return Teuchos::rcp(new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer(name)));
      else {
        if (comm.is_null())
          return Teuchos::rcp(new Teuchos::SyncTimeMonitor(*Teuchos::TimeMonitor::getNewTimer(name), SM_Matrix_->getRowMap()->getComm().ptr()));
        else
          return Teuchos::rcp(new Teuchos::SyncTimeMonitor(*Teuchos::TimeMonitor::getNewTimer(name), comm.ptr()));
      }
    } else
      return Teuchos::null;
  }


  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void RefMaxwell<Scalar,LocalOrdinal,GlobalOrdinal,Node>::buildNodalProlongator(const Teuchos::RCP<Matrix> &A_nodal,
                                                                                 Teuchos::RCP<Matrix> &P_nodal,
                                                                                 Teuchos::RCP<MultiVector> &Nullspace_nodal,
                                                                                 Teuchos::RCP<RealValuedMultiVector> &Coords_nodal) const {
    // build prolongator: algorithm 1 in the reference paper
    // First, build nodal unsmoothed prolongator using the matrix A_nodal

    const SC SC_ONE = Teuchos::ScalarTraits<SC>::one();

    {
      Level fineLevel, coarseLevel;
      fineLevel.SetFactoryManager(null);
      coarseLevel.SetFactoryManager(null);
      coarseLevel.SetPreviousLevel(rcpFromRef(fineLevel));
      fineLevel.SetLevelID(0);
      coarseLevel.SetLevelID(1);
      fineLevel.Set("A",A_nodal);
      fineLevel.Set("Coordinates",Coords_);
      fineLevel.Set("DofsPerNode",1);
      coarseLevel.setlib(A_nodal->getDomainMap()->lib());
      fineLevel.setlib(A_nodal->getDomainMap()->lib());
      coarseLevel.setObjectLabel("RefMaxwell (1,1) A_nodal");
      fineLevel.setObjectLabel("RefMaxwell (1,1) A_nodal");

      LocalOrdinal NSdim = 1;
      RCP<MultiVector> nullSpace = MultiVectorFactory::Build(A_nodal->getRowMap(),NSdim);
      nullSpace->putScalar(SC_ONE);
      fineLevel.Set("Nullspace",nullSpace);

      std::string algo = parameterList_.get<std::string>("multigrid algorithm");

      RCP<Factory> amalgFact, dropFact, UncoupledAggFact, coarseMapFact, TentativePFact, Tfact, SaPFact;
      amalgFact = rcp(new AmalgamationFactory());
      coarseMapFact = rcp(new CoarseMapFactory());
      Tfact = rcp(new CoordinatesTransferFactory());
      if (useKokkos_) {
        dropFact = rcp(new CoalesceDropFactory_kokkos());
        UncoupledAggFact = rcp(new UncoupledAggregationFactory_kokkos());
        TentativePFact = rcp(new TentativePFactory_kokkos());
        if (algo == "sa")
          SaPFact = rcp(new SaPFactory_kokkos());
      } else {
        dropFact = rcp(new CoalesceDropFactory());
        UncoupledAggFact = rcp(new UncoupledAggregationFactory());
        TentativePFact = rcp(new TentativePFactory());
        if (algo == "sa")
          SaPFact = rcp(new SaPFactory());
      }
      dropFact->SetFactory("UnAmalgamationInfo", amalgFact);

      double dropTol = parameterList_.get<double>("aggregation: drop tol");
      std::string dropScheme = parameterList_.get<std::string>("aggregation: drop scheme");
      std::string distLaplAlgo = parameterList_.get<std::string>("aggregation: distance laplacian algo");
      dropFact->SetParameter("aggregation: drop tol",Teuchos::ParameterEntry(dropTol));
      dropFact->SetParameter("aggregation: drop scheme",Teuchos::ParameterEntry(dropScheme));
      if (!useKokkos_)
        dropFact->SetParameter("aggregation: distance laplacian algo",Teuchos::ParameterEntry(distLaplAlgo));

      UncoupledAggFact->SetFactory("Graph", dropFact);
      int minAggSize = parameterList_.get<int>("aggregation: min agg size");
      UncoupledAggFact->SetParameter("aggregation: min agg size",Teuchos::ParameterEntry(minAggSize));
      int maxAggSize = parameterList_.get<int>("aggregation: max agg size");
      UncoupledAggFact->SetParameter("aggregation: max agg size",Teuchos::ParameterEntry(maxAggSize));

      coarseMapFact->SetFactory("Aggregates", UncoupledAggFact);

      TentativePFact->SetFactory("Aggregates", UncoupledAggFact);
      TentativePFact->SetFactory("UnAmalgamationInfo", amalgFact);
      TentativePFact->SetFactory("CoarseMap", coarseMapFact);

      Tfact->SetFactory("Aggregates", UncoupledAggFact);
      Tfact->SetFactory("CoarseMap", coarseMapFact);

      if (algo == "sa") {
        SaPFact->SetFactory("P", TentativePFact);
        coarseLevel.Request("P", SaPFact.get());
      } else
        coarseLevel.Request("P",TentativePFact.get());
      coarseLevel.Request("Nullspace",TentativePFact.get());
      coarseLevel.Request("Coordinates",Tfact.get());

      RCP<AggregationExportFactory> aggExport;
      bool exportVizData = parameterList_.get<bool>("aggregation: export visualization data");
      if (exportVizData) {
        aggExport = rcp(new AggregationExportFactory());
        ParameterList aggExportParams;
        aggExportParams.set("aggregation: output filename", "aggs.vtk");
        aggExportParams.set("aggregation: output file: agg style", "Jacks");
        aggExport->SetParameterList(aggExportParams);

        aggExport->SetFactory("Aggregates", UncoupledAggFact);
        aggExport->SetFactory("UnAmalgamationInfo", amalgFact);
        fineLevel.Request("Aggregates",UncoupledAggFact.get());
        fineLevel.Request("UnAmalgamationInfo",amalgFact.get());
      }

      if (algo == "sa")
        coarseLevel.Get("P",P_nodal,SaPFact.get());
      else
        coarseLevel.Get("P",P_nodal,TentativePFact.get());
      coarseLevel.Get("Nullspace",Nullspace_nodal,TentativePFact.get());
      coarseLevel.Get("Coordinates",Coords_nodal,Tfact.get());


      if (exportVizData)
        aggExport->Build(fineLevel, coarseLevel);
    }
    dump(*P_nodal, "P_nodal.m");
    dump(*Nullspace_nodal, "Nullspace_nodal.m");

  }

  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void RefMaxwell<Scalar,LocalOrdinal,GlobalOrdinal,Node>::buildFaceProlongator(const Teuchos::RCP<Matrix> &A_nodal,
                                                                                Teuchos::RCP<Matrix> &specialP,
                                                                                Teuchos::RCP<MultiVector> &specialNullspace,
                                                                                Teuchos::RCP<RealValuedMultiVector> &specialCoords) const {

    RCP<Matrix> P_nodal;
    RCP<CrsMatrix> P_nodal_imported;
    RCP<MultiVector> Nullspace_nodal;
    buildNodalProlongator(A_nodal, P_nodal, Nullspace_nodal, specialCoords);

    TEUCHOS_ASSERT(spaceNumber_ == 2);
    RCP<const Map> faceMap = Dk_1_->getRowMap();
    size_t numLocalRows = faceMap->getLocalNumEntries();

    // Import off-rank rows of P_nodal into P_nodal_imported
    int numProcs = P_nodal->getDomainMap()->getComm()->getSize();
    if (numProcs > 1) {
      RCP<CrsMatrixWrap> P_nodal_temp;
      RCP<const Map> targetMap = D0Crs->getColMap();
      P_nodal_temp = rcp(new CrsMatrixWrap(targetMap));
      RCP<const Import> importer = D0Crs->getCrsGraph()->getImporter();
      P_nodal_temp->doImport(*P_nodal, *importer, Xpetra::INSERT);
      P_nodal_temp->fillComplete(rcp_dynamic_cast<CrsMatrixWrap>(P_nodal)->getCrsMatrix()->getDomainMap(),
                                 rcp_dynamic_cast<CrsMatrixWrap>(P_nodal)->getCrsMatrix()->getRangeMap());
      P_nodal_imported = P_nodal_temp->getCrsMatrix();
      dump(*P_nodal_temp, "P_nodal_imported.m");
    } else
      P_nodal_imported = rcp_dynamic_cast<CrsMatrixWrap>(P_nodal)->getCrsMatrix();

    using ATS        = Kokkos::ArithTraits<SC>;
    using impl_Scalar = typename ATS::val_type;
    using impl_ATS = Kokkos::ArithTraits<impl_Scalar>;
    using range_type = Kokkos::RangePolicy<LO, typename NO::execution_space>;

    typedef typename Matrix::local_matrix_type KCRS;
    typedef typename KCRS::StaticCrsGraphType graph_t;
    typedef typename graph_t::row_map_type::non_const_type lno_view_t;
    typedef typename graph_t::entries_type::non_const_type lno_nnz_view_t;
    typedef typename KCRS::values_type::non_const_type scalar_view_t;

    const impl_Scalar impl_SC_ZERO = impl_ATS::zero();
    const impl_Scalar impl_SC_ONE = impl_ATS::one();
    const impl_Scalar impl_half = impl_SC_ONE / (impl_SC_ONE + impl_SC_ONE);


    RCP<Matrix> absD0 = MatrixFactory::BuildCopy(D0_);
    {
      auto localAbsD0 = absD0->getLocalMatrixDevice();
      Kokkos::deep_copy(localAbsD0.values, impl_SC_ONE);
    }

    RCP<Matrix> absD0_P_nodal = MatrixFactory::Build(D0->getRowMap());
    Xpetra::MatrixMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Multiply(*absD0,false,*P_nodal,false,*absD0_P_nodal,true,true);
    absD0 = Teuchos::null;

    RCP<Matrix> absD1 = MatrixFactory::BuildCopy(Dk_1_);
    {
      auto localAbsD1 = absD1->getLocalMatrixDevice();
      Kokkos::deep_copy(localAbsD1.values, impl_SC_ONE);
    }

    RCP<Matrix> absD1_absD0_P_nodal = MatrixFactory::Build(D1->getRowMap());
    Xpetra::MatrixMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Multiply(*absD1,false,*absD0_P_nodal,false,*absD1_absD0_P_nodal,true,true);
    absD0_P_nodal = Teuchos::null;
    absD1 = Teuchos::null;

    // Get data out of |D1|*|D0|*P_nodal.
    auto local_absD1_absD0_P_nodal = absD1_absD0_P_nodal->getLocalMatrixDevice();

    // Create the matrix object
    RCP<Map> blockColMap    = Xpetra::MapFactory<LO,GO,NO>::Build(P_nodal_imported->getColMap(), dim);
    RCP<Map> blockDomainMap = Xpetra::MapFactory<LO,GO,NO>::Build(P_nodal->getDomainMap(), dim);

    size_t nnzEstimate = dim*localD0P.graph.entries.size();
    lno_view_t specialProwptr("specialP_rowptr", numLocalRows+1);
    lno_nnz_view_t specialPcolind("specialP_colind",nnzEstimate);
    scalar_view_t specialPvals("specialP_vals",nnzEstimate);

    // adjust rowpointer
    Kokkos::parallel_for("MueLu:RefMaxwell::buildEdgeProlongator_adjustRowptr", range_type(0,numLocalRows+1),
                         KOKKOS_LAMBDA(const size_t i) {
                           specialProwptr(i) = dim*localD0P.graph.row_map(i);
                         });

    // adjust column indices
    Kokkos::parallel_for("MueLu:RefMaxwell::buildEdgeProlongator_adjustColind", range_type(0,localD0P.graph.entries.size()),
                         KOKKOS_LAMBDA(const size_t jj) {
                           for (size_t k = 0; k < dim; k++) {
                             specialPcolind(dim*jj+k) = dim*localD0P.graph.entries(jj)+k;
                             specialPvals(dim*jj+k) = impl_SC_ZERO;
                           }
                         });

    auto localNullspace = Nullspace_->getDeviceLocalView(Xpetra::Access::ReadOnly);

    // enter values
    {
      auto localD0 = D0_->getLocalMatrixDevice();
      auto localP = P_nodal_imported->getLocalMatrixDevice();
      Kokkos::parallel_for("MueLu:RefMaxwell::buildEdgeProlongator_enterValues", range_type(0,numLocalRows),
                           KOKKOS_LAMBDA(const size_t i) {
                             for (size_t ll = localD0.graph.row_map(i); ll < localD0.graph.row_map(i+1); ll++) {
                               LO l = localD0.graph.entries(ll);
                               for (size_t jj = localP.graph.row_map(l); jj < localP.graph.row_map(l+1); jj++) {
                                 LO j = localP.graph.entries(jj);
                                 impl_Scalar v = localP.values(jj);
                                 for (size_t k = 0; k < dim; k++) {
                                   LO jNew = dim*j+k;
                                   impl_Scalar n = localNullspace(i,k);
                                   size_t m;
                                   for (m = specialProwptr(i); m < specialProwptr(i+1); m++)
                                     if (specialPcolind(m) == jNew)
                                       break;
#if defined(HAVE_MUELU_DEBUG) && !defined(HAVE_MUELU_CUDA) && !defined(HAVE_MUELU_HIP)
                                   TEUCHOS_ASSERT_EQUALITY(specialPcolind(m),jNew);
#endif
                                   specialPvals(m) += impl_half * v * n;
                                 }
                               }
                             }
                           });
    }

    specialP = rcp(new CrsMatrixWrap(SM_Matrix_->getRowMap(), blockColMap, 0));
    RCP<CrsMatrix> specialPCrs = rcp_dynamic_cast<CrsMatrixWrap>(specialP)->getCrsMatrix();
    specialPCrs->setAllValues(specialProwptr, specialPcolind, specialPvals);
    specialPCrs->expertStaticFillComplete(blockDomainMap, SM_Matrix_->getRangeMap());



  }

  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void RefMaxwell<Scalar,LocalOrdinal,GlobalOrdinal,Node>::buildEdgeProlongator(const Teuchos::RCP<Matrix> &A_nodal,
                                                                                Teuchos::RCP<Matrix> &specialP,
                                                                                Teuchos::RCP<MultiVector> &specialNullspace,
                                                                                Teuchos::RCP<RealValuedMultiVector> &specialCoords) const {
    // The P11 matrix maps node based aggregrates { A_j } to edges { e_i }.
    //
    // The old implementation used
    // P11(i, j*dim+k) = sum_{nodes n_l in e_i intersected with A_j}  0.5 * phi_k(e_i) * P(n_l, A_j)
    // yet the paper gives
    // P11(i, j*dim+k) = sum_{nodes n_l in e_i intersected with A_j}  0.5 * phi_k(e_i)
    // where phi_k is the k-th nullspace vector.
    //
    // The graph of D0 contains the incidence from nodes to edges.
    // The nodal prolongator P maps aggregates to nodes.

    const SC SC_ZERO = Teuchos::ScalarTraits<SC>::zero();
    const SC SC_ONE = Teuchos::ScalarTraits<SC>::one();
    const Scalar half = SC_ONE / (SC_ONE + SC_ONE);
    size_t dim = Nullspace_->getNumVectors();
    // RCP<const Map> edgeMap = SM_Matrix_->getRowMap();
    // size_t numLocalRows = SM_Matrix_->getLocalNumRows();
    RCP<const Map> edgeMap = D0_->getRowMap();
    size_t numLocalRows = edgeMap->getLocalNumEntries();

    RCP<Matrix> P_nodal;
    RCP<CrsMatrix> P_nodal_imported;
    RCP<MultiVector> Nullspace_nodal;
    if (skipFirstLevel_) {

      buildNodalProlongator(A_nodal, P_nodal, Nullspace_nodal, specialCoords);

      RCP<CrsMatrix> D0Crs = rcp_dynamic_cast<CrsMatrixWrap>(D0_)->getCrsMatrix();

      // Import off-rank rows of P_nodal into P_nodal_imported
      int numProcs = P_nodal->getDomainMap()->getComm()->getSize();
      if (numProcs > 1) {
        RCP<CrsMatrixWrap> P_nodal_temp;
        RCP<const Map> targetMap = D0Crs->getColMap();
        P_nodal_temp = rcp(new CrsMatrixWrap(targetMap));
        RCP<const Import> importer = D0Crs->getCrsGraph()->getImporter();
        P_nodal_temp->doImport(*P_nodal, *importer, Xpetra::INSERT);
        P_nodal_temp->fillComplete(rcp_dynamic_cast<CrsMatrixWrap>(P_nodal)->getCrsMatrix()->getDomainMap(),
                                   rcp_dynamic_cast<CrsMatrixWrap>(P_nodal)->getCrsMatrix()->getRangeMap());
        P_nodal_imported = P_nodal_temp->getCrsMatrix();
        dump(*P_nodal_temp, "P_nodal_imported.m");
      } else
        P_nodal_imported = rcp_dynamic_cast<CrsMatrixWrap>(P_nodal)->getCrsMatrix();
    }

    if (useKokkos_) {

      using ATS        = Kokkos::ArithTraits<SC>;
      using impl_Scalar = typename ATS::val_type;
      using impl_ATS = Kokkos::ArithTraits<impl_Scalar>;
      using range_type = Kokkos::RangePolicy<LO, typename NO::execution_space>;

      typedef typename Matrix::local_matrix_type KCRS;
      typedef typename KCRS::StaticCrsGraphType graph_t;
      typedef typename graph_t::row_map_type::non_const_type lno_view_t;
      typedef typename graph_t::entries_type::non_const_type lno_nnz_view_t;
      typedef typename KCRS::values_type::non_const_type scalar_view_t;

      const impl_Scalar impl_SC_ZERO = impl_ATS::zero();
      const impl_Scalar impl_SC_ONE = impl_ATS::one();
      const impl_Scalar impl_half = impl_SC_ONE / (impl_SC_ONE + impl_SC_ONE);


      // Which algorithm should we use for the construction of the special prolongator?
      // Option "mat-mat":
      //   Multiply D0 * P_nodal, take graph, blow up the domain space and compute the entries.
      std::string defaultAlgo = "mat-mat";

      std::string algo = defaultAlgo;
      if (parameterList_.isType<std::string>("refmaxwell: prolongator compute algorithm"))
        algo = parameterList_.get<std::string>("refmaxwell: prolongator compute algorithm");

      if (skipFirstLevel_) {
	// Get data out of P_nodal_imported and D0.

        if (algo == "mat-mat") {
          RCP<Matrix> D0_P_nodal = MatrixFactory::Build(edgeMap);
          Xpetra::MatrixMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Multiply(*D0_,false,*P_nodal,false,*D0_P_nodal,true,true);

#ifdef HAVE_MUELU_DEBUG
          TEUCHOS_ASSERT(D0_P_nodal->getColMap()->isSameAs(*P_nodal_imported->getColMap()));
#endif

          // Get data out of D0*P.
          auto localD0P = D0_P_nodal->getLocalMatrixDevice();

          // Create the matrix object
          RCP<Map> blockColMap    = Xpetra::MapFactory<LO,GO,NO>::Build(P_nodal_imported->getColMap(), dim);
          RCP<Map> blockDomainMap = Xpetra::MapFactory<LO,GO,NO>::Build(P_nodal->getDomainMap(), dim);

          size_t nnzEstimate = dim*localD0P.graph.entries.size();
          lno_view_t specialProwptr("specialP_rowptr", numLocalRows+1);
          lno_nnz_view_t specialPcolind("specialP_colind",nnzEstimate);
          scalar_view_t specialPvals("specialP_vals",nnzEstimate);

          // adjust rowpointer
          Kokkos::parallel_for("MueLu:RefMaxwell::buildEdgeProlongator_adjustRowptr", range_type(0,numLocalRows+1),
                               KOKKOS_LAMBDA(const size_t i) {
                                 specialProwptr(i) = dim*localD0P.graph.row_map(i);
                               });

          // adjust column indices
          Kokkos::parallel_for("MueLu:RefMaxwell::buildEdgeProlongator_adjustColind", range_type(0,localD0P.graph.entries.size()),
                               KOKKOS_LAMBDA(const size_t jj) {
                                 for (size_t k = 0; k < dim; k++) {
                                   specialPcolind(dim*jj+k) = dim*localD0P.graph.entries(jj)+k;
                                   specialPvals(dim*jj+k) = impl_SC_ZERO;
                                 }
                               });

          auto localNullspace = Nullspace_->getDeviceLocalView(Xpetra::Access::ReadOnly);

          // enter values
          if (D0_->getLocalMaxNumRowEntries()>2) {
            // The matrix D0 has too many entries per row.
            // Therefore we need to check whether its entries are actually non-zero.
            // This is the case for the matrices built by MiniEM.
            GetOStream(Warnings0) << "RefMaxwell::buildEdgeProlongator(): D0 matrix has more than 2 entries per row. Taking inefficient code path." << std::endl;

            magnitudeType tol = Teuchos::ScalarTraits<magnitudeType>::eps();

	    auto localD0 = D0_->getLocalMatrixDevice();
	    auto localP = P_nodal_imported->getLocalMatrixDevice();
            Kokkos::parallel_for("MueLu:RefMaxwell::buildEdgeProlongator_enterValues_D0wZeros", range_type(0,numLocalRows),
                                 KOKKOS_LAMBDA(const size_t i) {
                                   for (size_t ll = localD0.graph.row_map(i); ll < localD0.graph.row_map(i+1); ll++) {
                                     LO l = localD0.graph.entries(ll);
                                     impl_Scalar p = localD0.values(ll);
                                     if (impl_ATS::magnitude(p) < tol)
                                       continue;
                                     for (size_t jj = localP.graph.row_map(l); jj < localP.graph.row_map(l+1); jj++) {
                                       LO j = localP.graph.entries(jj);
                                       impl_Scalar v = localP.values(jj);
                                       for (size_t k = 0; k < dim; k++) {
                                         LO jNew = dim*j+k;
                                         impl_Scalar n = localNullspace(i,k);
                                         size_t m;
                                         for (m = specialProwptr(i); m < specialProwptr(i+1); m++)
                                           if (specialPcolind(m) == jNew)
                                             break;
#if defined(HAVE_MUELU_DEBUG) && !defined(HAVE_MUELU_CUDA) && !defined(HAVE_MUELU_HIP) && !defined(HAVE_MUELU_SYCL)
                                         TEUCHOS_ASSERT_EQUALITY(specialPcolind(m),jNew);
#endif
                                         specialPvals(m) += impl_half * v * n;
                                       }
                                     }
                                   }
                                 });

          } else {
	    auto localD0 = D0_->getLocalMatrixDevice();
	    auto localP = P_nodal_imported->getLocalMatrixDevice();
            Kokkos::parallel_for("MueLu:RefMaxwell::buildEdgeProlongator_enterValues", range_type(0,numLocalRows),
                                 KOKKOS_LAMBDA(const size_t i) {
                                   for (size_t ll = localD0.graph.row_map(i); ll < localD0.graph.row_map(i+1); ll++) {
                                     LO l = localD0.graph.entries(ll);
                                     for (size_t jj = localP.graph.row_map(l); jj < localP.graph.row_map(l+1); jj++) {
                                       LO j = localP.graph.entries(jj);
                                       impl_Scalar v = localP.values(jj);
                                       for (size_t k = 0; k < dim; k++) {
                                         LO jNew = dim*j+k;
                                         impl_Scalar n = localNullspace(i,k);
                                         size_t m;
                                         for (m = specialProwptr(i); m < specialProwptr(i+1); m++)
                                           if (specialPcolind(m) == jNew)
                                             break;
#if defined(HAVE_MUELU_DEBUG) && !defined(HAVE_MUELU_CUDA) && !defined(HAVE_MUELU_HIP) && !defined(HAVE_MUELU_SYCL)
                                         TEUCHOS_ASSERT_EQUALITY(specialPcolind(m),jNew);
#endif
                                         specialPvals(m) += impl_half * v * n;
                                       }
                                     }
                                   }
                                 });
          }

          specialP = rcp(new CrsMatrixWrap(edgeMap, blockColMap, 0));
          RCP<CrsMatrix> specialPCrs = rcp_dynamic_cast<CrsMatrixWrap>(specialP)->getCrsMatrix();
          specialPCrs->setAllValues(specialProwptr, specialPcolind, specialPvals);
          specialPCrs->expertStaticFillComplete(blockDomainMap, edgeMap);

        } else
          TEUCHOS_TEST_FOR_EXCEPTION(false,std::invalid_argument,algo << " is not a valid option for \"refmaxwell: prolongator compute algorithm\"");

        specialNullspace = MultiVectorFactory::Build(specialP->getDomainMap(), dim);

        auto localNullspace_nodal = Nullspace_nodal->getDeviceLocalView(Xpetra::Access::ReadOnly);
        auto localSpecialNullspace = specialNullspace->getDeviceLocalView(Xpetra::Access::ReadWrite);
        Kokkos::parallel_for("MueLu:RefMaxwell::buildEdgeProlongator_nullspace", range_type(0,Nullspace_nodal->getLocalLength()),
                             KOKKOS_LAMBDA(const size_t i) {
                               impl_Scalar val = localNullspace_nodal(i,0);
                               for (size_t j = 0; j < dim; j++)
                                 localSpecialNullspace(dim*i+j, j) = val;
                             });

      } else { // !skipFirstLevel_
	// Get data out of P_nodal_imported and D0.
	auto localD0 = D0_->getLocalMatrixDevice();

        specialCoords = Coords_;

        if (algo == "mat-mat") {

          // Create the matrix object
          RCP<Map> blockColMap    = Xpetra::MapFactory<LO,GO,NO>::Build(D0_->getColMap(), dim);
          RCP<Map> blockDomainMap = Xpetra::MapFactory<LO,GO,NO>::Build(D0_->getDomainMap(), dim);

          size_t nnzEstimate = dim*localD0.graph.entries.size();
          lno_view_t specialProwptr("specialP_rowptr", numLocalRows+1);
          lno_nnz_view_t specialPcolind("specialP_colind",nnzEstimate);
          scalar_view_t specialPvals("specialP_vals",nnzEstimate);

          // adjust rowpointer
          Kokkos::parallel_for("MueLu:RefMaxwell::buildEdgeProlongator_adjustRowptr", range_type(0,numLocalRows+1),
                               KOKKOS_LAMBDA(const size_t i) {
                                 specialProwptr(i) = dim*localD0.graph.row_map(i);
                               });

          // adjust column indices
          Kokkos::parallel_for("MueLu:RefMaxwell::buildEdgeProlongator_adjustColind", range_type(0,localD0.graph.entries.size()),
                               KOKKOS_LAMBDA(const size_t jj) {
                                 for (size_t k = 0; k < dim; k++) {
                                   specialPcolind(dim*jj+k) = dim*localD0.graph.entries(jj)+k;
                                   specialPvals(dim*jj+k) = impl_SC_ZERO;
                                 }
                               });

          auto localNullspace = Nullspace_->getDeviceLocalView(Xpetra::Access::ReadOnly);

          // enter values
          if (D0_->getLocalMaxNumRowEntries()>2) {
            // The matrix D0 has too many entries per row.
            // Therefore we need to check whether its entries are actually non-zero.
            // This is the case for the matrices built by MiniEM.
            GetOStream(Warnings0) << "RefMaxwell::buildEdgeProlongator(): D0 matrix has more than 2 entries per row. Taking inefficient code path." << std::endl;

            magnitudeType tol = Teuchos::ScalarTraits<magnitudeType>::eps();

            Kokkos::parallel_for("MueLu:RefMaxwell::buildEdgeProlongator_enterValues_D0wZeros", range_type(0,numLocalRows),
                                 KOKKOS_LAMBDA(const size_t i) {
                                   for (size_t jj = localD0.graph.row_map(i); jj < localD0.graph.row_map(i+1); jj++) {
                                     LO j = localD0.graph.entries(jj);
                                     impl_Scalar p = localD0.values(jj);
                                     if (impl_ATS::magnitude(p) < tol)
                                       continue;
                                     for (size_t k = 0; k < dim; k++) {
                                       LO jNew = dim*j+k;
                                       impl_Scalar n = localNullspace(i,k);
                                       size_t m;
                                       for (m = specialProwptr(i); m < specialProwptr(i+1); m++)
                                         if (specialPcolind(m) == jNew)
                                           break;
#if defined(HAVE_MUELU_DEBUG) && !defined(HAVE_MUELU_CUDA) && !defined(HAVE_MUELU_HIP) && !defined(HAVE_MUELU_SYCL)
                                       TEUCHOS_ASSERT_EQUALITY(specialPcolind(m),jNew);
#endif
                                       specialPvals(m) += impl_half * n;
                                     }
                                   }
                                 });

          } else {
            Kokkos::parallel_for("MueLu:RefMaxwell::buildEdgeProlongator_enterValues", range_type(0,numLocalRows),
                                 KOKKOS_LAMBDA(const size_t i) {
                                   for (size_t jj = localD0.graph.row_map(i); jj < localD0.graph.row_map(i+1); jj++) {
                                     LO j = localD0.graph.entries(jj);
                                     for (size_t k = 0; k < dim; k++) {
                                       LO jNew = dim*j+k;
                                       impl_Scalar n = localNullspace(i,k);
                                       size_t m;
                                       for (m = specialProwptr(i); m < specialProwptr(i+1); m++)
                                         if (specialPcolind(m) == jNew)
                                           break;
#if defined(HAVE_MUELU_DEBUG) && !defined(HAVE_MUELU_CUDA) && !defined(HAVE_MUELU_HIP) && !defined(HAVE_MUELU_SYCL)
                                       TEUCHOS_ASSERT_EQUALITY(specialPcolind(m),jNew);
#endif
                                       specialPvals(m) += impl_half * n;
                                     }
                                   }
                                 });
          }

          specialP = rcp(new CrsMatrixWrap(edgeMap, blockColMap, 0));
          RCP<CrsMatrix> specialPCrs = rcp_dynamic_cast<CrsMatrixWrap>(specialP)->getCrsMatrix();
          specialPCrs->setAllValues(specialProwptr, specialPcolind, specialPvals);
          specialPCrs->expertStaticFillComplete(blockDomainMap, edgeMap);
        } else
          TEUCHOS_TEST_FOR_EXCEPTION(false,std::invalid_argument,algo << " is not a valid option for \"refmaxwell: prolongator compute algorithm\"");

      }
    } else
      {
        // get nullspace vectors
        ArrayRCP<ArrayRCP<const SC> > nullspaceRCP(dim);
        ArrayRCP<ArrayView<const SC> > nullspace(dim);
        for(size_t i=0; i<dim; i++) {
          nullspaceRCP[i] = Nullspace_->getData(i);
          nullspace[i] = nullspaceRCP[i]();
        }

        // Get data out of P_nodal_imported and D0.
        ArrayRCP<size_t>            specialProwptr_RCP;
        ArrayRCP<LO>                specialPcolind_RCP;
        ArrayRCP<SC>                specialPvals_RCP;


        // Which algorithm should we use for the construction of the special prolongator?
        // Option "mat-mat":
        //   Multiply D0 * P_nodal, take graph, blow up the domain space and compute the entries.
        // Option "gustavson":
        //   Loop over D0, P and nullspace and allocate directly. (Gustavson-like)
        //   More efficient, but only available for serial node.
        std::string defaultAlgo = "mat-mat";
        std::string algo = defaultAlgo;
        if (parameterList_.isType<std::string>("refmaxwell: prolongator compute algorithm"))
          algo = parameterList_.get<std::string>("refmaxwell: prolongator compute algorithm");

        if (skipFirstLevel_) {

          if (algo == "mat-mat") {
            RCP<Matrix> D0_P_nodal = MatrixFactory::Build(edgeMap);
            Xpetra::MatrixMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Multiply(*D0_,false,*P_nodal,false,*D0_P_nodal,true,true);


	    ArrayRCP<const size_t>      D0rowptr_RCP;
	    ArrayRCP<const LO>          D0colind_RCP;
	    ArrayRCP<const SC>          D0vals_RCP;
	    rcp_dynamic_cast<CrsMatrixWrap>(D0_)->getCrsMatrix()->getAllValues(D0rowptr_RCP, D0colind_RCP, D0vals_RCP);
	    // For efficiency
	    // Refers to an issue where Teuchos::ArrayRCP::operator[] may be
	    // slower than Teuchos::ArrayView::operator[].
	    ArrayView<const size_t>     D0rowptr;
	    ArrayView<const LO>         D0colind;
	    ArrayView<const SC>         D0vals;
	    D0rowptr = D0rowptr_RCP();  D0colind = D0colind_RCP();  D0vals = D0vals_RCP();

	    // Get data out of P_nodal_imported and D0.
	    ArrayRCP<const size_t>      Prowptr_RCP;
	    ArrayRCP<const LO>          Pcolind_RCP;
	    ArrayRCP<const SC>          Pvals_RCP;
	    P_nodal_imported->getAllValues(Prowptr_RCP, Pcolind_RCP, Pvals_RCP);
	    ArrayView<const size_t>     Prowptr;
	    ArrayView<const LO>         Pcolind;
	    ArrayView<const SC>         Pvals;
	    Prowptr  = Prowptr_RCP();   Pcolind  = Pcolind_RCP();   Pvals = Pvals_RCP();

            // Get data out of D0*P.
            ArrayRCP<const size_t>      D0Prowptr_RCP;
            ArrayRCP<const LO>          D0Pcolind_RCP;
            ArrayRCP<const SC>          D0Pvals_RCP;
            rcp_dynamic_cast<CrsMatrixWrap>(D0_P_nodal)->getCrsMatrix()->getAllValues(D0Prowptr_RCP, D0Pcolind_RCP, D0Pvals_RCP);

            // For efficiency
            // Refers to an issue where Teuchos::ArrayRCP::operator[] may be
            // slower than Teuchos::ArrayView::operator[].
            ArrayView<const size_t>     D0Prowptr;
            ArrayView<const LO>         D0Pcolind;
            D0Prowptr = D0Prowptr_RCP(); D0Pcolind = D0Pcolind_RCP();

            // Create the matrix object
            RCP<Map> blockColMap    = Xpetra::MapFactory<LO,GO,NO>::Build(P_nodal_imported->getColMap(), dim);
            RCP<Map> blockDomainMap = Xpetra::MapFactory<LO,GO,NO>::Build(P_nodal->getDomainMap(), dim);
            specialP = rcp(new CrsMatrixWrap(edgeMap, blockColMap, 0));
            RCP<CrsMatrix> specialPCrs = rcp_dynamic_cast<CrsMatrixWrap>(specialP)->getCrsMatrix();
            size_t nnzEstimate = dim*D0Prowptr[numLocalRows];
            specialPCrs->allocateAllValues(nnzEstimate, specialProwptr_RCP, specialPcolind_RCP, specialPvals_RCP);

            ArrayView<size_t> specialProwptr = specialProwptr_RCP();
            ArrayView<LO>     specialPcolind = specialPcolind_RCP();
            ArrayView<SC>     specialPvals   = specialPvals_RCP();

            // adjust rowpointer
            for (size_t i = 0; i < numLocalRows+1; i++) {
              specialProwptr[i] = dim*D0Prowptr[i];
            }

            // adjust column indices
            for (size_t jj = 0; jj < (size_t) D0Prowptr[numLocalRows]; jj++)
              for (size_t k = 0; k < dim; k++) {
                specialPcolind[dim*jj+k] = dim*D0Pcolind[jj]+k;
                specialPvals[dim*jj+k] = SC_ZERO;
              }

            RCP<const Map> P_nodal_imported_colmap = P_nodal_imported->getColMap();
            RCP<const Map> D0_P_nodal_colmap = D0_P_nodal->getColMap();
            // enter values
            if (D0_->getLocalMaxNumRowEntries()>2) {
              // The matrix D0 has too many entries per row.
              // Therefore we need to check whether its entries are actually non-zero.
              // This is the case for the matrices built by MiniEM.
              GetOStream(Warnings0) << "RefMaxwell::buildEdgeProlongator(): D0 matrix has more than 2 entries per row. Taking inefficient code path." << std::endl;

              magnitudeType tol = Teuchos::ScalarTraits<magnitudeType>::eps();
              for (size_t i = 0; i < numLocalRows; i++) {
                for (size_t ll = D0rowptr[i]; ll < D0rowptr[i+1]; ll++) {
                  LO l = D0colind[ll];
                  SC p = D0vals[ll];
                  if (Teuchos::ScalarTraits<Scalar>::magnitude(p) < tol)
                    continue;
                  for (size_t jj = Prowptr[l]; jj < Prowptr[l+1]; jj++) {
                    LO j = Pcolind[jj];
                    j = D0_P_nodal_colmap->getLocalElement(P_nodal_imported_colmap->getGlobalElement(j));
                    SC v = Pvals[jj];
                    for (size_t k = 0; k < dim; k++) {
                      LO jNew = dim*j+k;
                      SC n = nullspace[k][i];
                      size_t m;
                      for (m = specialProwptr[i]; m < specialProwptr[i+1]; m++)
                        if (specialPcolind[m] == jNew)
                          break;
#ifdef HAVE_MUELU_DEBUG
                      TEUCHOS_ASSERT_EQUALITY(specialPcolind[m],jNew);
#endif
                      specialPvals[m] += half * v * n;
                    }
                  }
                }
              }
            } else {
              // enter values
              for (size_t i = 0; i < numLocalRows; i++) {
                for (size_t ll = D0rowptr[i]; ll < D0rowptr[i+1]; ll++) {
                  LO l = D0colind[ll];
                  for (size_t jj = Prowptr[l]; jj < Prowptr[l+1]; jj++) {
                    LO j = Pcolind[jj];
                    j = D0_P_nodal_colmap->getLocalElement(P_nodal_imported_colmap->getGlobalElement(j));
                    SC v = Pvals[jj];
                    for (size_t k = 0; k < dim; k++) {
                      LO jNew = dim*j+k;
                      SC n = nullspace[k][i];
                      size_t m;
                      for (m = specialProwptr[i]; m < specialProwptr[i+1]; m++)
                        if (specialPcolind[m] == jNew)
                          break;
#ifdef HAVE_MUELU_DEBUG
                      TEUCHOS_ASSERT_EQUALITY(specialPcolind[m],jNew);
#endif
                      specialPvals[m] += half * v * n;
                    }
                  }
                }
              }
            }

            specialPCrs->setAllValues(specialProwptr_RCP, specialPcolind_RCP, specialPvals_RCP);
            specialPCrs->expertStaticFillComplete(blockDomainMap, edgeMap);

          } else if (algo == "gustavson") {
	    ArrayRCP<const size_t>      D0rowptr_RCP;
	    ArrayRCP<const LO>          D0colind_RCP;
	    ArrayRCP<const SC>          D0vals_RCP;
	    rcp_dynamic_cast<CrsMatrixWrap>(D0_)->getCrsMatrix()->getAllValues(D0rowptr_RCP, D0colind_RCP, D0vals_RCP);
	    // For efficiency
	    // Refers to an issue where Teuchos::ArrayRCP::operator[] may be
	    // slower than Teuchos::ArrayView::operator[].
	    ArrayView<const size_t>     D0rowptr;
	    ArrayView<const LO>         D0colind;
	    ArrayView<const SC>         D0vals;
	    D0rowptr = D0rowptr_RCP();  D0colind = D0colind_RCP();  D0vals = D0vals_RCP();

	    // Get data out of P_nodal_imported and D0.
	    ArrayRCP<const size_t>      Prowptr_RCP;
	    ArrayRCP<const LO>          Pcolind_RCP;
	    ArrayRCP<const SC>          Pvals_RCP;
	    P_nodal_imported->getAllValues(Prowptr_RCP, Pcolind_RCP, Pvals_RCP);
	    ArrayView<const size_t>     Prowptr;
	    ArrayView<const LO>         Pcolind;
	    ArrayView<const SC>         Pvals;
	    Prowptr  = Prowptr_RCP();   Pcolind  = Pcolind_RCP();   Pvals = Pvals_RCP();

            LO maxspecialPcol = dim * P_nodal_imported->getColMap()->getMaxLocalIndex();
            const size_t ST_INVALID = Teuchos::OrdinalTraits<LO>::invalid();
            Array<size_t> specialP_status(dim*maxspecialPcol, ST_INVALID);
            // This is ad-hoc and should maybe be replaced with some better heuristics.
            size_t nnz_alloc = dim*D0vals_RCP.size();

            // Create the matrix object
            RCP<Map> blockColMap    = Xpetra::MapFactory<LO,GO,NO>::Build(P_nodal_imported->getColMap(), dim);
            RCP<Map> blockDomainMap = Xpetra::MapFactory<LO,GO,NO>::Build(P_nodal->getDomainMap(), dim);
            specialP = rcp(new CrsMatrixWrap(edgeMap, blockColMap, 0));
            RCP<CrsMatrix> specialPCrs = rcp_dynamic_cast<CrsMatrixWrap>(specialP)->getCrsMatrix();
            specialPCrs->allocateAllValues(nnz_alloc, specialProwptr_RCP, specialPcolind_RCP, specialPvals_RCP);

            ArrayView<size_t> specialProwptr = specialProwptr_RCP();
            ArrayView<LO>     specialPcolind = specialPcolind_RCP();
            ArrayView<SC>     specialPvals   = specialPvals_RCP();

            size_t nnz;
            if (D0_->getLocalMaxNumRowEntries()>2) {
              // The matrix D0 has too many entries per row.
              // Therefore we need to check whether its entries are actually non-zero.
              // This is the case for the matrices built by MiniEM.
              GetOStream(Warnings0) << "RefMaxwell::buildEdgeProlongator(): D0 matrix has more than 2 entries per row. Taking inefficient code path." << std::endl;

              magnitudeType tol = Teuchos::ScalarTraits<magnitudeType>::eps();
              nnz = 0;
              size_t nnz_old = 0;
              for (size_t i = 0; i < numLocalRows; i++) {
                specialProwptr[i] = nnz;
                for (size_t ll = D0rowptr[i]; ll < D0rowptr[i+1]; ll++) {
                  LO l = D0colind[ll];
                  SC p = D0vals[ll];
                  if (Teuchos::ScalarTraits<Scalar>::magnitude(p) < tol)
                    continue;
                  for (size_t jj = Prowptr[l]; jj < Prowptr[l+1]; jj++) {
                    LO j = Pcolind[jj];
                    SC v = Pvals[jj];
                    for (size_t k = 0; k < dim; k++) {
                      LO jNew = dim*j+k;
                      SC n = nullspace[k][i];
                      // do we already have an entry for (i, jNew)?
                      if (specialP_status[jNew] == ST_INVALID || specialP_status[jNew] < nnz_old) {
                        specialP_status[jNew] = nnz;
                        specialPcolind[nnz] = jNew;
                        specialPvals[nnz] = half * v * n;
                        // or should it be
                        // specialPvals[nnz] = half * n;
                        nnz++;
                      } else {
                        specialPvals[specialP_status[jNew]] += half * v * n;
                        // or should it be
                        // specialPvals[specialP_status[jNew]] += half * n;
                      }
                    }
                  }
                }
                nnz_old = nnz;
              }
              specialProwptr[numLocalRows] = nnz;
            } else {
              nnz = 0;
              size_t nnz_old = 0;
              for (size_t i = 0; i < numLocalRows; i++) {
                specialProwptr[i] = nnz;
                for (size_t ll = D0rowptr[i]; ll < D0rowptr[i+1]; ll++) {
                  LO l = D0colind[ll];
                  for (size_t jj = Prowptr[l]; jj < Prowptr[l+1]; jj++) {
                    LO j = Pcolind[jj];
                    SC v = Pvals[jj];
                    for (size_t k = 0; k < dim; k++) {
                      LO jNew = dim*j+k;
                      SC n = nullspace[k][i];
                      // do we already have an entry for (i, jNew)?
                      if (specialP_status[jNew] == ST_INVALID || specialP_status[jNew] < nnz_old) {
                        specialP_status[jNew] = nnz;
                        specialPcolind[nnz] = jNew;
                        specialPvals[nnz] = half * v * n;
                        // or should it be
                        // specialPvals[nnz] = half * n;
                        nnz++;
                      } else {
                        specialPvals[specialP_status[jNew]] += half * v * n;
                        // or should it be
                        // specialPvals[specialP_status[jNew]] += half * n;
                      }
                    }
                  }
                }
                nnz_old = nnz;
              }
              specialProwptr[numLocalRows] = nnz;
            }

            if (blockDomainMap->lib() == Xpetra::UseTpetra) {
              // Downward resize
              // - Cannot resize for Epetra, as it checks for same pointers
              // - Need to resize for Tpetra, as it checks ().size() == specialProwptr[numLocalRows]
              specialPvals_RCP.resize(nnz);
              specialPcolind_RCP.resize(nnz);
            }

            specialPCrs->setAllValues(specialProwptr_RCP, specialPcolind_RCP, specialPvals_RCP);
            specialPCrs->expertStaticFillComplete(blockDomainMap, edgeMap);
          } else
            TEUCHOS_TEST_FOR_EXCEPTION(false,std::invalid_argument,algo << " is not a valid option for \"refmaxwell: prolongator compute algorithm\"");

          specialNullspace = MultiVectorFactory::Build(specialP->getDomainMap(), dim);

          ArrayRCP<const Scalar> ns_rcp = Nullspace_nodal->getData(0);
          ArrayView<const Scalar> ns_view = ns_rcp();
          for (size_t i = 0; i < Nullspace_nodal->getLocalLength(); i++) {
            Scalar val = ns_view[i];
            for (size_t j = 0; j < dim; j++)
              specialNullspace->replaceLocalValue(dim*i+j, j, val);
          }


        } else { // !skipFirstLevel_
	  ArrayRCP<const size_t>      D0rowptr_RCP;
	  ArrayRCP<const LO>          D0colind_RCP;
	  ArrayRCP<const SC>          D0vals_RCP;
	  rcp_dynamic_cast<CrsMatrixWrap>(D0_)->getCrsMatrix()->getAllValues(D0rowptr_RCP, D0colind_RCP, D0vals_RCP);
	  // For efficiency
	  // Refers to an issue where Teuchos::ArrayRCP::operator[] may be
	  // slower than Teuchos::ArrayView::operator[].
	  ArrayView<const size_t>     D0rowptr;
	  ArrayView<const LO>         D0colind;
	  ArrayView<const SC>         D0vals;
	  D0rowptr = D0rowptr_RCP();  D0colind = D0colind_RCP();  D0vals = D0vals_RCP();

          specialCoords = Coords_;
          if (algo == "mat-mat") {

            // Create the matrix object
            RCP<Map> blockColMap    = Xpetra::MapFactory<LO,GO,NO>::Build(D0_->getColMap(), dim);
            RCP<Map> blockDomainMap = Xpetra::MapFactory<LO,GO,NO>::Build(D0_->getDomainMap(), dim);
            specialP = rcp(new CrsMatrixWrap(edgeMap, blockColMap, 0));
            RCP<CrsMatrix> specialPCrs = rcp_dynamic_cast<CrsMatrixWrap>(specialP)->getCrsMatrix();
            size_t nnzEstimate = dim*D0rowptr[numLocalRows];
            specialPCrs->allocateAllValues(nnzEstimate, specialProwptr_RCP, specialPcolind_RCP, specialPvals_RCP);

            ArrayView<size_t> specialProwptr = specialProwptr_RCP();
            ArrayView<LO>     specialPcolind = specialPcolind_RCP();
            ArrayView<SC>     specialPvals   = specialPvals_RCP();

            // adjust rowpointer
            for (size_t i = 0; i < numLocalRows+1; i++) {
              specialProwptr[i] = dim*D0rowptr[i];
            }

            // adjust column indices
            for (size_t jj = 0; jj < (size_t) D0rowptr[numLocalRows]; jj++)
              for (size_t k = 0; k < dim; k++) {
                specialPcolind[dim*jj+k] = dim*D0colind[jj]+k;
                specialPvals[dim*jj+k] = SC_ZERO;
              }

            // enter values
            if (D0_->getLocalMaxNumRowEntries()>2) {
              // The matrix D0 has too many entries per row.
              // Therefore we need to check whether its entries are actually non-zero.
              // This is the case for the matrices built by MiniEM.
              GetOStream(Warnings0) << "RefMaxwell::buildEdgeProlongator(): D0 matrix has more than 2 entries per row. Taking inefficient code path." << std::endl;

              magnitudeType tol = Teuchos::ScalarTraits<magnitudeType>::eps();
              for (size_t i = 0; i < numLocalRows; i++) {
                for (size_t jj = D0rowptr[i]; jj < D0rowptr[i+1]; jj++) {
                  LO j = D0colind[jj];
                  SC p = D0vals[jj];
                  if (Teuchos::ScalarTraits<Scalar>::magnitude(p) < tol)
                    continue;
                  for (size_t k = 0; k < dim; k++) {
                    LO jNew = dim*j+k;
                    SC n = nullspace[k][i];
                    size_t m;
                    for (m = specialProwptr[i]; m < specialProwptr[i+1]; m++)
                      if (specialPcolind[m] == jNew)
                        break;
#ifdef HAVE_MUELU_DEBUG
                    TEUCHOS_ASSERT_EQUALITY(specialPcolind[m],jNew);
#endif
                    specialPvals[m] += half * n;
                  }
                }
              }
            } else {
              // enter values
              for (size_t i = 0; i < numLocalRows; i++) {
                for (size_t jj = D0rowptr[i]; jj < D0rowptr[i+1]; jj++) {
                  LO j = D0colind[jj];

                  for (size_t k = 0; k < dim; k++) {
                    LO jNew = dim*j+k;
                    SC n = nullspace[k][i];
                    size_t m;
                    for (m = specialProwptr[i]; m < specialProwptr[i+1]; m++)
                      if (specialPcolind[m] == jNew)
                        break;
#ifdef HAVE_MUELU_DEBUG
                    TEUCHOS_ASSERT_EQUALITY(specialPcolind[m],jNew);
#endif
                    specialPvals[m] += half * n;
                  }
                }
              }
            }

            specialPCrs->setAllValues(specialProwptr_RCP, specialPcolind_RCP, specialPvals_RCP);
            specialPCrs->expertStaticFillComplete(blockDomainMap, edgeMap);

          } else
            TEUCHOS_TEST_FOR_EXCEPTION(false,std::invalid_argument,algo << " is not a valid option for \"refmaxwell: prolongator compute algorithm\"");
        }
      }
  }


  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void
  RefMaxwell<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
  buildFaceProlongator(const Teuchos::RCP<Matrix> &A_nodal,
                       Teuchos::RCP<Matrix> &specialP,
                       Teuchos::RCP<MultiVector> &specialNullspace,
                       Teuchos::RCP<RealValuedMultiVector> &specialCoords) const {
    throw;
  }


  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void
  RefMaxwell<Scalar,LocalOrdinal,GlobalOrdinal,Node>::setupSubSolve(Teuchos::RCP<Hierarchy> &hierarchy,
                                                                    Teuchos::RCP<Operator> &thyraPrecOp,
                                                                    const Teuchos::RCP<Matrix> &A,
                                                                    const Teuchos::RCP<MultiVector> &Nullspace,
                                                                    const Teuchos::RCP<RealValuedMultiVector> &Coords,
                                                                    Teuchos::ParameterList &params,
                                                                    std::string &label,
                                                                    const bool reuse,
                                                                    const bool isSingular) {
    int oldRank = SetProcRankVerbose(A->getDomainMap()->getComm()->getRank());
    if (IsPrint(Statistics2)) {
      RCP<ParameterList> pl = rcp(new ParameterList());;
      pl->set("printLoadBalancingInfo", true);
      pl->set("printCommInfo",          true);
      GetOStream(Statistics2) << PerfUtils::PrintMatrixInfo(*A, label, pl);
    }
#if defined(HAVE_MUELU_STRATIMIKOS) && defined(HAVE_MUELU_THYRA)
    if (params.isType<std::string>("Preconditioner Type")) {
      TEUCHOS_ASSERT(!reuse);
      // build a Stratimikos preconditioner
      if (params.get<std::string>("Preconditioner Type") == "MueLu") {
        ParameterList& userParamList = params.sublist("Preconditioner Types").sublist("MueLu").sublist("user data");
        if (!Nullspace.is_null())
          userParamList.set<RCP<MultiVector> >("Nullspace", Nullspace);
        userParamList.set<RCP<RealValuedMultiVector> >("Coordinates", Coords);
      }
      thyraPrecOp = rcp(new XpetraThyraLinearOp<Scalar,LocalOrdinal,GlobalOrdinal,Node>(coarseA11_, rcp(&params, false)));
    } else
#endif
      {
        // build a MueLu hierarchy

        if (!reuse) {
          ParameterList& userParamList = params.sublist("user data");
          userParamList.set<RCP<RealValuedMultiVector> >("Coordinates", Coords);
          if (!Nullspace.is_null())
            userParamList.set<RCP<MultiVector> >("Nullspace", Nullspace);

          if (isSingular) {
            std::string coarseType = "";
            if (params.isParameter("coarse: type")) {
              coarseType = params.get<std::string>("coarse: type");
              // Transform string to "Abcde" notation
              std::transform(coarseType.begin(),   coarseType.end(),   coarseType.begin(), ::tolower);
              std::transform(coarseType.begin(), ++coarseType.begin(), coarseType.begin(), ::toupper);
            }
            if ((coarseType == "" ||
                 coarseType == "Klu" ||
                 coarseType == "Klu2") &&
                (!params.isSublist("coarse: params") ||
                 !params.sublist("coarse: params").isParameter("fix nullspace")))
              params.sublist("coarse: params").set("fix nullspace",true);
          }

          hierarchy = MueLu::CreateXpetraPreconditioner(A, params);
        } else {
          RCP<MueLu::Level> level0 = hierarchy->GetLevel(0);
          level0->Set("A", A);
          hierarchy->SetupRe();
        }
      }
    SetProcRankVerbose(oldRank);
  }


  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void RefMaxwell<Scalar,LocalOrdinal,GlobalOrdinal,Node>::resetMatrix(RCP<Matrix> SM_Matrix_new, bool ComputePrec) {
    bool reuse = !SM_Matrix_.is_null();
    SM_Matrix_ = SM_Matrix_new;
    dump(*SM_Matrix_, "SM.m");
    if (ComputePrec)
      compute(reuse);
  }


  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void RefMaxwell<Scalar,LocalOrdinal,GlobalOrdinal,Node>::applyInverseAdditive(const MultiVector& RHS, MultiVector& X) const {

    Scalar one = Teuchos::ScalarTraits<Scalar>::one();

    { // compute residual

      RCP<Teuchos::TimeMonitor> tmRes = getTimer("MueLu RefMaxwell: residual calculation");
      Utilities::Residual(*SM_Matrix_, X, RHS, *residual_);
    }

    { // restrict residual to sub-hierarchies

      if (implicitTranspose_) {
        {
          RCP<Teuchos::TimeMonitor> tmRes = getTimer("MueLu RefMaxwell: restriction coarse (1,1) (implicit)");
          P11_->apply(*residual_,*P11res_,Teuchos::TRANS);
        }
        if (!onlyBoundary22_) {
          RCP<Teuchos::TimeMonitor> tmD = getTimer("MueLu RefMaxwell: restriction (2,2) (implicit)");
          Dk_1_->apply(*residual_,*Dres_,Teuchos::TRANS);
        }
      } else {
#ifdef MUELU_HAVE_TPETRA
        if (Dk_1_T_R11_colMapsMatch_) {
          // Column maps of D_T and R11 match, and we're running Tpetra
          {
            RCP<Teuchos::TimeMonitor> tmD = getTimer("MueLu RefMaxwell: restrictions import");
            DTR11Tmp_->doImport(*residual_, *rcp_dynamic_cast<CrsMatrixWrap>(R11_)->getCrsMatrix()->getCrsGraph()->getImporter(), Xpetra::INSERT);
          }
          if (!onlyBoundary22_) {
            RCP<Teuchos::TimeMonitor> tmD = getTimer("MueLu RefMaxwell: restriction (2,2) (explicit)");
            rcp_dynamic_cast<TpetraCrsMatrix>(rcp_dynamic_cast<CrsMatrixWrap>(Dk_1_T_)->getCrsMatrix())->getTpetra_CrsMatrix()->localApply(toTpetra(*DTR11Tmp_),toTpetra(*Dres_),Teuchos::NO_TRANS);
          }
          {
            RCP<Teuchos::TimeMonitor> tmP11 = getTimer("MueLu RefMaxwell: restriction coarse (1,1) (explicit)");
            rcp_dynamic_cast<TpetraCrsMatrix>(rcp_dynamic_cast<CrsMatrixWrap>(R11_)->getCrsMatrix())->getTpetra_CrsMatrix()->localApply(toTpetra(*DTR11Tmp_),toTpetra(*P11res_),Teuchos::NO_TRANS);
          }
        } else
#endif
        {
          {
            RCP<Teuchos::TimeMonitor> tmP11 = getTimer("MueLu RefMaxwell: restriction coarse (1,1) (explicit)");
            R11_->apply(*residual_,*P11res_,Teuchos::NO_TRANS);
          }
          if (!onlyBoundary22_) {
            RCP<Teuchos::TimeMonitor> tmD = getTimer("MueLu RefMaxwell: restriction (2,2) (explicit)");
            Dk_1_T_->apply(*residual_,*Dres_,Teuchos::NO_TRANS);
          }
        }
      }
    }

    {
      RCP<Teuchos::TimeMonitor> tmSubSolves = getTimer("MueLu RefMaxwell: subsolves");

      // block diagonal preconditioner on 2x2 (V-cycle for diagonal blocks)

      if (!ImporterCoarse11_.is_null() && !implicitTranspose_) {
        RCP<Teuchos::TimeMonitor> tmH = getTimer("MueLu RefMaxwell: import coarse (1,1)");
        P11resTmp_->beginImport(*P11res_, *ImporterCoarse11_, Xpetra::INSERT);
      }
      if (!onlyBoundary22_ && !Importer22_.is_null() && !implicitTranspose_) {
        RCP<Teuchos::TimeMonitor> tm22 = getTimer("MueLu RefMaxwell: import (2,2)");
        DresTmp_->beginImport(*Dres_, *Importer22_, Xpetra::INSERT);
      }

      // iterate on coarse (1, 1) block
      if (!coarseA11_.is_null()) {
        if (!ImporterCoarse11_.is_null() && !implicitTranspose_)
          P11resTmp_->endImport(*P11res_, *ImporterCoarse11_, Xpetra::INSERT);

        RCP<Teuchos::TimeMonitor> tmH = getTimer("MueLu RefMaxwell: solve coarse (1,1)", coarseA11_->getRowMap()->getComm());

#if defined(HAVE_MUELU_STRATIMIKOS) && defined(HAVE_MUELU_THYRA)
        if (!thyraPrecOpH_.is_null()) {
          Scalar zero = Teuchos::ScalarTraits<Scalar>::zero();
          thyraPrecOpH_->apply(*P11resSubComm_, *P11xSubComm_, Teuchos::NO_TRANS, one, zero);
        }
        else
#endif
           HierarchyCoarse11_->Iterate(*P11resSubComm_, *P11xSubComm_, numItersCoarse11_, true);
      }

      // iterate on (2, 2) block
      if (!A22_.is_null()) {
        if (!onlyBoundary22_ && !Importer22_.is_null() && !implicitTranspose_)
          DresTmp_->endImport(*Dres_, *Importer22_, Xpetra::INSERT);

        RCP<Teuchos::TimeMonitor> tm22 = getTimer("MueLu RefMaxwell: solve (2,2)", A22_->getRowMap()->getComm());

#if defined(HAVE_MUELU_STRATIMIKOS) && defined(HAVE_MUELU_THYRA)
        if (!thyraPrecOp22_.is_null()) {
          Scalar zero = Teuchos::ScalarTraits<Scalar>::zero();
          thyraPrecOp22_->apply(*DresSubComm_, *DxSubComm_, Teuchos::NO_TRANS, one, zero);
        }
        else
#endif
          Hierarchy22_->Iterate(*DresSubComm_, *DxSubComm_, numIters22_, true);
      }

      if (coarseA11_.is_null() && !ImporterCoarse11_.is_null() && !implicitTranspose_)
        P11resTmp_->endImport(*P11res_, *ImporterCoarse11_, Xpetra::INSERT);
      if (A22_.is_null() && !onlyBoundary22_ && !Importer22_.is_null() && !implicitTranspose_)
        DresTmp_->endImport(*Dres_, *Importer22_, Xpetra::INSERT);
    }

    if (fuseProlongationAndUpdate_) {
      { // prolongate (1,1) block
        RCP<Teuchos::TimeMonitor> tmP11 = getTimer("MueLu RefMaxwell: prolongation coarse (1,1) (fused)");
        P11_->apply(*P11x_,X,Teuchos::NO_TRANS,one,one);
      }

      if (!onlyBoundary22_) { // prolongate (2,2) block
        RCP<Teuchos::TimeMonitor> tmD = getTimer("MueLu RefMaxwell: prolongation (2,2) (fused)");
        Dk_1_->apply(*Dx_,X,Teuchos::NO_TRANS,one,one);
      }
    } else {
      { // prolongate (1,1) block
        RCP<Teuchos::TimeMonitor> tmP11 = getTimer("MueLu RefMaxwell: prolongation coarse (1,1) (unfused)");
        P11_->apply(*P11x_,*residual_,Teuchos::NO_TRANS);
      }

      if (!onlyBoundary22_) { // prolongate (2,2) block
        RCP<Teuchos::TimeMonitor> tmD = getTimer("MueLu RefMaxwell: prolongation (2,2) (unfused)");
        Dk_1_->apply(*Dx_,*residual_,Teuchos::NO_TRANS,one,one);
      }

      { // update current solution
        RCP<Teuchos::TimeMonitor> tmUpdate = getTimer("MueLu RefMaxwell: update");
        X.update(one, *residual_, one);
      }
    }

  }


  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void RefMaxwell<Scalar,LocalOrdinal,GlobalOrdinal,Node>::solveH(const MultiVector& RHS, MultiVector& X) const {

    Scalar one = Teuchos::ScalarTraits<Scalar>::one();

    { // compute residual
      RCP<Teuchos::TimeMonitor> tmRes = getTimer("MueLu RefMaxwell: residual calculation");
      Utilities::Residual(*SM_Matrix_, X, RHS,*residual_);
      if (implicitTranspose_)
        P11_->apply(*residual_,*P11res_,Teuchos::TRANS);
      else
        R11_->apply(*residual_,*P11res_,Teuchos::NO_TRANS);
    }

    { // solve coarse (1,1) block
      if (!ImporterCoarse11_.is_null() && !implicitTranspose_) {
        RCP<Teuchos::TimeMonitor> tmH = getTimer("MueLu RefMaxwell: import coarse (1,1)");
        P11resTmp_->doImport(*P11res_, *ImporterCoarse11_, Xpetra::INSERT);
      }
      if (!coarseA11_.is_null()) {
        RCP<Teuchos::TimeMonitor> tmH = getTimer("MueLu RefMaxwell: solve coarse (1,1)", coarseA11_->getRowMap()->getComm());
        HierarchyCoarse11_->Iterate(*P11resSubComm_, *P11xSubComm_, numItersCoarse11_, true);
      }
    }

    { // update current solution
      RCP<Teuchos::TimeMonitor> tmUp = getTimer("MueLu RefMaxwell: update");
      P11_->apply(*P11x_,*residual_,Teuchos::NO_TRANS);
      X.update(one, *residual_, one);
    }

  }


  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void RefMaxwell<Scalar,LocalOrdinal,GlobalOrdinal,Node>::solve22(const MultiVector& RHS, MultiVector& X) const {

    if (onlyBoundary22_)
      return;

    Scalar one = Teuchos::ScalarTraits<Scalar>::one();

    { // compute residual
      RCP<Teuchos::TimeMonitor> tmRes = getTimer("MueLu RefMaxwell: residual calculation");
      Utilities::Residual(*SM_Matrix_, X, RHS, *residual_);
      if (implicitTranspose_)
        Dk_1_->apply(*residual_,*Dres_,Teuchos::TRANS);
      else
        Dk_1_T_->apply(*residual_,*Dres_,Teuchos::NO_TRANS);
    }

    { // solve (2,2) block
      if (!Importer22_.is_null() && !implicitTranspose_) {
        RCP<Teuchos::TimeMonitor> tm22 = getTimer("MueLu RefMaxwell: import (2,2)");
        DresTmp_->doImport(*Dres_, *Importer22_, Xpetra::INSERT);
      }
      if (!A22_.is_null()) {
        RCP<Teuchos::TimeMonitor> tm22 = getTimer("MueLu RefMaxwell: solve (2,2)", A22_->getRowMap()->getComm());
        Hierarchy22_->Iterate(*DresSubComm_, *DxSubComm_, numIters22_, true);
      }
    }

    { //update current solution
      RCP<Teuchos::TimeMonitor> tmUp = getTimer("MueLu RefMaxwell: update");
      Dk_1_->apply(*Dx_,*residual_,Teuchos::NO_TRANS);
      X.update(one, *residual_, one);
    }

  }


  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void RefMaxwell<Scalar,LocalOrdinal,GlobalOrdinal,Node>::apply (const MultiVector& RHS, MultiVector& X,
                                                                  Teuchos::ETransp /* mode */,
                                                                  Scalar /* alpha */,
                                                                  Scalar /* beta */) const {

    RCP<Teuchos::TimeMonitor> tm = getTimer("MueLu RefMaxwell: solve");

    // make sure that we have enough temporary memory
    if (!onlyBoundary11_ && X.getNumVectors() != P11res_->getNumVectors())
      allocateMemory(X.getNumVectors());

    { // apply pre-smoothing

      RCP<Teuchos::TimeMonitor> tmSm = getTimer("MueLu RefMaxwell: smoothing");

      PreSmoother_->Apply(X, RHS, use_as_preconditioner_);
    }

    // do solve for the 2x2 block system
    if(mode_=="additive")
      applyInverseAdditive(RHS,X);
    else if(mode_=="121") {
      solveH(RHS,X);
      solve22(RHS,X);
      solveH(RHS,X);
    } else if(mode_=="212") {
      solve22(RHS,X);
      solveH(RHS,X);
      solve22(RHS,X);
    } else if(mode_=="1")
      solveH(RHS,X);
    else if(mode_=="2")
      solve22(RHS,X);
    else if(mode_=="7") {
      solveH(RHS,X);
      { // apply pre-smoothing

        RCP<Teuchos::TimeMonitor> tmSm = getTimer("MueLu RefMaxwell: smoothing");

        PreSmoother_->Apply(X, RHS, false);
      }
      solve22(RHS,X);
      { // apply post-smoothing

        RCP<Teuchos::TimeMonitor> tmSm = getTimer("MueLu RefMaxwell: smoothing");

        PostSmoother_->Apply(X, RHS, false);
      }
      solveH(RHS,X);
    } else if(mode_=="none") {
      // do nothing
    }
    else
      applyInverseAdditive(RHS,X);

    { // apply post-smoothing

      RCP<Teuchos::TimeMonitor> tmSm = getTimer("MueLu RefMaxwell: smoothing");

      PostSmoother_->Apply(X, RHS, false);
    }

  }

    
  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  bool RefMaxwell<Scalar,LocalOrdinal,GlobalOrdinal,Node>::hasTransposeApply() const {
    return false;
  }


  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  RefMaxwell<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
  RefMaxwell(const Teuchos::RCP<Matrix> & SM_Matrix,
             Teuchos::ParameterList& List,
             bool ComputePrec)
    {

      int spaceNumber  = List.get<int>("refmaxwell: space number", 1);

      RCP<Matrix> Dk_1 = List.get<RCP<Matrix> >("Dk_1", Teuchos::null);
      RCP<Matrix> Dk_2 = List.get<RCP<Matrix> >("Dk_2", Teuchos::null);
      RCP<Matrix> D0   = List.get<RCP<Matrix> >("D0",   Teuchos::null);

      RCP<Matrix> M1_beta = Teuchos::null;
      if (List.isType<RCP<Matrix> >("M1_beta"))
        M1_beta = List.get<RCP<Matrix> >("M1_beta");
      else if ((spaceNumber == 1) && List.isType<RCP<Matrix> >("Ms"))
        M1_beta = List.get<RCP<Matrix> >("Ms");
      RCP<Matrix> M1_alpha = List.get<RCP<Matrix> >("M1_alpha", Teuchos::null);

      RCP<Matrix> Mk_one = Teuchos::null;
      if (List.isType<RCP<Matrix> >("Mk_one"))
        Mk_one = List.get<RCP<Matrix> >("Mk_one");
      else if ((spaceNumber == 1) && List.isType<RCP<Matrix> >("M1"))
        Mk_one = List.get<RCP<Matrix> >("M1");
      RCP<Matrix> Mk_1_one = List.get<RCP<Matrix> >("Mk_1_one", Teuchos::null);

      RCP<Matrix> invMk_1_invBeta = Teuchos::null;
      if (List.isType<RCP<Matrix> >("invMk_1_invBeta"))
        invMk_1_invBeta = List.get<RCP<Matrix> >("invMk_1_invBeta", Teuchos::null);
      else if ((spaceNumber == 1) && List.isType<RCP<Matrix> >("M0inv"))
        invMk_1_invBeta = List.get<RCP<Matrix> >("M0inv", Teuchos::null);
      RCP<Matrix> invMk_2_invAlpha = List.get<RCP<Matrix> >("invMk_2_invAlpha", Teuchos::null);

      RCP<MultiVector> Nullspace = List.get<RCP<MultiVector> >("Nullspace", Teuchos::null);
      RCP<RealValuedMultiVector> Coords = List.get<RCP<RealValuedMultiVector> >("Coordinates", Teuchos::null);

      if (spaceNumber == 1) {
        if (Dk_1.is_null())
          Dk_1 = D0;
        else if (D0.is_null())
          D0 = Dk_1;
      } else if (spaceNumber == 2) {
        if (Dk_2.is_null())
          Dk_2 = D0;
        else if (D0.is_null())
          D0 = Dk_2;
      }

      initialize(spaceNumber,
                 Dk_1, Dk_2, D0,
                 M1_beta, M1_alpha,
                 Mk_one, Mk_1_one,
                 invMk_1_invBeta, invMk_2_invAlpha,
                 Nullspace, Coords,
                 List);

      if (SM_Matrix != Teuchos::null)
        resetMatrix(SM_Matrix,ComputePrec);
    }


  template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void RefMaxwell<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
  initialize(const Teuchos::RCP<Matrix> & D0_Matrix,
             const Teuchos::RCP<Matrix> & Ms_Matrix,
             const Teuchos::RCP<Matrix> & M0inv_Matrix,
             const Teuchos::RCP<Matrix> & M1_Matrix,
             const Teuchos::RCP<MultiVector>  & Nullspace,
             const Teuchos::RCP<RealValuedMultiVector>  & Coords,
             Teuchos::ParameterList& List)
  {
    initialize(1,
               D0_Matrix, Teuchos::null, D0_Matrix,
               Ms_Matrix, Teuchos::null,
               M1_Matrix, Teuchos::null,
               M0inv_Matrix, Teuchos::null,
               Nullspace, Coords,
               List);
  }


  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void RefMaxwell<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
  initialize(const int k,
             const Teuchos::RCP<Matrix> & Dk_1,
             const Teuchos::RCP<Matrix> & Dk_2,
             const Teuchos::RCP<Matrix> & D0,
             const Teuchos::RCP<Matrix> & M1_beta,
             const Teuchos::RCP<Matrix> & M1_alpha,
             const Teuchos::RCP<Matrix> & Mk_one,
             const Teuchos::RCP<Matrix> & Mk_1_one,
             const Teuchos::RCP<Matrix> & invMk_1_invBeta,
             const Teuchos::RCP<Matrix> & invMk_2_invAlpha,
             const Teuchos::RCP<MultiVector> & Nullspace,
             const Teuchos::RCP<RealValuedMultiVector> & Coords,
             Teuchos::ParameterList& List) {

    spaceNumber_ = k;
    HierarchyCoarse11_    = Teuchos::null;
    Hierarchy22_          = Teuchos::null;
    PreSmoother_          = Teuchos::null;
    PostSmoother_         = Teuchos::null;
    disable_addon_        = false;
    disable_addon_22_     = true;
    mode_                 = "additive";

    // set parameters
    setParameters(List);

    // some pre-conditions
    TEUCHOS_ASSERT((k == 1) || (k == 2));
    // Need Dk_1
    TEUCHOS_ASSERT(Dk_1!=Teuchos::null);
    // Need D0 for aggregation
    TEUCHOS_ASSERT(D0!=Teuchos::null);

    // Need M1_beta for aggregation
    TEUCHOS_ASSERT(M1_beta!=Teuchos::null);
    // Need M1_alpha for aggregation if k>=1
    if (k >= 2)
      TEUCHOS_ASSERT(M1_alpha!=Teuchos::null);

    if (!disable_addon_) {
      // Need Mk_one and invMk_1_invBeta for addon11
      TEUCHOS_ASSERT(Mk_one!=Teuchos::null);
      TEUCHOS_ASSERT(invMk_1_invBeta!=Teuchos::null);
    }

    if ((k >= 2) && !disable_addon_22_) {
      // Need Dk_2, Mk_1_one and invMk_2_invAlpha for addon22
      TEUCHOS_ASSERT(Dk_2!=Teuchos::null);
      TEUCHOS_ASSERT(Mk_1_one!=Teuchos::null);
      TEUCHOS_ASSERT(invMk_2_invAlpha!=Teuchos::null);
    }

#ifdef HAVE_MUELU_DEBUG

    TEUCHOS_ASSERT(D0->getRangeMap()->isSameAs(*D0->getRowMap()));

    // M1_beta is square
    TEUCHOS_ASSERT(M1_beta->getDomainMap()->isSameAs(*M1_beta->getRangeMap()));
    TEUCHOS_ASSERT(M1_beta->getDomainMap()->isSameAs(*M1_beta->getRowMap()));

    // M1_beta is consistent with D0
    TEUCHOS_ASSERT(M1_beta->getDomainMap()->isSameAs(*D0->getRangeMap()));

    if (k == 1)
      TEUCHOS_ASSERT(D0 == Dk_1);

    if (k >= 2) {
      // M1_alpha is square
      TEUCHOS_ASSERT(M1_alpha->getDomainMap()->isSameAs(*M1_alpha->getRangeMap()));
      TEUCHOS_ASSERT(M1_alpha->getDomainMap()->isSameAs(*M1_alpha->getRowMap()));

      // M1_alpha is consistent with D0
      TEUCHOS_ASSERT(M1_alpha->getDomainMap()->isSameAs(*D0->getRangeMap()))
    }

    if (!disable_addon_) {
      // Mk_one is square
      TEUCHOS_ASSERT(Mk_one->getDomainMap()->isSameAs(*Mk_one->getRangeMap()));
      TEUCHOS_ASSERT(Mk_one->getDomainMap()->isSameAs(*Mk_one->getRowMap()));

      // Mk_one is consistent with Dk_1
      TEUCHOS_ASSERT(Mk_one->getDomainMap()->isSameAs(*Dk_1->getRangeMap()));

      // invMk_1_invBeta is square
      TEUCHOS_ASSERT(invMk_1_invBeta->getDomainMap()->isSameAs(*invMk_1_invBeta->getRangeMap()));
      TEUCHOS_ASSERT(invMk_1_invBeta->getDomainMap()->isSameAs(*invMk_1_invBeta->getRowMap()));

      // invMk_1_invBeta is consistent with Dk_1
      TEUCHOS_ASSERT(Mk_one->getDomainMap()->isSameAs(*Dk_1->getDomainMap()));
    }

    if ((k >= 2) && !disable_addon_22_) {
      // Mk_1_one is square
      TEUCHOS_ASSERT(Mk_1_one->getDomainMap()->isSameAs(*Mk_1_one->getRangeMap()));
      TEUCHOS_ASSERT(Mk_1_one->getDomainMap()->isSameAs(*Mk_1_one->getRowMap()));

      // Mk_1_one is consistent with Dk_1
      TEUCHOS_ASSERT(Mk_1_one->getDomainMap()->isSameAs(*Dk_1->getDomainMap()));

      // Mk_1_one is consistent with Dk_2
      TEUCHOS_ASSERT(Mk_1_one->getDomainMap()->isSameAs(*Dk_2->getRangeMap()));

      // invMk_2_invAlpha is square
      TEUCHOS_ASSERT(invMk_2_invAlpha->getDomainMap()->isSameAs(*invMk_2_invAlpha->getRangeMap()));
      TEUCHOS_ASSERT(invMk_2_invAlpha->getDomainMap()->isSameAs(*invMk_2_invAlpha->getRowMap()));

      // invMk_2_invAlpha is consistent with Dk_2
      TEUCHOS_ASSERT(invMk_2_invAlpha->getDomainMap()->isSameAs(*Dk_2->getDomainMap()));

      if (k == 2) {
        TEUCHOS_ASSERT(D0 == Dk_2);
      }

    }
#endif

    D0_ = D0;
    if (Dk_1->getRowMap()->lib() == Xpetra::UseTpetra) {
      // We will remove boundary conditions from Dk_1, and potentially change maps, so we copy the input.
      // Fortunately, Dk_1 is quite sparse.
      // We cannot use the Tpetra copy constructor, since it does not copy the graph.

      RCP<Matrix> Dk_1copy = MatrixFactory::Build(Dk_1->getRowMap(), Dk_1->getColMap(), 0);
      RCP<CrsMatrix> Dk_1copyCrs = rcp_dynamic_cast<CrsMatrixWrap>(Dk_1copy,true)->getCrsMatrix();
      ArrayRCP<const size_t> Dk_1rowptr_RCP;
      ArrayRCP<const LO>     Dk_1colind_RCP;
      ArrayRCP<const SC>     Dk_1vals_RCP;
      rcp_dynamic_cast<CrsMatrixWrap>(Dk_1,true)->getCrsMatrix()->getAllValues(Dk_1rowptr_RCP,
                                                                               Dk_1colind_RCP,
                                                                               Dk_1vals_RCP);

      ArrayRCP<size_t> Dk_1copyrowptr_RCP;
      ArrayRCP<LO>     Dk_1copycolind_RCP;
      ArrayRCP<SC>     Dk_1copyvals_RCP;
      Dk_1copyCrs->allocateAllValues(Dk_1vals_RCP.size(),Dk_1copyrowptr_RCP,Dk_1copycolind_RCP,Dk_1copyvals_RCP);
      Dk_1copyrowptr_RCP.deepCopy(Dk_1rowptr_RCP());
      Dk_1copycolind_RCP.deepCopy(Dk_1colind_RCP());
      Dk_1copyvals_RCP.deepCopy(Dk_1vals_RCP());
      Dk_1copyCrs->setAllValues(Dk_1copyrowptr_RCP,
                              Dk_1copycolind_RCP,
                              Dk_1copyvals_RCP);
      Dk_1copyCrs->expertStaticFillComplete(Dk_1->getDomainMap(), Dk_1->getRangeMap(),
                                          rcp_dynamic_cast<CrsMatrixWrap>(Dk_1,true)->getCrsMatrix()->getCrsGraph()->getImporter(),
                                          rcp_dynamic_cast<CrsMatrixWrap>(Dk_1,true)->getCrsMatrix()->getCrsGraph()->getExporter());
      Dk_1_ = Dk_1copy;
    } else
      Dk_1_ = MatrixFactory::BuildCopy(Dk_1);

    if ((!Dk_2.is_null()) && (Dk_2->getRowMap()->lib() == Xpetra::UseTpetra)) {
      // We will remove boundary conditions from Dk_2, and potentially change maps, so we copy the input.
      // Fortunately, Dk_2 is quite sparse.
      // We cannot use the Tpetra copy constructor, since it does not copy the graph.

      RCP<Matrix> Dk_2copy = MatrixFactory::Build(Dk_2->getRowMap(), Dk_2->getColMap(), 0);
      RCP<CrsMatrix> Dk_2copyCrs = rcp_dynamic_cast<CrsMatrixWrap>(Dk_2copy,true)->getCrsMatrix();
      ArrayRCP<const size_t> Dk_2rowptr_RCP;
      ArrayRCP<const LO>     Dk_2colind_RCP;
      ArrayRCP<const SC>     Dk_2vals_RCP;
      rcp_dynamic_cast<CrsMatrixWrap>(Dk_2,true)->getCrsMatrix()->getAllValues(Dk_2rowptr_RCP,
                                                                               Dk_2colind_RCP,
                                                                               Dk_2vals_RCP);

      ArrayRCP<size_t> Dk_2copyrowptr_RCP;
      ArrayRCP<LO>     Dk_2copycolind_RCP;
      ArrayRCP<SC>     Dk_2copyvals_RCP;
      Dk_2copyCrs->allocateAllValues(Dk_2vals_RCP.size(),Dk_2copyrowptr_RCP,Dk_2copycolind_RCP,Dk_2copyvals_RCP);
      Dk_2copyrowptr_RCP.deepCopy(Dk_2rowptr_RCP());
      Dk_2copycolind_RCP.deepCopy(Dk_2colind_RCP());
      Dk_2copyvals_RCP.deepCopy(Dk_2vals_RCP());
      Dk_2copyCrs->setAllValues(Dk_2copyrowptr_RCP,
                                Dk_2copycolind_RCP,
                                Dk_2copyvals_RCP);
      Dk_2copyCrs->expertStaticFillComplete(Dk_2->getDomainMap(), Dk_2->getRangeMap(),
                                            rcp_dynamic_cast<CrsMatrixWrap>(Dk_2,true)->getCrsMatrix()->getCrsGraph()->getImporter(),
                                            rcp_dynamic_cast<CrsMatrixWrap>(Dk_2,true)->getCrsMatrix()->getCrsGraph()->getExporter());
      Dk_2_ = Dk_2copy;
    } else if (!Dk_2.is_null())
      Dk_2_ = MatrixFactory::BuildCopy(Dk_2);

    M1_beta_  = M1_beta;
    M1_alpha_ = M1_alpha;

    Mk_one_   = Mk_one;
    Mk_1_one_ = Mk_1_one;

    invMk_1_invBeta_  = invMk_1_invBeta;
    invMk_2_invAlpha_ = invMk_2_invAlpha;

    Coords_       = Coords;
    Nullspace_    = Nullspace;

    if (!D0_.is_null())               dump(*D0_,               "D0.m");
    if (!Dk_1_.is_null())             dump(*Dk_1_,             "Dk_1_clean.m");
    if (!Dk_2_.is_null())             dump(*Dk_2_,             "Dk_2_clean.m");

    if (!M1_beta_.is_null())          dump(*M1_beta_,          "M1_beta.m");
    if (!M1_alpha_.is_null())         dump(*M1_alpha_,         "M1_alpha.m");

    if (!Mk_one_.is_null())           dump(*Mk_one_,           "Mk_one.m");
    if (!Mk_1_one_.is_null())         dump(*Mk_1_one_,         "Mk_1_one.m");

    if (!invMk_1_invBeta_.is_null())  dump(*invMk_1_invBeta_,  "invMk_1_invBeta.m");
    if (!invMk_2_invAlpha_.is_null()) dump(*invMk_2_invAlpha_, "invMk_2_invAlpha.m");

    if (!Coords_.is_null())           dumpCoords(*Coords_,     "coords.m");

  }


  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void RefMaxwell<Scalar,LocalOrdinal,GlobalOrdinal,Node>::
  describe(Teuchos::FancyOStream& out, const Teuchos::EVerbosityLevel /* verbLevel */) const {

    std::ostringstream oss;

    RCP<const Teuchos::Comm<int> > comm = SM_Matrix_->getDomainMap()->getComm();

#ifdef HAVE_MPI
    int root;
    if (!coarseA11_.is_null())
      root = comm->getRank();
    else
      root = -1;

    int actualRoot;
    reduceAll(*comm, Teuchos::REDUCE_MAX, root, Teuchos::ptr(&actualRoot));
    root = actualRoot;
#endif


    oss << "\n--------------------------------------------------------------------------------\n" <<
      "---                            RefMaxwell Summary                            ---\n"
      "--------------------------------------------------------------------------------" << std::endl;
    oss << std::endl;

    GlobalOrdinal numRows;
    GlobalOrdinal nnz;

    SM_Matrix_->getRowMap()->getComm()->barrier();

    numRows = SM_Matrix_->getGlobalNumRows();
    nnz = SM_Matrix_->getGlobalNumEntries();

    Xpetra::global_size_t tt = numRows;
    int rowspacer = 3; while (tt != 0) { tt /= 10; rowspacer++; }
    tt = nnz;
    int nnzspacer = 2; while (tt != 0) { tt /= 10; nnzspacer++; }

    oss  << "block " << std::setw(rowspacer) << " rows " << std::setw(nnzspacer) << " nnz " << std::setw(9) << " nnz/row" << std::endl;
    oss << "(1, 1)" << std::setw(rowspacer) << numRows << std::setw(nnzspacer) << nnz << std::setw(9) << as<double>(nnz) / numRows << std::endl;

    if (!A22_.is_null()) {
      // ToDo: make sure that this is printed correctly
      numRows = A22_->getGlobalNumRows();
      nnz = A22_->getGlobalNumEntries();

      oss << "(2, 2)" << std::setw(rowspacer) << numRows << std::setw(nnzspacer) << nnz << std::setw(9) << as<double>(nnz) / numRows << std::endl;
    }

    oss << std::endl;

    {
      if (PreSmoother_ != null && PreSmoother_ == PostSmoother_)
        oss << "Smoother both : " << PreSmoother_->description() << std::endl;
      else {
        oss << "Smoother pre  : "
            << (PreSmoother_ != null ?  PreSmoother_->description() : "no smoother") << std::endl;
        oss << "Smoother post : "
            << (PostSmoother_ != null ?  PostSmoother_->description() : "no smoother") << std::endl;
      }
    }
    oss << std::endl;

    std::string outstr = oss.str();

#ifdef HAVE_MPI
    RCP<const Teuchos::MpiComm<int> > mpiComm = rcp_dynamic_cast<const Teuchos::MpiComm<int> >(comm);
    MPI_Comm rawComm = (*mpiComm->getRawMpiComm())();

    int strLength = outstr.size();
    MPI_Bcast(&strLength, 1, MPI_INT, root, rawComm);
    if (comm->getRank() != root)
      outstr.resize(strLength);
    MPI_Bcast(&outstr[0], strLength, MPI_CHAR, root, rawComm);
#endif

    out << outstr;

    if (!HierarchyCoarse11_.is_null())
      HierarchyCoarse11_->describe(out, GetVerbLevel());

    if (!Hierarchy22_.is_null())
      Hierarchy22_->describe(out, GetVerbLevel());

    if (IsPrint(Statistics2)) {
      // Print the grid of processors
      std::ostringstream oss2;

      oss2 << "Sub-solver distribution over ranks" << std::endl;
      oss2 << "( (1,1) block only is indicated by '1', (2,2) block only by '2', and both blocks by 'B' and none by '.')" << std::endl;

      int numProcs = comm->getSize();
#ifdef HAVE_MPI
      RCP<const Teuchos::MpiComm<int> > tmpic = rcp_dynamic_cast<const Teuchos::MpiComm<int> >(comm);
      TEUCHOS_TEST_FOR_EXCEPTION(tmpic == Teuchos::null, Exceptions::RuntimeError, "Cannot cast base Teuchos::Comm to Teuchos::MpiComm object.");
      RCP<const Teuchos::OpaqueWrapper<MPI_Comm> > rawMpiComm = tmpic->getRawMpiComm();
#endif

      char status = 0;
      if (!coarseA11_.is_null())
        status += 1;
      if (!A22_.is_null())
        status += 2;
      std::vector<char> states(numProcs, 0);
#ifdef HAVE_MPI
      MPI_Gather(&status, 1, MPI_CHAR, &states[0], 1, MPI_CHAR, 0, *rawMpiComm);
#else
      states.push_back(status);
#endif

      int rowWidth = std::min(Teuchos::as<int>(ceil(sqrt(numProcs))), 100);
      for (int proc = 0; proc < numProcs; proc += rowWidth) {
        for (int j = 0; j < rowWidth; j++)
          if (proc + j < numProcs)
            if (states[proc+j] == 0)
              oss2 << ".";
            else if (states[proc+j] == 1)
              oss2 << "1";
            else if (states[proc+j] == 2)
              oss2 << "2";
            else
              oss2 << "B";
          else
            oss2 << " ";

        oss2 << "      " << proc << ":" << std::min(proc + rowWidth, numProcs) - 1 << std::endl;
      }
      oss2 << std::endl;
      GetOStream(Statistics2) << oss2.str();
    }


  }


} // namespace

#define MUELU_REFMAXWELL_SHORT
#endif //ifdef MUELU_REFMAXWELL_DEF_HPP
