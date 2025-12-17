// @HEADER
// *****************************************************************************
//             Xpetra: A linear algebra interface package
//
// Copyright 2012 NTESS and the Xpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_ScalarTraits.hpp>
#include <Teuchos_TimeMonitor.hpp>
#include <Teuchos_StackedTimer.hpp>

#include <Xpetra_ConfigDefs.hpp>

#include <Teuchos_RCP.hpp>

#include <Xpetra_DefaultPlatform.hpp>

#include <Xpetra_Map.hpp>
#include <Xpetra_Matrix.hpp>
#include <Xpetra_CrsMatrix.hpp>
#include <Xpetra_CrsMatrixWrap.hpp>
#include <Xpetra_TpetraCrsMatrix.hpp>
#include <MatrixMarket_Tpetra.hpp>
#include <Xpetra_MapFactory.hpp>
#include <Xpetra_MatrixMatrix.hpp>
#include <Xpetra_IO.hpp>

#include <Xpetra_EpetraCrsMatrix.hpp>
#include <stdexcept>
#include "Epetra_CrsMatrix.h"
#include "EpetraExt_Transpose_RowMatrix.h"

#include "ml_common.h"
#include "ml_comm.h"
#include "ml_operator.h"
#include "ml_epetra_utils.h"

using Teuchos::RCP;
using Teuchos::rcp;
using Teuchos::rcp_dynamic_cast;
using Teuchos::rcpFromRef;

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
Epetra_CrsMatrix& Op2NonConstEpetraCrs(Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>& Op) {
  using CrsMatrixWrap   = Xpetra::CrsMatrixWrap<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
  using EpetraCrsMatrix = Xpetra::EpetraCrsMatrixT<GlobalOrdinal, Node>;
  try {
    CrsMatrixWrap& crsOp = dynamic_cast<CrsMatrixWrap&>(Op);
    try {
      EpetraCrsMatrix& tmp_ECrsMtx = dynamic_cast<EpetraCrsMatrix&>(*crsOp.getCrsMatrix());
      return *tmp_ECrsMtx.getEpetra_CrsMatrixNonConst();
    } catch (std::bad_cast&) {
      throw std::runtime_error("Cast from Xpetra::CrsMatrix to Xpetra::EpetraCrsMatrix failed");
    }
  } catch (std::bad_cast&) {
    throw std::runtime_error("Cast from Xpetra::Matrix to Xpetra::CrsMatrixWrap failed");
  }
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>> ML_Multiply(Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>& Op1, Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>& Op2) {
  using Matrix          = Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
  using CrsMatrix       = Xpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
  using CrsMatrixWrap   = Xpetra::CrsMatrixWrap<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
  using EpetraCrsMatrix = Xpetra::EpetraCrsMatrixT<GlobalOrdinal, Node>;

  auto op1e  = Op2NonConstEpetraCrs(Op1);
  auto op2e  = Op2NonConstEpetraCrs(Op2);
  auto prode = Epetra_MatrixMult(&op1e, &op2e);

  RCP<Epetra_CrsMatrix> rcpA(prode);
  RCP<EpetraCrsMatrix> AA = rcp(new EpetraCrsMatrix(rcpA));
  RCP<CrsMatrix> AAA      = Teuchos::rcp_implicit_cast<CrsMatrix>(AA);
  RCP<Matrix> AAAA        = rcp(new CrsMatrixWrap(AAA));
  return AAAA;
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>> Transpose(Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>& Op, bool /* optimizeTranspose */ = false, const std::string& label = std::string(), const Teuchos::RCP<Teuchos::ParameterList>& params = Teuchos::null) {
  using Matrix          = Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
  using CrsMatrix       = Xpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
  using CrsMatrixWrap   = Xpetra::CrsMatrixWrap<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
  using EpetraCrsMatrix = Xpetra::EpetraCrsMatrixT<GlobalOrdinal, Node>;
  switch (Op.getRowMap()->lib()) {
    case Xpetra::UseTpetra: {
      const Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>& tpetraOp = toTpetra(Op);

      // Compute the transpose A of the Tpetra matrix tpetraOp.
      RCP<Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>> A;
      Tpetra::RowMatrixTransposer<Scalar, LocalOrdinal, GlobalOrdinal, Node> transposer(rcpFromRef(tpetraOp), label);

      {
        using Teuchos::ParameterList;
        using Teuchos::rcp;
        RCP<ParameterList> transposeParams = params.is_null() ? rcp(new ParameterList) : rcp(new ParameterList(*params));
        transposeParams->set("sort", false);
        A = transposer.createTranspose(transposeParams);
      }

      RCP<Xpetra::TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>> AA = rcp(new Xpetra::TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>(A));
      RCP<CrsMatrix> AAA                                                         = Teuchos::rcp_implicit_cast<CrsMatrix>(AA);
      RCP<Matrix> AAAA                                                           = rcp(new CrsMatrixWrap(AAA));

      if (Op.IsView("stridedMaps"))
        AAAA->CreateView("stridedMaps", Teuchos::rcpFromRef(Op), true /*doTranspose*/);

      return AAAA;
    }
    case Xpetra::UseEpetra: {
      // Epetra case
      Epetra_CrsMatrix& epetraOp = Op2NonConstEpetraCrs(Op);
      EpetraExt::RowMatrix_Transpose transposer;
      Epetra_CrsMatrix* A = dynamic_cast<Epetra_CrsMatrix*>(&transposer(epetraOp));
      transposer.ReleaseTranspose();  // So we can keep A in Muelu...

      RCP<Epetra_CrsMatrix> rcpA(A);
      RCP<EpetraCrsMatrix> AA = rcp(new EpetraCrsMatrix(rcpA));
      RCP<CrsMatrix> AAA      = Teuchos::rcp_implicit_cast<CrsMatrix>(AA);
      RCP<Matrix> AAAA        = rcp(new CrsMatrixWrap(AAA));

      if (Op.IsView("stridedMaps"))
        AAAA->CreateView("stridedMaps", Teuchos::rcpFromRef(Op), true /*doTranspose*/);

      return AAAA;
    }
  }

  TEUCHOS_UNREACHABLE_RETURN(Teuchos::null);
}

int main(int argc, char** argv) {
  Teuchos::oblackholestream blackhole;
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, &blackhole);
  auto comm = Teuchos::DefaultComm<int>::getComm();

  auto out = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));

  using Scalar        = double;
  using LocalOrdinal  = int;
  using GlobalOrdinal = int;
  using Node          = Xpetra::EpetraNode;

  using MapFactory = Xpetra::MapFactory<LocalOrdinal, GlobalOrdinal, Node>;
  using Matrix     = Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>;

  Teuchos::RCP<Teuchos::StackedTimer> stacked_timer = rcp(new Teuchos::StackedTimer("MatrixMatrix Comparison"));
  Teuchos::TimeMonitor::setStackedTimer(stacked_timer);

  int numRepeats = 100;

  // 224 ranks
  std::string map_file        = "rowmap_A_0.m";
  std::string coarse_map_file = "domainmap_P_1.m";
  std::string A_file          = "A_0.m";
  std::string P_file          = "P_1.m";

  // std::string map_file = "rowmap_A_3.m";
  // std::string coarse_map_file = "domainmap_P_4.m";
  // std::string A_file = "A_3.m";
  // std::string P_file = "P_4.m";

  // 34 ranks
  // std::string map_file        = "rowmap_A_2.m";
  // std::string coarse_map_file = "domainmap_P_3.m";
  // std::string A_file          = "A_2.m";
  // std::string P_file          = "P_3.m";

  auto lib = Xpetra::UseTpetra;

  auto map        = Xpetra::IO<Scalar, LocalOrdinal, GlobalOrdinal, Node>::ReadMap(map_file, lib, comm, false);
  auto coarse_map = Xpetra::IO<Scalar, LocalOrdinal, GlobalOrdinal, Node>::ReadMap(coarse_map_file, lib, comm, false);
  auto xA         = Xpetra::IO<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Read(A_file, map, Teuchos::null, map, map);
  auto xP         = Xpetra::IO<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Read(P_file, map, Teuchos::null, coarse_map, map);

  // Epetra
  {
    auto ep_map        = MapFactory::Build(Xpetra::UseEpetra, map->getGlobalNumElements(), map->getLocalElementList(), 0, comm);
    auto A_colmap      = xA->getColMap();
    auto ep_A_colmap   = MapFactory::Build(Xpetra::UseEpetra, A_colmap->getGlobalNumElements(), A_colmap->getLocalElementList(), 0, comm);
    auto ep_coarse_map = MapFactory::Build(Xpetra::UseEpetra, coarse_map->getGlobalNumElements(), coarse_map->getLocalElementList(), 0, comm);
    auto P_colmap      = xP->getColMap();
    auto ep_P_colmap   = MapFactory::Build(Xpetra::UseEpetra, P_colmap->getGlobalNumElements(), P_colmap->getLocalElementList(), 0, comm);

    auto eA = Xpetra::MatrixFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Build(xA->getLocalMatrixHost(), ep_map, ep_A_colmap, ep_map, ep_map);
    auto eP = Xpetra::MatrixFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Build(xP->getLocalMatrixHost(), ep_map, ep_P_colmap, ep_coarse_map, ep_map);

    Xpetra::IO<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Write("epetra_" + A_file, *eA);
    Xpetra::IO<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Write("epetra_" + P_file, *eP);

    for (int iter = 0; iter < numRepeats; ++iter) {
      auto timer = rcp(new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("Epetra implicit")));
      RCP<Matrix> yAP;
      {
        auto timer2 = rcp(new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("Epetra implicit AP")));
        yAP         = Xpetra::MatrixMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Multiply(*eA, false, *eP, false, yAP, *out, true, true);
      }
      RCP<Matrix> yPtAP;
      {
        auto timer2 = rcp(new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("Epetra implicit PtAP")));
        yPtAP       = Xpetra::MatrixMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Multiply(*eP, true, *yAP, false, yPtAP, *out, true, true);
      }
    }

    for (int iter = 0; iter < numRepeats; ++iter) {
      auto timer = rcp(new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("Epetra explicit")));
      RCP<Matrix> yAP;
      {
        auto timer2 = rcp(new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("Epetra explicit AP")));
        yAP         = Xpetra::MatrixMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Multiply(*eA, false, *eP, false, yAP, *out, true, true);
      }
      RCP<Matrix> yR;
      {
        auto timer2 = rcp(new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("Epetra explicit Pt")));
        // By default, we don't need global constants for transpose
        auto Tparams = rcp(new Teuchos::ParameterList());
        Tparams->set("compute global constants: temporaries", Tparams->get("compute global constants: temporaries", false));
        Tparams->set("compute global constants", Tparams->get("compute global constants", false));
        yR = Transpose(*eP, true, "TransP", Tparams);
      }
      RCP<Matrix> yRAP;
      {
        auto timer2 = rcp(new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("Epetra explicit RAP")));
        yRAP        = Xpetra::MatrixMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Multiply(*yR, false, *yAP, false, yRAP, *out, true, true);
      }
    }

    auto Ae = Op2NonConstEpetraCrs(*eA);
    auto Pe = Op2NonConstEpetraCrs(*eP);

    ML_Comm *comm, *temp;
    // temp = global_comm;
    ML_Comm_Create(&comm);

    const Epetra_MpiComm* Mcomm = dynamic_cast<const Epetra_MpiComm*>(&Ae.Comm());
    if (Mcomm) ML_Comm_Set_UsrComm(comm, Mcomm->GetMpiComm());

    ML_Operator *A_ml, *P_ml, *AP_ml;
    A_ml  = ML_Operator_Create(comm);
    P_ml  = ML_Operator_Create(comm);
    AP_ml = ML_Operator_Create(comm);
    ML_Operator_WrapEpetraMatrix(&Ae, A_ml);
    ML_Operator_WrapEpetraMatrix(&Pe, P_ml);
    for (int iter = 0; iter < numRepeats; ++iter) {
      auto timer = rcp(new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("ML explicit")));
      RCP<Matrix> yAP;
      {
        auto timer2 = rcp(new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("ML explicit AP")));
        AP_ml       = ML_Operator_Create(comm);
        ML_2matmult(A_ml, P_ml, AP_ml, ML_EpetraCRS_MATRIX);
        ML_Operator_Destroy(&AP_ml);
      }
      // RCP<Matrix> yR;
      // {
      //   auto timer2 = rcp(new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("Epetra explicit AP")));
      //   // By default, we don't need global constants for transpose
      //   auto Tparams = rcp(new Teuchos::ParameterList() );
      //   Tparams->set("compute global constants: temporaries", Tparams->get("compute global constants: temporaries", false));
      //   Tparams->set("compute global constants", Tparams->get("compute global constants", false));
      //   yR = Transpose(*eP, true, "TransP", Tparams);
      // }
      // RCP<Matrix> yRAP;
      // {
      //   auto timer2 = rcp(new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("ML explicit RAP")));
      //   yRAP        = ML_Multiply(*yR, *yAP);
      // }
    }

    ML_Comm_Destroy(&comm);
    // global_comm = temp;

    /* Need to blow about BBt_ml but keep epetra stuff */

    // result = (Epetra_RowMatrix *) AP_ml->data;
    ML_Operator_Destroy(&A_ml);
    ML_Operator_Destroy(&P_ml);
  }

  // Tpetra
  {
    for (int iter = 0; iter < numRepeats; ++iter) {
      auto timer = rcp(new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("Tpetra implicit")));
      RCP<Matrix> yAP;
      {
        auto timer2 = rcp(new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("Tpetra implicit AP")));
        yAP         = Xpetra::MatrixMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Multiply(*xA, false, *xP, false, yAP, *out, true, true);
      }
      RCP<Matrix> yPtAP;
      {
        auto timer2 = rcp(new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("Tpetra implicit PtAP")));
        yPtAP       = Xpetra::MatrixMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Multiply(*xP, true, *yAP, false, yPtAP, *out, true, true);
      }
    }

    for (int iter = 0; iter < numRepeats; ++iter) {
      auto timer = rcp(new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("Tpetra explicit")));
      RCP<Matrix> yAP;
      {
        auto timer2 = rcp(new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("Tpetra explicit AP")));
        yAP         = Xpetra::MatrixMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Multiply(*xA, false, *xP, false, yAP, *out, true, true);
      }
      RCP<Matrix> yR;
      {
        auto timer2 = rcp(new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("Tpetra explicit Pt")));
        // By default, we don't need global constants for transpose
        auto Tparams = rcp(new Teuchos::ParameterList());
        Tparams->set("compute global constants: temporaries", Tparams->get("compute global constants: temporaries", false));
        Tparams->set("compute global constants", Tparams->get("compute global constants", false));
        yR = Transpose(*xP, true, "TransP", Tparams);
      }
      RCP<Matrix> yRAP;
      {
        auto timer2 = rcp(new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("Tpetra explicit RAP")));
        yRAP        = Xpetra::MatrixMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Multiply(*yR, false, *yAP, false, yRAP, *out, true, true);
      }
    }
  }

  stacked_timer->stopBaseTimer();
  Teuchos::StackedTimer::OutputOptions options;
  options.output_fraction = options.output_histogram = options.output_minmax = true;
  stacked_timer->report(*out, comm, options);
}
