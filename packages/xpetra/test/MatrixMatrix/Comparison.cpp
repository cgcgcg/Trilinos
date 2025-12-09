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

using Teuchos::RCP;
using Teuchos::rcp;
using Teuchos::rcp_dynamic_cast;
using Teuchos::rcpFromRef;

int main(int argc, char **argv) {
  Teuchos::oblackholestream blackhole;
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, &blackhole);
  auto comm = Teuchos::DefaultComm<int>::getComm();

  auto out = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));

  using Scalar        = double;
  using LocalOrdinal  = int;
  using GlobalOrdinal = int;
  using Node          = Xpetra::EpetraNode;

  using MapFactory    = Xpetra::MapFactory<LocalOrdinal, GlobalOrdinal, Node>;
  using Matrix        = Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
  using CrsMatrix     = Xpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
  using CrsMatrixWrap = Xpetra::CrsMatrixWrap<Scalar, LocalOrdinal, GlobalOrdinal, Node>;

  Teuchos::RCP<Teuchos::StackedTimer> stacked_timer = rcp(new Teuchos::StackedTimer("MatrixMatrix Comparison"));
  Teuchos::TimeMonitor::setStackedTimer(stacked_timer);

  int numRepeats = 100;

  // std::string map_file = "rowmap_A_3.m";
  // std::string coarse_map_file = "domainmap_P_4.m";
  // std::string A_file = "A_3.m";
  // std::string P_file = "P_4.m";

  std::string map_file        = "rowmap_A_2.m";
  std::string coarse_map_file = "domainmap_P_3.m";
  std::string A_file          = "A_2.m";
  std::string P_file          = "P_3.m";

  auto lib = Xpetra::UseTpetra;

  auto map        = Xpetra::IO<Scalar, LocalOrdinal, GlobalOrdinal, Node>::ReadMap(map_file, lib, comm, false);
  auto coarse_map = Xpetra::IO<Scalar, LocalOrdinal, GlobalOrdinal, Node>::ReadMap(coarse_map_file, lib, comm, false);
  auto xA         = Xpetra::IO<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Read(A_file, map, Teuchos::null, map, map);
  auto xP         = Xpetra::IO<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Read(P_file, map, Teuchos::null, coarse_map, map);

  // Epetra
  {
    using MA = Xpetra::EpetraCrsMatrixT<GlobalOrdinal, Node>;

    auto lib = Xpetra::UseEpetra;

    auto ep_map        = Xpetra::MapFactory<LocalOrdinal, GlobalOrdinal, Node>::Build(lib, map->getGlobalNumElements(), map->getLocalElementList(), 0, comm);
    auto A_colmap      = xA->getColMap();
    auto ep_A_colmap   = Xpetra::MapFactory<LocalOrdinal, GlobalOrdinal, Node>::Build(lib, A_colmap->getGlobalNumElements(), A_colmap->getLocalElementList(), 0, comm);
    auto ep_coarse_map = Xpetra::MapFactory<LocalOrdinal, GlobalOrdinal, Node>::Build(lib, coarse_map->getGlobalNumElements(), coarse_map->getLocalElementList(), 0, comm);
    auto P_colmap      = xP->getColMap();
    auto ep_P_colmap   = Xpetra::MapFactory<LocalOrdinal, GlobalOrdinal, Node>::Build(lib, P_colmap->getGlobalNumElements(), P_colmap->getLocalElementList(), 0, comm);

    auto eA = Xpetra::MatrixFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Build(xA->getLocalMatrixHost(), ep_map, ep_A_colmap, ep_map, ep_map);
    auto eP = Xpetra::MatrixFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Build(xP->getLocalMatrixHost(), ep_map, ep_P_colmap, ep_coarse_map, ep_map);

    for (int iter = 0; iter < numRepeats; ++iter) {
      auto timer = rcp(new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("Epetra")));
      RCP<Matrix> yAP;
      {
        auto timer2 = rcp(new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("Epetra AP")));
        yAP         = Xpetra::MatrixMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Multiply(*eA, false, *eP, false, yAP, *out, true, true);
      }
      RCP<Matrix> yPtAP;
      {
        auto timer2 = rcp(new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("Epetra PtAP")));
        yPtAP       = Xpetra::MatrixMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Multiply(*eP, true, *yAP, false, yPtAP, *out, true, true);
      }
    }
  }

  // Tpetra
  {
    for (int iter = 0; iter < numRepeats; ++iter) {
      auto timer = rcp(new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("Tpetra")));
      RCP<Matrix> yAP;
      {
        auto timer2 = rcp(new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("Tpetra AP")));
        yAP         = Xpetra::MatrixMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Multiply(*xA, false, *xP, false, yAP, *out, true, true);
      }
      RCP<Matrix> yPtAP;
      {
        auto timer2 = rcp(new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("Tpetra PtAP")));
        yPtAP       = Xpetra::MatrixMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Multiply(*xP, true, *yAP, false, yPtAP, *out, true, true);
      }
    }
  }

  stacked_timer->stopBaseTimer();
  Teuchos::StackedTimer::OutputOptions options;
  options.output_fraction = options.output_histogram = options.output_minmax = true;
  stacked_timer->report(*out, comm, options);
}
