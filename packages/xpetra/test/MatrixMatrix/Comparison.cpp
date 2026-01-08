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
#include <Xpetra_MultiVector.hpp>
#include <Xpetra_CrsMatrix.hpp>
#include <Xpetra_CrsMatrixWrap.hpp>
#include <Xpetra_TpetraCrsMatrix.hpp>
#include <MatrixMarket_Tpetra.hpp>
#include <Xpetra_MapFactory.hpp>
#include <Xpetra_MatrixMatrix.hpp>
#include <Xpetra_MultiVectorFactory.hpp>
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

int ML_Gen_Restrictor_TransP2(ML_Operator *Rmat,
                              ML_Operator *Pmat) {
  // ML_Operator *Pmat, *Rmat;
  int *row_ptr, *colbuf, *cols;
  int isize, osize, i, j, N_nzs, flag, length, sum, new_sum;
  int Nneighbors, *neigh_list, *send_list, *rcv_list, Nsend, Nrcv;
  void *data = NULL;
  double *valbuf, *vals;
  int (*getrow)(ML_Operator *, int, int *, int, int *, double *, int *) = NULL;
  struct ML_CSR_MSRdata *temp;
  int Nghost = 0, Nghost2 = 0;
  int *remap, remap_leng;
  ML_CommInfoOP *c_info, **c2_info;

  /* pull out things from ml_handle */

  temp   = (struct ML_CSR_MSRdata *)Pmat->data;
  isize  = Pmat->outvec_leng;
  osize  = Pmat->invec_leng;
  data   = (void *)Pmat;
  getrow = Pmat->getrow->func_ptr;

  /* transpose Pmat's communication list. This means that PRE communication */
  /* is replaced by POST, ML_OVERWRITE is replaced by ML_ADD, and the send  */
  /* send and receive lists are swapped.                                    */

  c_info     = Pmat->getrow->pre_comm;
  Nneighbors = ML_CommInfoOP_Get_Nneighbors(c_info);
  neigh_list = ML_CommInfoOP_Get_neighbors(c_info);
  remap_leng = osize;
  Nrcv       = 0;
  Nsend      = 0;
  for (i = 0; i < Nneighbors; i++) {
    Nrcv += ML_CommInfoOP_Get_Nrcvlist(c_info, neigh_list[i]);
    Nsend += ML_CommInfoOP_Get_Nsendlist(c_info, neigh_list[i]);
  }
  remap_leng = osize + Nrcv + Nsend;
  remap      = (int *)ML_allocate(remap_leng * sizeof(int));
  for (i = 0; i < osize; i++) remap[i] = i;
  for (i = osize; i < osize + Nrcv + Nsend; i++)
    remap[i] = -1;

  c2_info = &(Rmat->getrow->post_comm);
  ML_CommInfoOP_Set_neighbors(c2_info, Nneighbors,
                              neigh_list, ML_ADD, remap, remap_leng);
  ML_free(remap);
  for (i = 0; i < Nneighbors; i++) {
    Nsend     = ML_CommInfoOP_Get_Nsendlist(c_info, neigh_list[i]);
    send_list = ML_CommInfoOP_Get_sendlist(c_info, neigh_list[i]);
    Nrcv      = ML_CommInfoOP_Get_Nrcvlist(c_info, neigh_list[i]);
    Nghost += Nrcv;
    rcv_list = ML_CommInfoOP_Get_rcvlist(c_info, neigh_list[i]);
    /* handle empty rows ... i.e. ghost variables not used */
    if (rcv_list != NULL) {
      for (j = 0; j < Nrcv; j++) {
        if (rcv_list[j] > Nghost2 + osize - 1)
          Nghost2 = rcv_list[j] - osize + 1;
      }
    }

    ML_CommInfoOP_Set_exch_info(*c2_info, neigh_list[i], Nsend, send_list,
                                Nrcv, rcv_list);
    if (send_list != NULL) ML_free(send_list);
    if (rcv_list != NULL) ML_free(rcv_list);
  }
  if (Nghost2 > Nghost) Nghost = Nghost2;
  if (neigh_list != NULL) ML_free(neigh_list);

  row_ptr = (int *)ML_allocate(sizeof(int) * (Nghost + osize + 1));
  colbuf  = (int *)ML_allocate(sizeof(int) * (Nghost + osize + 1));
  valbuf  = (double *)ML_allocate(sizeof(double) * (Nghost + osize + 1));

  /* count the total number of nonzeros and compute */
  /* the length of each row in the transpose.       */

  for (i = 0; i < Nghost + osize; i++) row_ptr[i] = 0;

  N_nzs = 0;
  for (i = 0; i < isize; i++) {
    flag = getrow((ML_Operator *)data, 1, &i, Nghost + osize + 1, colbuf, valbuf, &length);
    if (flag == 0) pr_error("ML_Transpose_Prolongator: sizes don't work\n");
    N_nzs += length;
    for (j = 0; j < length; j++)
      row_ptr[colbuf[j]]++;
  }

  cols = (int *)ML_allocate(sizeof(int) * (N_nzs + 1));
  vals = (double *)ML_allocate(sizeof(double) * (N_nzs + 1));
  if (vals == NULL)
    pr_error("ML_Gen_Restrictor_TransP: Out of space\n");

  /* set 'row_ptr' so it points to the beginning of each row */

  sum = 0;
  for (i = 0; i < Nghost + osize; i++) {
    new_sum    = sum + row_ptr[i];
    row_ptr[i] = sum;
    sum        = new_sum;
  }
  row_ptr[osize + Nghost] = sum;

  /* read in the prolongator matrix and store transpose in Rmat */

  for (i = 0; i < isize; i++) {
    getrow((ML_Operator *)data, 1, &i, Nghost + osize + 1, colbuf, valbuf, &length);
    for (j = 0; j < length; j++) {
      cols[row_ptr[colbuf[j]]]   = i;
      vals[row_ptr[colbuf[j]]++] = valbuf[j];
    }
  }

  /* row_ptr[i] now points to the i+1th row.    */
  /* Reset it so that it points to the ith row. */

  for (i = Nghost + osize; i > 0; i--)
    row_ptr[i] = row_ptr[i - 1];
  row_ptr[0] = 0;

  ML_free(valbuf);
  ML_free(colbuf);

  // std::cout << "HERE "<< N_nzs << std::endl;

  /* store the matrix into ML */

  temp          = (struct ML_CSR_MSRdata *)ML_allocate(sizeof(struct ML_CSR_MSRdata));
  temp->columns = cols;
  temp->values  = vals;
  temp->rowptr  = row_ptr;

  Rmat->data_destroy = ML_CSR_MSRdata_Destroy;
  // ML_Init_Restrictor(ml_handle, level, level2, isize, osize, (void *) temp);
  ML_Operator_Set_ApplyFuncData(Rmat, isize, osize,
                                (void *)temp, osize, NULL, 0);
  ML_Operator_Set_ApplyFunc(Rmat, CSR_matvec);
  ML_Operator_Set_Getrow(Rmat,
                         Nghost + osize, CSR_getrow);
  Rmat->N_nonzeros = N_nzs;
  return (1);
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
Epetra_CrsMatrix &Op2NonConstEpetraCrs(Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> &Op) {
  using CrsMatrixWrap   = Xpetra::CrsMatrixWrap<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
  using EpetraCrsMatrix = Xpetra::EpetraCrsMatrixT<GlobalOrdinal, Node>;
  try {
    CrsMatrixWrap &crsOp = dynamic_cast<CrsMatrixWrap &>(Op);
    try {
      EpetraCrsMatrix &tmp_ECrsMtx = dynamic_cast<EpetraCrsMatrix &>(*crsOp.getCrsMatrix());
      return *tmp_ECrsMtx.getEpetra_CrsMatrixNonConst();
    } catch (std::bad_cast &) {
      throw std::runtime_error("Cast from Xpetra::CrsMatrix to Xpetra::EpetraCrsMatrix failed");
    }
  } catch (std::bad_cast &) {
    throw std::runtime_error("Cast from Xpetra::Matrix to Xpetra::CrsMatrixWrap failed");
  }
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>> ML_Multiply(Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> &Op1, Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> &Op2) {
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
RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>> Transpose(Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> &Op, bool /* optimizeTranspose */ = false, const std::string &label = std::string(), const Teuchos::RCP<Teuchos::ParameterList> &params = Teuchos::null) {
  using Matrix          = Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
  using CrsMatrix       = Xpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
  using CrsMatrixWrap   = Xpetra::CrsMatrixWrap<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
  using EpetraCrsMatrix = Xpetra::EpetraCrsMatrixT<GlobalOrdinal, Node>;
  switch (Op.getRowMap()->lib()) {
    case Xpetra::UseTpetra: {
      const Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> &tpetraOp = toTpetra(Op);

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
      Epetra_CrsMatrix &epetraOp = Op2NonConstEpetraCrs(Op);
      EpetraExt::RowMatrix_Transpose transposer;
      Epetra_CrsMatrix *A = dynamic_cast<Epetra_CrsMatrix *>(&transposer(epetraOp));
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

int main(int argc, char **argv) {
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
  // std::string map_file        = "rowmap_A_0.m";
  // std::string coarse_map_file = "domainmap_P_1.m";
  // std::string A_file          = "A_0.m";
  // std::string P_file          = "P_1.m";

  std::string coords_file = "coords.mm";

  // 1 rank
  std::string map_file        = "rowmap_A_3.m";
  std::string coarse_map_file = "domainmap_P_4.m";
  std::string A_file          = "A_3.m";
  std::string P_file          = "P_4.m";

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
  // auto xCoords    = Xpetra::IO<Scalar, LocalOrdinal, GlobalOrdinal, Node>::ReadMultiVector(coords_file, map);

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
    // auto eCoords = Xpetra::MultiVectorFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Build(ep_map, xCoords->getNumVectors());

    // Kokkos::deep_copy(eCoords->getLocalViewDevice(Tpetra::Access::OverwriteAll),
    //                   xCoords->getLocalViewDevice(Tpetra::Access::ReadOnly));

    Xpetra::IO<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Write("epetra_" + A_file, *eA);
    Xpetra::IO<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Write("epetra_" + P_file, *eP);
    // Xpetra::IO<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Write("epetra_" + coords_file, *eCoords);

    map->getComm()->barrier();

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

    map->getComm()->barrier();

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

    const Epetra_MpiComm *Mcomm = dynamic_cast<const Epetra_MpiComm *>(&Ae.Comm());
    if (Mcomm) ML_Comm_Set_UsrComm(comm, Mcomm->GetMpiComm());

    ML_Operator *R_ml, *A_ml, *P_ml, *Ac_ml;
    A_ml = ML_Operator_Create(comm);
    P_ml = ML_Operator_Create(comm);
    ML_Operator_WrapEpetraMatrix(&Ae, A_ml);
    ML_Operator_WrapEpetraMatrix(&Pe, P_ml);

    map->getComm()->barrier();

    for (int iter = 0; iter < numRepeats; ++iter) {
      auto timer = rcp(new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("ML implicit")));
      R_ml       = ML_Operator_Create(comm);
      {
        auto timer2 = rcp(new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("ML implicit Pt")));
        // ML_Operator_ImplicitTranspose(P_ml, R_ml, 0);
        ML_Gen_Restrictor_TransP2(R_ml, P_ml);
      }
      Ac_ml = ML_Operator_Create(comm);
      {
        auto timer2 = rcp(new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("ML RAP")));
        ML_rap(R_ml, A_ml, P_ml, Ac_ml, ML_CSR_MATRIX);
      }
      // std::cout << "HERE2 " << ML_Operator_Get_Nnz(Ac_ml) << std::endl;
      ML_Operator_Destroy(&Ac_ml);
      ML_Operator_Destroy(&R_ml);
    }

    map->getComm()->barrier();

    for (int iter = 0; iter < numRepeats; ++iter) {
      auto timer = rcp(new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("ML explicit")));
      R_ml       = ML_Operator_Create(comm);
      {
        auto timer2 = rcp(new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("ML explicit Pt")));
        ML_Operator_Transpose_byrow(P_ml, R_ml);
      }
      Ac_ml = ML_Operator_Create(comm);
      {
        auto timer2 = rcp(new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("ML RAP")));
        ML_rap(R_ml, A_ml, P_ml, Ac_ml, ML_CSR_MATRIX);
      }
      // std::cout << "HERE3 " << ML_Operator_Get_Nnz(Ac_ml) << std::endl;
      ML_Operator_Destroy(&Ac_ml);
      ML_Operator_Destroy(&R_ml);
    }

    map->getComm()->barrier();

    ML_Comm_Destroy(&comm);
    // global_comm = temp;

    /* Need to blow about BBt_ml but keep epetra stuff */

    // result = (Epetra_RowMatrix *) AP_ml->data;
    ML_Operator_Destroy(&A_ml);
    ML_Operator_Destroy(&P_ml);
  }

  // Tpetra
  {
    map->getComm()->barrier();

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

    map->getComm()->barrier();

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
