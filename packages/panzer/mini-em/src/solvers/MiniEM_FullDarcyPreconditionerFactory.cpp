#include "MiniEM_FullDarcyPreconditionerFactory.hpp"

#include "Teko_BlockLowerTriInverseOp.hpp"
#include "Teko_BlockUpperTriInverseOp.hpp"

#include "Teko_SolveInverseFactory.hpp"

#include "Thyra_DiagonalLinearOpBase.hpp"
#include "Thyra_DefaultProductVectorSpace.hpp"
#include "Thyra_DefaultProductMultiVector.hpp"

#include "Teuchos_as.hpp"
#include "Teuchos_Time.hpp"

#include "Teko_TpetraHelpers.hpp"
#include "Teko_TpetraOperatorWrapper.hpp"

#include "Thyra_TpetraLinearOp.hpp"
#include "Panzer_NodeType.hpp"
#include "PanzerDiscFE_config.hpp"
#ifdef PANZER_HAVE_EPETRA_STACK
#include "Thyra_EpetraThyraWrappers.hpp"
#include "ml_epetra_utils.h"
#include "EpetraExt_MatrixMatrix.h"
#endif
#include "Panzer_LOCPair_GlobalEvaluationData.hpp"
#include "Panzer_LinearObjContainer.hpp"
#include "Panzer_ThyraObjContainer.hpp"

#include "Thyra_DefaultDiagonalLinearOp.hpp"

#include "MiniEM_Utils.hpp"

using Teuchos::RCP;
using Teuchos::rcp_dynamic_cast;

namespace mini_em {


///////////////////////////////////////
// FullDarcyPreconditionerFactory  //
///////////////////////////////////////

Teko::LinearOp FullDarcyPreconditionerFactory::buildPreconditionerOperator(Teko::BlockedLinearOp & blo, Teko::BlockPreconditionerState & /* state */) const
{
   typedef double Scalar;
   typedef int LocalOrdinal;
   typedef panzer::GlobalOrdinal GlobalOrdinal;
   typedef panzer::TpetraNodeType Node;

   Teuchos::TimeMonitor tM(*Teuchos::TimeMonitor::getNewTimer(std::string("DarcyPreconditioner::build")));

   // Output stream for debug information
   RCP<Teuchos::FancyOStream> debug = Teuchos::null;
   if (doDebug)
     debug = Teko::getOutputStream();

   // Check that system is right size
   int rows = Teko::blockRowCount(blo);
   int cols = Teko::blockColCount(blo);
   TEUCHOS_ASSERT(rows==cols);
   TEUCHOS_ASSERT(rows==2);

   // Notation:
   // 0 - Hgrad
   // 1 - Hcurl
   // 2 - HDiv
   // 3 - HVol

   // M_k(a) - mass matrix on space k with weight a
   // D_k - derivative from space k to k+1

   // The block matrix is
   //
   // | Q_u  K   |
   // | Kt   Q_sigma |
   //
   // where
   // Q_u = 1/dt * M_3(1)
   // K   = -M_3(1) * D_2
   // Kt  = D_2^T * M_3(1)
   // Q_sigma = M_2(1/kappa)

   // S_sigma = Q_sigma - Kt * Q_u^-1 * K
   //     = M_2(1/kappa) + dt * D_2^T * M_3(1) * D_2
   //
   // -> grad-div term scales like dt
   //
   // for refmaxwell: Q_rho = M_1(1/dt) so that the addon is:
   // M_2(1) * D_1 * M_1(1/dt)^-1 * D_1^T * M_2(1)

   // Modify the system
   if (simplifyFaraday_) {
     RCP<Teuchos::FancyOStream> out = Teko::getOutputStream();
     *out << std::endl;
     *out << "*** WARNING ***" << std::endl;
     *out << "We are modifying the linear system. That's not a friendly thing to do." << std::endl;
     *out << std::endl;

     Teko::LinearOp Q_u  = Teko::getBlock(0,0,blo);
     Teko::LinearOp id_u = getIdentityMatrix(Q_u, 1/dt);
     Teko::LinearOp hoDiv  = Teko::scale(-1.0, getRequestHandler()->request<Teko::LinearOp>(Teko::RequestMesg("Discrete Div")));
     Teko::LinearOp Kt    = Teko::getBlock(1,0,blo);
     Teko::LinearOp Q_sigma   = Teko::getBlock(1,1,blo);
     blo->beginBlockFill(2,2);
     Teko::setBlock(0,0,blo,id_u);
     Teko::setBlock(0,1,blo,hoDiv);
     Teko::setBlock(1,0,blo,Kt);
     Teko::setBlock(1,1,blo,Q_sigma);
     blo->endBlockFill();
   }

   // Extract the blocks
   Teko::LinearOp Q_u   = Teko::getBlock(0,0,blo);
   Teko::LinearOp K     = Teko::getBlock(0,1,blo);
   Teko::LinearOp Kt    = Teko::getBlock(1,0,blo);
   Teko::LinearOp Q_sigma   = Teko::getBlock(1,1,blo);

   // discrete curl and its transpose
   Teko::LinearOp Div, DivT;
   if (use_discrete_div_) {
     Div = getRequestHandler()->request<Teko::LinearOp>(Teko::RequestMesg("Discrete Div"));
     DivT = Teko::explicitTranspose(Div);
   }

   // Set up the Schur complement
   Teko::LinearOp S_sigma;
   {
     Teuchos::TimeMonitor tm(*Teuchos::TimeMonitor::getNewTimer("DarcyPreconditioner: Schur complement"));
     S_sigma = getRequestHandler()->request<Teko::LinearOp>(Teko::RequestMesg("DarcySchurComplement AUXILIARY_FACE"));
   }

   // Check whether we are using Tpetra or Epetra
   RCP<const Thyra::TpetraLinearOp<Scalar,LocalOrdinal,GlobalOrdinal,Node> > checkTpetra = Teuchos::rcp_dynamic_cast<const Thyra::TpetraLinearOp<Scalar,LocalOrdinal,GlobalOrdinal,Node>>(Q_sigma);
   bool useTpetra = nonnull(checkTpetra);

   /////////////////////////////////////////////////
   // Debug and matrix dumps                      //
   /////////////////////////////////////////////////

   if (dump) {
     writeOut("Q_u.mm",*Q_u);
     writeOut("K.mm",*K);
     writeOut("Kt.mm",*Kt);
     writeOut("Q_sigma.mm",*Q_sigma);
     writeOut("S_sigma.mm",*S_sigma);

     if (Div != Teuchos::null) {
       Teko::LinearOp K2 = Teko::explicitMultiply(Q_u, Div);
       Teko::LinearOp diffK;

       if (useTpetra) {
         typedef panzer::TpetraNodeType Node;
         typedef int LocalOrdinal;
         typedef panzer::GlobalOrdinal GlobalOrdinal;

         RCP<const Thyra::TpetraLinearOp<Scalar,LocalOrdinal,GlobalOrdinal,Node> > tOp = rcp_dynamic_cast<const Thyra::TpetraLinearOp<Scalar,LocalOrdinal,GlobalOrdinal,Node> >(K2,true);
         RCP<Thyra::TpetraLinearOp<Scalar,LocalOrdinal,GlobalOrdinal,Node> > tOp2 = Teuchos::rcp_const_cast<Thyra::TpetraLinearOp<Scalar,LocalOrdinal,GlobalOrdinal,Node>>(tOp);
         RCP<Tpetra::CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> > crsOp = rcp_dynamic_cast<Tpetra::CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >(tOp2->getTpetraOperator(),true);
         crsOp->scale(dt);
         diffK = Teko::explicitAdd(K, Teko::scale(-1.0,K2));

         writeOut("DiscreteDiv.mm",*Div);
         writeOut("K2.mm",*K2);
         writeOut("diff.mm",*diffK);

       } else {
         diffK = Teko::explicitAdd(K, Teko::scale(-dt,K2));

         writeOut("DiscreteDiv.mm",*Div);
         writeOut("K2.mm",*K2);
         writeOut("diff.mm",*diffK);
       }

       TEUCHOS_ASSERT(Teko::infNorm(diffK) < 1.0e-8 * Teko::infNorm(K));
     }
   }
   describeMatrix("Q_u",*Q_u,debug);
   describeMatrix("K",*K,debug);
   describeMatrix("Kt",*Kt,debug);
   describeMatrix("Q_sigma",*Q_sigma,debug);
   if (Div != Teuchos::null)
     describeMatrix("Div",*Div,debug);
   describeMatrix("S_sigma",*S_sigma,debug);


   /////////////////////////////////////////////////
   // Set up inverses for sub-blocks              //
   /////////////////////////////////////////////////

   // Inverse of B mass matrix
   Teko::LinearOp invQ_u;
   if (!simplifyFaraday_) {
     Teuchos::TimeMonitor tm(*Teuchos::TimeMonitor::getNewTimer("DarcyPreconditioner: Inverse Q_u"));
     // Are we building a solver or a preconditioner?
     if (useAsPreconditioner) {
       invQ_u = Teko::buildInverse(*invLib.getInverseFactory("Q_u Preconditioner"),Q_u);
     } else {
       Teko::LinearOp invDiagQ_u = Teko::buildInverse(*invLib.getInverseFactory("Q_u Preconditioner"),Q_u);
       describeMatrix("invDiagQ_u",*invDiagQ_u,debug);
       invQ_u = Teko::buildInverse(*invLib.getInverseFactory("Q_u Solve"),Q_u, invDiagQ_u);
     }
   }

   // Solver for S_sigma
   Teko::LinearOp invS_sigma;
   {
     Teuchos::TimeMonitor tm1(*Teuchos::TimeMonitor::getNewTimer("DarcyPreconditioner: Solver S_sigma"));

     if ((S_sigma_prec_type_ == "MueLuRefDarcy") || (S_sigma_prec_type_ == "MueLuRefMaxwell")) {// refDarcy

       // Teko::LinearOp T = getRequestHandler()->request<Teko::LinearOp>(Teko::RequestMesg("Discrete Gradient"));
       // Teko::LinearOp KT = Teko::explicitMultiply(K,T);
       // TEUCHOS_ASSERT(Teko::infNorm(KT) < 1.0e-14 * Teko::infNorm(T) * Teko::infNorm(K));

       RCP<Teko::InverseFactory> S_sigma_prec_factory;
       Teuchos::ParameterList S_sigma_prec_pl;
       S_sigma_prec_factory = invLib.getInverseFactory("S_sigma Preconditioner");
       S_sigma_prec_pl = *S_sigma_prec_factory->getParameterList();

       // Get coordinates
       {
         if (useTpetra) {
           RCP<Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> > Coordinates = S_sigma_prec_pl.get<RCP<Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> > >("Coordinates");
           S_sigma_prec_pl.sublist("Preconditioner Types").sublist(S_sigma_prec_type_).set("Coordinates",Coordinates);
         }
         else
           TEUCHOS_ASSERT(false);
       }

       // edge mass matrix
       Teko::LinearOp Mk_1_invBeta = getRequestHandler()->request<Teko::LinearOp>(Teko::RequestMesg("Mass Matrix kappa weighted AUXILIARY_EDGE"));
       describeMatrix("Mk_1_invBeta",*Mk_1_invBeta,debug);
       if (dump)
         writeOut("Mk_1_invBeta.mm",*Mk_1_invBeta);

       RCP<const Thyra::DiagonalLinearOpBase<Scalar> > invMk_1_invBeta;
       {
         Teuchos::TimeMonitor tm(*Teuchos::TimeMonitor::getNewTimer("MaxwellPreconditioner: Lumped diagonal Mk_1_invBeta"));

         // Get inverse of lumped Mk_1_invBeta
         RCP<Thyra::VectorBase<Scalar> > ones = Thyra::createMember(Mk_1_invBeta->domain());
         RCP<Thyra::VectorBase<Scalar> > diagonal = Thyra::createMember(Mk_1_invBeta->range());
         Thyra::assign(ones.ptr(),1.0);
         // compute lumped diagonal
         Thyra::apply(*Mk_1_invBeta,Thyra::NOTRANS,*ones,diagonal.ptr());
         Thyra::reciprocal(*diagonal,diagonal.ptr());
         invMk_1_invBeta = rcp(new Thyra::DefaultDiagonalLinearOp<Scalar>(diagonal));
       }

       // nodal mass matrix
       Teko::LinearOp Mk_2_invAlpha = getRequestHandler()->request<Teko::LinearOp>(Teko::RequestMesg("Mass Matrix dt weighted AUXILIARY_NODE"));
       describeMatrix("Mk_2_invAlpha",*Mk_2_invAlpha,debug);
       if (dump)
         writeOut("Mk_2_invAlpha.mm",*Mk_2_invAlpha);

       RCP<const Thyra::DiagonalLinearOpBase<Scalar> > invMk_2_invAlpha;
       {
         Teuchos::TimeMonitor tm(*Teuchos::TimeMonitor::getNewTimer("MaxwellPreconditioner: Lumped diagonal Mk_2_invAlpha"));

         // Get inverse of lumped Mk_2_invAlpha
         RCP<Thyra::VectorBase<Scalar> > ones = Thyra::createMember(Mk_2_invAlpha->domain());
         RCP<Thyra::VectorBase<Scalar> > diagonal = Thyra::createMember(Mk_2_invAlpha->range());
         Thyra::assign(ones.ptr(),1.0);
         // compute lumped diagonal
         Thyra::apply(*Mk_2_invAlpha,Thyra::NOTRANS,*ones,diagonal.ptr());
         Thyra::reciprocal(*diagonal,diagonal.ptr());
         invMk_2_invAlpha = rcp(new Thyra::DefaultDiagonalLinearOp<Scalar>(diagonal));
       }

       {
         Teko::InverseLibrary myInvLib = invLib;
         S_sigma_prec_pl.sublist("Preconditioner Types").sublist(S_sigma_prec_type_).set("invMk_1_invBeta",invMk_1_invBeta);
         S_sigma_prec_pl.sublist("Preconditioner Types").sublist(S_sigma_prec_type_).set("invMk_2_invAlpha",invMk_2_invAlpha);
         S_sigma_prec_pl.sublist("Preconditioner Types").sublist(S_sigma_prec_type_).set("Type",S_sigma_prec_type_);
         myInvLib.addInverse("S_sigma Preconditioner",S_sigma_prec_pl.sublist("Preconditioner Types").sublist(S_sigma_prec_type_));
         S_sigma_prec_factory = myInvLib.getInverseFactory("S_sigma Preconditioner");
       }

       // Are we building a solver or a preconditioner?
       {
         Teuchos::TimeMonitor tm(*Teuchos::TimeMonitor::getNewTimer("MaxwellPreconditioner: Build S_sigma preconditioner"));

         if (useAsPreconditioner)
           invS_sigma = Teko::buildInverse(*S_sigma_prec_factory,S_sigma);
         else {
           if (S_sigma_prec_.is_null())
             S_sigma_prec_ = Teko::buildInverse(*S_sigma_prec_factory,S_sigma);
           else
             Teko::rebuildInverse(*S_sigma_prec_factory,S_sigma, S_sigma_prec_);
           invS_sigma = Teko::buildInverse(*invLib.getInverseFactory("S_sigma Solve"),S_sigma,S_sigma_prec_);
         }
       }
     }
#ifdef PANZER_HAVE_EPETRA_STACK
     else if (S_sigma_prec_type_ == "ML") {
       RCP<Teko::InverseFactory> S_sigma_prec_factory;
       Teuchos::ParameterList S_sigma_prec_pl;
       S_sigma_prec_factory = invLib.getInverseFactory("S_sigma Preconditioner");
       S_sigma_prec_pl = *S_sigma_prec_factory->getParameterList();

       double* x_coordinates = S_sigma_prec_pl.sublist("ML Settings").get<double*>("x-coordinates");
       S_sigma_prec_pl.sublist("ML Settings").sublist("graddiv: 11list").set("x-coordinates",x_coordinates);
       S_sigma_prec_pl.sublist("ML Settings").sublist("graddiv: 22list").set("x-coordinates",x_coordinates);
       double* y_coordinates = S_sigma_prec_pl.sublist("ML Settings").get<double*>("y-coordinates");
       S_sigma_prec_pl.sublist("ML Settings").sublist("graddiv: 11list").set("y-coordinates",y_coordinates);
       S_sigma_prec_pl.sublist("ML Settings").sublist("graddiv: 22list").set("y-coordinates",y_coordinates);
       double* z_coordinates = S_sigma_prec_pl.sublist("ML Settings").get<double*>("z-coordinates");
       S_sigma_prec_pl.sublist("ML Settings").sublist("graddiv: 11list").set("z-coordinates",z_coordinates);
       S_sigma_prec_pl.sublist("ML Settings").sublist("graddiv: 22list").set("z-coordinates",z_coordinates);

       // add discrete curl and face mass matrix
       Teko::LinearOp Curl = getRequestHandler()->request<Teko::LinearOp>(Teko::RequestMesg("Discrete Curl"));
       Teko::LinearOp Grad = getRequestHandler()->request<Teko::LinearOp>(Teko::RequestMesg("Discrete Gradient"));

       RCP<const Epetra_CrsMatrix> D1 = get_Epetra_CrsMatrix(*Curl);
       RCP<const Epetra_CrsMatrix> D0 = get_Epetra_CrsMatrix(*Grad);

       S_sigma_prec_pl.sublist("ML Settings").set("D1",D1);
       S_sigma_prec_pl.sublist("ML Settings").set("D0",D0);

       if (dump) {
         writeOut("DiscreteGradient.mm",*Grad);
         writeOut("DiscreteCurl.mm",*Curl);
       }
       describeMatrix("DiscreteGradient",*Grad,debug);
       describeMatrix("DiscreteCurl",*Curl,debug);


       // We might have zero entries in the matrices, so instead of
       // setting all entries to one, we only modify the nonzero ones.
       Epetra_CrsMatrix D0_one(*D0);
       {
         int *rowptr;
         int *indices;
         double *values;
         D0_one.ExtractCrsDataPointers(rowptr, indices, values);
         for (int jj = 0; jj<D0_one.NumMyNonzeros(); jj++ )
           if (std::abs(values[jj])>1e-10)
             values[jj] = 1.0;
           else
             values[jj] = 0.0;
       }

       Epetra_CrsMatrix D1_one(*D1);
       {
         int *rowptr;
         int *indices;
         double *values;
         D1_one.ExtractCrsDataPointers(rowptr, indices, values);
         for (int jj = 0; jj<D1_one.NumMyNonzeros(); jj++ )
           if (std::abs(values[jj])>1e-10)
             values[jj] = 1.0;
           else
             values[jj] = 0.0;
       }

       RCP<Epetra_CrsMatrix> FaceNode = Teuchos::rcp(new Epetra_CrsMatrix(Copy,D1->RowMap(),0));
       EpetraExt::MatrixMatrix::Multiply(D1_one,false,D0_one,false, *FaceNode);
       {
         int *rowptr;
         int *indices;
         double *values;
         FaceNode->ExtractCrsDataPointers(rowptr, indices, values);
         for (int jj = 0; jj<FaceNode->NumMyNonzeros(); jj++ )
           if (std::abs(values[jj])>1e-10)
             values[jj] = 1.0;
           else
             values[jj] = 0.0;
       }
       std::cout << "FaceNode.nnz " << FaceNode->NumGlobalNonzeros() << std::endl;
       RCP<const Epetra_CrsMatrix> FaceNodeConst = FaceNode;
       S_sigma_prec_pl.sublist("ML Settings").set("FaceNode",FaceNodeConst);

       if (dump)
         EpetraExt::RowMatrixToMatrixMarketFile("FaceNode.dat", *FaceNodeConst);

       Teko::LinearOp Mk_1_invBeta = getRequestHandler()->request<Teko::LinearOp>(Teko::RequestMesg("Mass Matrix kappa weighted AUXILIARY_EDGE"));
       RCP<const Epetra_CrsMatrix> M1 = get_Epetra_CrsMatrix(*Mk_1_invBeta);
       Epetra_CrsMatrix * TMT_Agg_Matrix;
       ML_Epetra::ML_Epetra_PtAP(*M1, *D0, TMT_Agg_Matrix,false);
       RCP<const Epetra_CrsMatrix> TMT_Agg_MatrixConst = Teuchos::rcp(TMT_Agg_Matrix);
       S_sigma_prec_pl.sublist("ML Settings").set("K0",TMT_Agg_MatrixConst);

       if (dump)
         EpetraExt::RowMatrixToMatrixMarketFile("TMT.dat", *TMT_Agg_MatrixConst);

       {
         Teko::InverseLibrary myInvLib = invLib;
         S_sigma_prec_pl.set("Type",S_sigma_prec_type_);
         myInvLib.addInverse("S_sigma Preconditioner",S_sigma_prec_pl);
         S_sigma_prec_factory = myInvLib.getInverseFactory("S_sigma Preconditioner");
       }

       // Are we building a solver or a preconditioner?
       {
         Teuchos::TimeMonitor tm(*Teuchos::TimeMonitor::getNewTimer("MaxwellPreconditioner: Build S_sigma preconditioner"));

         if (useAsPreconditioner)
           invS_sigma = Teko::buildInverse(*S_sigma_prec_factory,S_sigma);
         else {
           if (S_sigma_prec_.is_null())
             S_sigma_prec_ = Teko::buildInverse(*S_sigma_prec_factory,S_sigma);
           else
             Teko::rebuildInverse(*S_sigma_prec_factory,S_sigma, S_sigma_prec_);
           invS_sigma = Teko::buildInverse(*invLib.getInverseFactory("S_sigma Solve"),S_sigma,S_sigma_prec_);
         }
       }
     }
#endif
     else {
       if (useAsPreconditioner)
         invS_sigma = Teko::buildInverse(*invLib.getInverseFactory("S_sigma Preconditioner"),S_sigma);
       else {
         if (S_sigma_prec_.is_null())
           S_sigma_prec_ = Teko::buildInverse(*invLib.getInverseFactory("S_sigma Preconditioner"),S_sigma);
         else
           Teko::rebuildInverse(*invLib.getInverseFactory("S_sigma Preconditioner"),S_sigma, S_sigma_prec_);
         invS_sigma = Teko::buildInverse(*invLib.getInverseFactory("S_sigma Solve"),S_sigma,S_sigma_prec_);
       }
     }
   }


   /////////////////////////////////////////////////
   // Build block  inverse matrices               //
   /////////////////////////////////////////////////

   if (!use_discrete_div_) {
     Teuchos::TimeMonitor tm(*Teuchos::TimeMonitor::getNewTimer("MaxwellPreconditioner: Block preconditioner"));

     // Inverse blocks
     std::vector<Teko::LinearOp> diag(2);
     diag[0] = invQ_u;
     diag[1] = invS_sigma;

     // Upper tri blocks
     Teko::BlockedLinearOp U = Teko::createBlockedOp();
     Teko::beginBlockFill(U,rows,rows);
     Teko::setBlock(0,0,U,Q_u);
     Teko::setBlock(1,1,U,S_sigma);
     Teko::setBlock(0,1,U,K);
     Teko::endBlockFill(U);

     Teko::LinearOp invU = Teko::createBlockUpperTriInverseOp(U,diag);

     if (!useAsPreconditioner) {
       Teko::BlockedLinearOp invL = Teko::createBlockedOp();
       Teko::LinearOp id_u = Teko::identity(Teko::rangeSpace(Q_u));
       Teko::LinearOp id_sigma = Teko::identity(Teko::rangeSpace(Q_sigma));
       Teko::beginBlockFill(invL,rows,rows);
       Teko::setBlock(0,0,invL,id_u);
       Teko::setBlock(1,0,invL,Teko::multiply(Thyra::scale(-1.0, Kt), invQ_u));
       Teko::setBlock(1,1,invL,id_sigma);
       Teko::endBlockFill(invL);

       return Teko::multiply(invU, Teko::toLinearOp(invL));
     } else
       return invU;
   } else {
     Teuchos::TimeMonitor tm(*Teuchos::TimeMonitor::getNewTimer("MaxwellPreconditioner: Block preconditioner"));

     Teko::LinearOp id_u = Teko::identity(Teko::rangeSpace(Q_u));

     // Inverse blocks
     std::vector<Teko::LinearOp> diag(2);
     diag[0] = Teko::scale(dt,id_u);
     diag[1] = invS_sigma;

     // Upper tri blocks
     Teko::BlockedLinearOp U = Teko::createBlockedOp();
     Teko::beginBlockFill(U,rows,rows);
     Teko::setBlock(0,0,U,Teko::scale(1/dt,id_u));
     Teko::setBlock(1,1,U,S_sigma);
     Teko::setBlock(0,1,U,Div);
     Teko::endBlockFill(U);

     Teko::LinearOp invU = Teko::createBlockUpperTriInverseOp(U,diag);

     if (!useAsPreconditioner) {
       Teko::BlockedLinearOp invL = Teko::createBlockedOp();
       Teko::LinearOp id_sigma = Teko::identity(Teko::rangeSpace(Q_sigma));
       Teko::beginBlockFill(invL,rows,rows);
       Teko::setBlock(0,0,invL,id_u);
       Teko::setBlock(1,0,invL,Thyra::scale(-dt, Kt));
       Teko::setBlock(1,1,invL,id_sigma);
       Teko::endBlockFill(invL);

       if (!simplifyFaraday_) {
         Teko::BlockedLinearOp invDiag = Teko::createBlockedOp();
         Teko::beginBlockFill(invDiag,rows,rows);
         Teko::setBlock(0,0,invDiag,Teko::scale(1/dt,invQ_u));
         Teko::setBlock(1,1,invDiag,id_sigma);
         Teko::endBlockFill(invDiag);

         return Teko::multiply(invU, Teko::multiply(Teko::toLinearOp(invL), Teko::toLinearOp(invDiag)));
       } else
         return Teko::multiply(invU, Teko::toLinearOp(invL));
     } else
       return invU;
   }
}

//! Initialize from a parameter list
void FullDarcyPreconditionerFactory::initializeFromParameterList(const Teuchos::ParameterList & pl)
{
   /////////////////////
   // Solver options  //
   // //////////////////

   params = pl;

   use_discrete_div_     = params.get("Use discrete div",false);
   dump                   = params.get("Dump",false);
   doDebug                = params.get("Debug",false);
   useAsPreconditioner    = params.get("Use as preconditioner",false);
   simplifyFaraday_       = params.get("Simplify Faraday",false) && use_discrete_div_;

   if(pl.isSublist("S_sigma Preconditioner") && pl.sublist("S_sigma Preconditioner").isParameter("Type"))
     S_sigma_prec_type_ = pl.sublist("S_sigma Preconditioner").get<std::string>("Type");
   else
     S_sigma_prec_type_ = "";

   // Output stream for debug information
   RCP<Teuchos::FancyOStream> debug = Teuchos::null;
   if (doDebug)
     debug = Teko::getOutputStream();

   //////////////////////////////////
   // Set up sub-solve factories   //
   //////////////////////////////////

   // New inverse lib to add inverse factories to
   invLib = *getInverseLibrary();

   // Q_u solve
   if (pl.isParameter("Q_u Solve")) {
     Teuchos::ParameterList cg_pl = pl.sublist("Q_u Solve");
     invLib.addInverse("Q_u Solve",cg_pl);
   }

   // Q_u preconditioner
   Teuchos::ParameterList Q_u_prec_pl = pl.sublist("Q_u Preconditioner");
   invLib.addStratPrecond("Q_u Preconditioner",
                          Q_u_prec_pl.get<std::string>("Prec Type"),
                          Q_u_prec_pl.sublist("Prec Types"));

   dt = params.get<double>("dt");

   if ((S_sigma_prec_type_ == "MueLuRefDarcy") || (S_sigma_prec_type_ == "MueLuRefMaxwell")) { // RefDarcy based solve

     // S_sigma solve
     Teuchos::ParameterList ml_pl = pl.sublist("S_sigma Solve");
     invLib.addInverse("S_sigma Solve",ml_pl);

     // S_sigma preconditioner
     Teuchos::ParameterList S_sigma_prec_pl = pl.sublist("S_sigma Preconditioner");

     // add discrete curl and face mass matrix
     Teko::LinearOp M1_beta = getRequestHandler()->request<Teko::LinearOp>(Teko::RequestMesg("Mass Matrix 1/kappa weighted AUXILIARY_EDGE"));
     Teko::LinearOp M1_alpha = getRequestHandler()->request<Teko::LinearOp>(Teko::RequestMesg("Mass Matrix 1/dt weighted AUXILIARY_EDGE"));
     Teko::LinearOp Mk_one = getRequestHandler()->request<Teko::LinearOp>(Teko::RequestMesg("Mass Matrix AUXILIARY_FACE"));
     Teko::LinearOp Mk_1_one = getRequestHandler()->request<Teko::LinearOp>(Teko::RequestMesg("Mass Matrix AUXILIARY_EDGE"));
     Teko::LinearOp Curl = getRequestHandler()->request<Teko::LinearOp>(Teko::RequestMesg("Discrete Curl"));
     Teko::LinearOp Grad = getRequestHandler()->request<Teko::LinearOp>(Teko::RequestMesg("Discrete Gradient"));

     S_sigma_prec_pl.sublist("Preconditioner Types").sublist(S_sigma_prec_type_).set("Dk_1",Curl);
     S_sigma_prec_pl.sublist("Preconditioner Types").sublist(S_sigma_prec_type_).set("Dk_2",Grad);
     S_sigma_prec_pl.sublist("Preconditioner Types").sublist(S_sigma_prec_type_).set("D0",Grad);

     S_sigma_prec_pl.sublist("Preconditioner Types").sublist(S_sigma_prec_type_).set("M1_beta",M1_beta);
     S_sigma_prec_pl.sublist("Preconditioner Types").sublist(S_sigma_prec_type_).set("M1_alpha",M1_alpha);

     S_sigma_prec_pl.sublist("Preconditioner Types").sublist(S_sigma_prec_type_).set("Mk_one",Mk_one);
     S_sigma_prec_pl.sublist("Preconditioner Types").sublist(S_sigma_prec_type_).set("Mk_1_one",Mk_1_one);

     if (dump) {
       writeOut("DiscreteGradient.mm",*Grad);
       writeOut("DiscreteCurl.mm",*Curl);
     }
     describeMatrix("DiscreteGradient",*Grad,debug);
     describeMatrix("DiscreteCurl",*Curl,debug);

     invLib.addInverse("S_sigma Preconditioner",S_sigma_prec_pl);
   }
#ifdef PANZER_HAVE_EPETRA_STACK
   else if (S_sigma_prec_type_ == "ML") {
     // S_sigma solve
     Teuchos::ParameterList ml_pl = pl.sublist("S_sigma Solve");
     invLib.addInverse("S_sigma Solve",ml_pl);

     // S_sigma preconditioner
     Teuchos::ParameterList S_sigma_prec_pl = pl.sublist("S_sigma Preconditioner");

     invLib.addInverse("S_sigma Preconditioner",S_sigma_prec_pl);
   }
#endif
   else {
     // S_sigma solve
     if (pl.isParameter("S_sigma Solve")) {
       Teuchos::ParameterList cg_pl = pl.sublist("S_sigma Solve");
       invLib.addInverse("S_sigma Solve",cg_pl);
     }

     // S_sigma preconditioner
     Teuchos::ParameterList S_sigma_prec_pl = pl.sublist("S_sigma Preconditioner");
     invLib.addStratPrecond("S_sigma Preconditioner",
                            S_sigma_prec_pl.get<std::string>("Prec Type"),
                            S_sigma_prec_pl.sublist("Prec Types"));
   }

}

} // namespace mini_em
