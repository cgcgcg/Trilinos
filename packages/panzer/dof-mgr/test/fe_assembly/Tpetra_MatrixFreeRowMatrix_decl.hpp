#ifndef TPETRA_MATRIXFREEROWMATRIX_DECL_HPP
#define TPETRA_MATRIXFREEROWMATRIX_DECL_HPP

#include "Intrepid2_Orientation.hpp"
#include "Intrepid2_OrientationTools.hpp"
#include "Intrepid2_ProjectionTools.hpp"
#include "Intrepid2_HGRAD_HEX_C1_FEM.hpp"
#include "Intrepid2_HGRAD_HEX_Cn_FEM.hpp"
#include "Intrepid2_HCURL_HEX_In_FEM.hpp"
#include "Intrepid2_HVOL_HEX_Cn_FEM.hpp"
#include "Intrepid2_PointTools.hpp"
#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"
#include "Intrepid2_LagrangianInterpolation.hpp"


#include <Teuchos_RCP.hpp>
#include "Intrepid2_Orientation.hpp"
#include <Tpetra_Map.hpp>
#include <Tpetra_Import.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_RowMatrix.hpp>
#include <stdexcept>

namespace Tpetra {


#define ConstructWithLabel(obj, ...) obj(#obj, __VA_ARGS__)

template <class Scalar        = Tpetra::Operator<>::scalar_type,
          class LocalOrdinal  = typename Tpetra::Operator<Scalar>::local_ordinal_type,
          class GlobalOrdinal = typename Tpetra::Operator<Scalar, LocalOrdinal>::global_ordinal_type,
          class Node          = typename Tpetra::Operator<Scalar, LocalOrdinal, GlobalOrdinal>::node_type>
class MatrixFreeRowMatrix : public Tpetra::RowMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> {
 public:
  using matrix_type = Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
  using mv_type     = Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
  using vector_type = Tpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
  using map_type    = Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node>;
  using import_type = Tpetra::Import<LocalOrdinal, GlobalOrdinal, Node>;

  //! The RowMatrix representing the base class of CrsMatrix
  using row_matrix_type = RowMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>;

  using impl_scalar_type = typename row_matrix_type::impl_scalar_type;
  using mag_type         = typename Kokkos::ArithTraits<impl_scalar_type>::mag_type;

  using local_inds_device_view_type =
      typename row_matrix_type::local_inds_device_view_type;
  using local_inds_host_view_type =
      typename row_matrix_type::local_inds_host_view_type;
  using nonconst_local_inds_host_view_type =
      typename row_matrix_type::nonconst_local_inds_host_view_type;

  using global_inds_device_view_type =
      typename row_matrix_type::global_inds_device_view_type;
  using global_inds_host_view_type =
      typename row_matrix_type::global_inds_host_view_type;
  using nonconst_global_inds_host_view_type =
      typename row_matrix_type::nonconst_global_inds_host_view_type;

  using values_device_view_type =
      typename row_matrix_type::values_device_view_type;
  using values_host_view_type =
      typename row_matrix_type::values_host_view_type;
  using nonconst_values_host_view_type =
      typename row_matrix_type::nonconst_values_host_view_type;

  using DeviceSpaceType = typename Node::execution_space;
  using scalar_t = Scalar;
  using local_ordinal_t = LocalOrdinal;
  using DynRankView = Kokkos::DynRankView<impl_scalar_type,DeviceSpaceType>;
  using element_orientation_type = Kokkos::DynRankView<Intrepid2::Orientation,DeviceSpaceType>;
  using basis_type = Intrepid2::Basis<DeviceSpaceType, scalar_t,scalar_t>;

  using ct = Intrepid2::CellTools<DeviceSpaceType>;
  using ots = Intrepid2::OrientationTools<DeviceSpaceType>;
  using rst = Intrepid2::RealSpaceTools<DeviceSpaceType>;
  using fst = Intrepid2::FunctionSpaceTools<DeviceSpaceType>;
  using li = Intrepid2::LagrangianInterpolation<DeviceSpaceType>;

  //! @name Constructor/Destructor
  //@{

  //! Constructor
  MatrixFreeRowMatrix(Teuchos::RCP<const map_type> &ownedMap,
                      Teuchos::RCP<const map_type> &ownedAndGhostedMap,
                      shards::CellTopology topology,
                      Teuchos::RCP<basis_type> &basis,
                      int cubDegree,
                      DynRankView physVertices,
                      element_orientation_type elemOrts,
                      Teuchos::RCP<panzer::DOFManager> &dofManager,
                      Teuchos::RCP<const panzer::GlobalIndexer> &globalIndexer)
    : ownedMap_(ownedMap)
    , ownedAndGhostedMap_(ownedAndGhostedMap)
    , topology_(topology)
    , basis_(basis)
    , cubDegree_(cubDegree)
    , physVertices_(physVertices)
    , elemOrts_(elemOrts)
    , dofManager_(dofManager)
    , globalIndexer_(globalIndexer)
  {
    importer_ = rcp(new import_type(ownedMap_, ownedAndGhostedMap_));
    X_ownedAndGhosted_ = rcp(new mv_type(ownedAndGhostedMap_, 1));
    Y_ownedAndGhosted_ = rcp(new mv_type(ownedAndGhostedMap_, 1));
  };

  //! Returns the Tpetra::Map object associated with the domain of this operator.
  Teuchos::RCP<const Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node> > getDomainMap() const {
    return ownedMap_;
  }

  //! Returns the Tpetra::Map object associated with the range of this operator.
  Teuchos::RCP<const Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node> > getRangeMap() const {
    return ownedMap_;
  }

  //! Returns in Y the result of a Tpetra::Operator applied to a Tpetra::MultiVector X.
  /*!
    \param[in]  X - Tpetra::MultiVector of dimension NumVectors to multiply with matrix.
    \param[out] Y -Tpetra::MultiVector of dimension NumVectors containing result.
  */
  void apply(const Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>& X,
             Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>& Y,
             Teuchos::ETransp mode = Teuchos::NO_TRANS,
             Scalar alpha          = Teuchos::ScalarTraits<Scalar>::one(),
             Scalar beta           = Teuchos::ScalarTraits<Scalar>::zero()) const {

    auto basisCardinality = basis_->getCardinality();
    LocalOrdinal numOwnedElems = elemOrts_.extent(0);
    DynRankView elemsMat("elemsMat", numOwnedElems, basisCardinality, basisCardinality);
    DynRankView elemsRHS("elemsRHS", numOwnedElems, basisCardinality);
    constexpr LocalOrdinal dim = 3;
    const std::string blockId = "eblock-0_0_0";

    {
      // ************************************ ASSEMBLY OF LOCAL ELEMENT MATRICES **************************************
      // Compute quadrature (cubature) points
      Intrepid2::DefaultCubatureFactory cubFactory;
      auto cellCub = cubFactory.create<DeviceSpaceType, scalar_t, scalar_t>(topology_.getBaseKey(), cubDegree_);
      auto numQPoints = cellCub->getNumPoints();
      DynRankView ConstructWithLabel(quadPoints, numQPoints, dim);
      DynRankView ConstructWithLabel(weights, numQPoints);
      cellCub->getCubature(quadPoints, weights);

      // compute oriented basis functions at quadrature points
      auto basisCardinality = basis_->getCardinality();
      DynRankView ConstructWithLabel(basisValuesAtQPointsOriented, numOwnedElems, basisCardinality, numQPoints);
      DynRankView ConstructWithLabel(transformedBasisValuesAtQPointsOriented, numOwnedElems, basisCardinality, numQPoints);
      DynRankView basisValuesAtQPointsCells("inValues", numOwnedElems, basisCardinality, numQPoints);
      DynRankView ConstructWithLabel(basisValuesAtQPoints, basisCardinality, numQPoints);
      basis_->getValues(basisValuesAtQPoints, quadPoints);
      rst::clone(basisValuesAtQPointsCells,basisValuesAtQPoints);

      // modify basis values to account for orientations
      ots::modifyBasisByOrientation(basisValuesAtQPointsOriented,
                                    basisValuesAtQPointsCells,
                                    elemOrts_,
                                    basis_.getRawPtr());

      // transform basis values
      fst::HGRADtransformVALUE(transformedBasisValuesAtQPointsOriented, basisValuesAtQPointsOriented);

      DynRankView ConstructWithLabel(basisGradsAtQPointsOriented, numOwnedElems, basisCardinality, numQPoints, dim);
      DynRankView ConstructWithLabel(transformedBasisGradsAtQPointsOriented, numOwnedElems, basisCardinality, numQPoints, dim);
      DynRankView basisGradsAtQPointsCells("inValues", numOwnedElems, basisCardinality, numQPoints, dim);
      DynRankView ConstructWithLabel(basisGradsAtQPoints, basisCardinality, numQPoints, dim);
      basis_->getValues(basisGradsAtQPoints, quadPoints, Intrepid2::OPERATOR_GRAD);
      rst::clone(basisGradsAtQPointsCells,basisGradsAtQPoints);

      // modify basis values to account for orientations
      ots::modifyBasisByOrientation(basisGradsAtQPointsOriented,
                                    basisGradsAtQPointsCells,
                                    elemOrts_,
                                    basis_.getRawPtr());

      // map basis functions to reference (oriented) element
      DynRankView ConstructWithLabel(jacobianAtQPoints, numOwnedElems, numQPoints, dim, dim);
      DynRankView ConstructWithLabel(jacobianAtQPoints_inv, numOwnedElems, numQPoints, dim, dim);
      DynRankView ConstructWithLabel(jacobianAtQPoints_det, numOwnedElems, numQPoints);
      ct::setJacobian(jacobianAtQPoints, quadPoints, physVertices_, topology_);
      ct::setJacobianInv (jacobianAtQPoints_inv, jacobianAtQPoints);

      fst::HGRADtransformGRAD(transformedBasisGradsAtQPointsOriented, jacobianAtQPoints_inv, basisGradsAtQPointsOriented);

      // compute integrals to assembly local matrices

      DynRankView ConstructWithLabel(weightedTransformedBasisValuesAtQPointsOriented, numOwnedElems, basisCardinality, numQPoints);
      DynRankView ConstructWithLabel(weightedTransformedBasisGradsAtQPointsOriented, numOwnedElems, basisCardinality, numQPoints, dim);
      DynRankView ConstructWithLabel(cellWeights, numOwnedElems, numQPoints);
      rst::clone(cellWeights, weights);

      fst::multiplyMeasure(weightedTransformedBasisGradsAtQPointsOriented, cellWeights, transformedBasisGradsAtQPointsOriented);
      fst::multiplyMeasure(weightedTransformedBasisValuesAtQPointsOriented, cellWeights, transformedBasisValuesAtQPointsOriented);

      fst::integrate(elemsMat, transformedBasisGradsAtQPointsOriented, weightedTransformedBasisGradsAtQPointsOriented);
      fst::integrate(elemsMat, transformedBasisValuesAtQPointsOriented, weightedTransformedBasisValuesAtQPointsOriented, true);
    }

    X_ownedAndGhosted_->doImport(X, *importer_, Tpetra::INSERT);
    Y_ownedAndGhosted_->putScalar(0.);
    {
      auto elementLIDs = globalIndexer_->getLIDs();
      auto elmtOffsetKokkos = dofManager_->getGIDFieldOffsetsKokkos(blockId,0);

      auto lclX = X_ownedAndGhosted_->getLocalViewDevice(Tpetra::Access::ReadOnly);
      auto lclY = Y_ownedAndGhosted_->getLocalViewDevice(Tpetra::Access::OverwriteAll);

      Kokkos::parallel_for
        ("Matrix-free apply",
         Kokkos::RangePolicy<DeviceSpaceType, int> (0, numOwnedElems),
         KOKKOS_LAMBDA (const size_t elemId) {
          // Get subviews
          auto elemMat = Kokkos::subview(elemsMat,elemId, Kokkos::ALL(), Kokkos::ALL());
          auto elemLIds  = Kokkos::subview(elementLIDs,elemId, Kokkos::ALL());

          // For each DoF (row) on the current element
          for (local_ordinal_t rowId = 0; rowId < basisCardinality; ++rowId) {
            const local_ordinal_t localRowId = elemLIds(elmtOffsetKokkos(rowId));

            // For each DoF (column) on the current element
            for (local_ordinal_t colId = 0; colId < basisCardinality; ++colId) {
              const local_ordinal_t localColId = elemLIds(elmtOffsetKokkos(colId));

              // For each column of the multivector
              for (size_t noVector = 0; noVector < lclX.extent(1); ++noVector)
                Kokkos::atomic_add (&(lclY(localRowId, noVector)), alpha * elemMat(rowId, colId) * lclX(localColId, noVector));
            }
          }
        });
      Kokkos::fence();
    }

    Y.scale(beta);
    Y.doExport(*Y_ownedAndGhosted_, *importer_, Tpetra::ADD_ASSIGN);
  }

  void computeDiagonal(Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>& diagonal) const {

    auto basisCardinality = basis_->getCardinality();
    LocalOrdinal numOwnedElems = elemOrts_.extent(0);
    DynRankView elemsMat("elemsMat", numOwnedElems, basisCardinality, basisCardinality);
    DynRankView elemsRHS("elemsRHS", numOwnedElems, basisCardinality);
    constexpr LocalOrdinal dim = 3;
    const std::string blockId = "eblock-0_0_0";

    {
      // ************************************ ASSEMBLY OF LOCAL ELEMENT MATRICES **************************************
      // Compute quadrature (cubature) points
      Intrepid2::DefaultCubatureFactory cubFactory;
      auto cellCub = cubFactory.create<DeviceSpaceType, scalar_t, scalar_t>(topology_.getBaseKey(), cubDegree_);
      auto numQPoints = cellCub->getNumPoints();
      DynRankView ConstructWithLabel(quadPoints, numQPoints, dim);
      DynRankView ConstructWithLabel(weights, numQPoints);
      cellCub->getCubature(quadPoints, weights);

      // compute oriented basis functions at quadrature points
      auto basisCardinality = basis_->getCardinality();
      DynRankView ConstructWithLabel(basisValuesAtQPointsOriented, numOwnedElems, basisCardinality, numQPoints);
      DynRankView ConstructWithLabel(transformedBasisValuesAtQPointsOriented, numOwnedElems, basisCardinality, numQPoints);
      DynRankView basisValuesAtQPointsCells("inValues", numOwnedElems, basisCardinality, numQPoints);
      DynRankView ConstructWithLabel(basisValuesAtQPoints, basisCardinality, numQPoints);
      basis_->getValues(basisValuesAtQPoints, quadPoints);
      rst::clone(basisValuesAtQPointsCells,basisValuesAtQPoints);

      // modify basis values to account for orientations
      ots::modifyBasisByOrientation(basisValuesAtQPointsOriented,
                                    basisValuesAtQPointsCells,
                                    elemOrts_,
                                    basis_.getRawPtr());

      // transform basis values
      fst::HGRADtransformVALUE(transformedBasisValuesAtQPointsOriented, basisValuesAtQPointsOriented);

      DynRankView ConstructWithLabel(basisGradsAtQPointsOriented, numOwnedElems, basisCardinality, numQPoints, dim);
      DynRankView ConstructWithLabel(transformedBasisGradsAtQPointsOriented, numOwnedElems, basisCardinality, numQPoints, dim);
      DynRankView basisGradsAtQPointsCells("inValues", numOwnedElems, basisCardinality, numQPoints, dim);
      DynRankView ConstructWithLabel(basisGradsAtQPoints, basisCardinality, numQPoints, dim);
      basis_->getValues(basisGradsAtQPoints, quadPoints, Intrepid2::OPERATOR_GRAD);
      rst::clone(basisGradsAtQPointsCells,basisGradsAtQPoints);

      // modify basis values to account for orientations
      ots::modifyBasisByOrientation(basisGradsAtQPointsOriented,
                                    basisGradsAtQPointsCells,
                                    elemOrts_,
                                    basis_.getRawPtr());

      // map basis functions to reference (oriented) element
      DynRankView ConstructWithLabel(jacobianAtQPoints, numOwnedElems, numQPoints, dim, dim);
      DynRankView ConstructWithLabel(jacobianAtQPoints_inv, numOwnedElems, numQPoints, dim, dim);
      DynRankView ConstructWithLabel(jacobianAtQPoints_det, numOwnedElems, numQPoints);
      ct::setJacobian(jacobianAtQPoints, quadPoints, physVertices_, topology_);
      ct::setJacobianInv (jacobianAtQPoints_inv, jacobianAtQPoints);

      fst::HGRADtransformGRAD(transformedBasisGradsAtQPointsOriented, jacobianAtQPoints_inv, basisGradsAtQPointsOriented);

      // compute integrals to assembly local matrices

      DynRankView ConstructWithLabel(weightedTransformedBasisValuesAtQPointsOriented, numOwnedElems, basisCardinality, numQPoints);
      DynRankView ConstructWithLabel(weightedTransformedBasisGradsAtQPointsOriented, numOwnedElems, basisCardinality, numQPoints, dim);
      DynRankView ConstructWithLabel(cellWeights, numOwnedElems, numQPoints);
      rst::clone(cellWeights, weights);

      fst::multiplyMeasure(weightedTransformedBasisGradsAtQPointsOriented, cellWeights, transformedBasisGradsAtQPointsOriented);
      fst::multiplyMeasure(weightedTransformedBasisValuesAtQPointsOriented, cellWeights, transformedBasisValuesAtQPointsOriented);

      fst::integrate(elemsMat, transformedBasisGradsAtQPointsOriented, weightedTransformedBasisGradsAtQPointsOriented);
      fst::integrate(elemsMat, transformedBasisValuesAtQPointsOriented, weightedTransformedBasisValuesAtQPointsOriented, true);
    }

    diagonal.putScalar(0.);
    Y_ownedAndGhosted_->putScalar(0.);
    {
      auto elementLIDs = globalIndexer_->getLIDs();
      auto elmtOffsetKokkos = dofManager_->getGIDFieldOffsetsKokkos(blockId,0);

      auto lclDiag = Y_ownedAndGhosted_->getLocalViewDevice(Tpetra::Access::OverwriteAll);

      Kokkos::parallel_for
        ("Matrix-free apply",
         Kokkos::RangePolicy<DeviceSpaceType, int> (0, numOwnedElems),
         KOKKOS_LAMBDA (const size_t elemId) {
          // Get subviews
          auto elemMat = Kokkos::subview(elemsMat,elemId, Kokkos::ALL(), Kokkos::ALL());
          auto elemLIds  = Kokkos::subview(elementLIDs,elemId, Kokkos::ALL());

          // For each DoF (row) on the current element
          for (local_ordinal_t rowId = 0; rowId < basisCardinality; ++rowId) {
            const local_ordinal_t localRowId = elemLIds(elmtOffsetKokkos(rowId));

            Kokkos::atomic_add (&(lclDiag(localRowId, 0)), elemMat(rowId, rowId));
          }
        });
      Kokkos::fence();
    }
    diagonal.doExport(*Y_ownedAndGhosted_, *importer_, Tpetra::ADD_ASSIGN);
  }

  // Fake RowMatrix interface
  Teuchos::RCP<const Map<LocalOrdinal, GlobalOrdinal, Node> > getRowMap() const {
    return getRangeMap();
  }

  Teuchos::RCP<const Map<LocalOrdinal, GlobalOrdinal, Node> > getColMap() const {
    return ownedAndGhostedMap_;
  }

  Teuchos::RCP<const Teuchos::Comm<int> > getComm() const {
    return ownedMap_->getComm();
  }

  Teuchos::RCP<const RowGraph<LocalOrdinal, GlobalOrdinal, Node> > getGraph() const {
    throw std::runtime_error("Not implemented.");
  }

  global_size_t getGlobalNumRows() const {
    return getRangeMap()->getGlobalNumElements();
  }

  global_size_t getGlobalNumCols() const {
    return getDomainMap()->getGlobalNumElements();
  }

  size_t getLocalNumRows() const {
    return getRangeMap()->getLocalNumElements();
  }

  size_t getLocalNumCols() const {
    return getDomainMap()->getLocalNumElements();
  }

  GlobalOrdinal getIndexBase() const {
    return 0;
  }

  global_size_t getGlobalNumEntries() const {
    return 0;
  }

  size_t getLocalNumEntries() const {
    return 0;
  }

  size_t getNumEntriesInGlobalRow(GlobalOrdinal globalRow) const {
    throw std::runtime_error("Not implemented.");
  }

  size_t getNumEntriesInLocalRow(LocalOrdinal localRow) const {
    throw std::runtime_error("Not implemented.");
  }

  size_t getGlobalMaxNumRowEntries() const {
    throw std::runtime_error("Not implemented.");
  }

  LocalOrdinal getBlockSize() const {
    throw std::runtime_error("Not implemented.");
  }

  size_t getLocalMaxNumRowEntries() const {
    throw std::runtime_error("Not implemented.");
  }

  bool hasColMap() const {
    return false;
  }

  bool isLocallyIndexed() const {
    return true;
  }

  bool isGloballyIndexed() const {
    return true;
  }

  bool isFillComplete() const {
    return true;
  }

  bool supportsRowViews() const {
    return false;
  }

  void
  getGlobalRowCopy(GlobalOrdinal GlobalRow,
                   nonconst_global_inds_host_view_type& Indices,
                   nonconst_values_host_view_type& Values,
                   size_t& NumEntries) const {
    throw std::runtime_error("Not implemented.");
  }

  void
  getLocalRowCopy(LocalOrdinal LocalRow,
                  nonconst_local_inds_host_view_type& Indices,
                  nonconst_values_host_view_type& Values,
                  size_t& NumEntries) const {
    throw std::runtime_error("Not implemented.");
  }

  void
  getGlobalRowView(GlobalOrdinal GlobalRow,
                   global_inds_host_view_type& indices,
                   values_host_view_type& values) const {
    throw std::runtime_error("Not implemented.");
  }

  void
  getLocalRowView(LocalOrdinal LocalRow,
                  local_inds_host_view_type& indices,
                  values_host_view_type& values) const {
    throw std::runtime_error("Not implemented.");
  }

  void getLocalDiagCopy(Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>& diag) const {
    computeDiagonal(diag);
  }

  void leftScale(const Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>& x) {
    throw std::runtime_error("Not implemented.");
  }

  void rightScale(const Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>& x) {
    throw std::runtime_error("Not implemented.");
  }

  mag_type getFrobeniusNorm() const {
    return 0.;
  }

  void describe(Teuchos::FancyOStream& out, const Teuchos::EVerbosityLevel verbLevel) const {
    describe(out, verbLevel, true);
  }

  void describe(Teuchos::FancyOStream& out, const Teuchos::EVerbosityLevel verbLevel, const bool printHeader) const {
    out << "MatrixFreeRowMatrix" << std::endl;
  }

 private:

  Teuchos::RCP<const map_type> ownedMap_;
  Teuchos::RCP<const map_type> ownedAndGhostedMap_;
  shards::CellTopology topology_;
  Teuchos::RCP<basis_type> basis_;
  int cubDegree_;
  DynRankView physVertices_;
  element_orientation_type elemOrts_;
  Teuchos::RCP<panzer::DOFManager> dofManager_;
  Teuchos::RCP<const panzer::GlobalIndexer> globalIndexer_;
  Teuchos::RCP<const import_type> importer_;
  Teuchos::RCP<mv_type> X_ownedAndGhosted_, Y_ownedAndGhosted_;
};
}  // namespace Tpetra

#include "Tpetra_MatrixFreeRowMatrix_def.hpp"

#endif
