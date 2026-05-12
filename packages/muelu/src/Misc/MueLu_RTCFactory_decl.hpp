// @HEADER
// *****************************************************************************
//        MueLu: A package for multigrid based preconditioning
//
// Copyright 2012 NTESS and the MueLu contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef MUELU_RTCFACTORY_DECL_HPP
#define MUELU_RTCFACTORY_DECL_HPP

#include "MueLu_ConfigDefs.hpp"

#include "Xpetra_MultiVector_fwd.hpp"
#include "Xpetra_MultiVectorFactory_fwd.hpp"

#include "MueLu_RTCFactory_fwd.hpp"

#include "MueLu_Level_fwd.hpp"
#include "MueLu_SingleLevelFactoryBase.hpp"

namespace MueLu {

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>

class RTCFactory : public SingleLevelFactoryBase {
 public:
  typedef LocalOrdinal local_ordinal_type;
  typedef GlobalOrdinal global_ordinal_type;
  typedef typename Node::execution_space execution_space;
  typedef Kokkos::RangePolicy<local_ordinal_type, execution_space> range_type;
  typedef Node node_type;

 private:
#undef MUELU_RTCFACTORY_SHORT
#include "MueLu_UseShortNames.hpp"

 public:
  //! @name Constructors/Destructors.
  //@{

  //! Constructor
  RTCFactory();

  //! Destructor
  ~RTCFactory();

  //@}

  //! @name Input

  //@{

  /*! @brief Define valid parameters for internal factory parameters */
  RCP<const ParameterList> GetValidParameterList() const;

  /*! @brief Specifies the data that this class needs, and the factories that generate that data.

      If the Build method of this class requires some data, but the generating factory is not specified in DeclareInput, then this class
      will fall back to the settings in FactoryManager.
  */

  void DeclareInput(Level& currentLevel) const;

  //@}

  //! @name Build methods.
  //@{

  //! Build an object with this factory.
  void Build(Level& currentLevel) const;

  //@}

 private:
  using coordinate_type       = typename Teuchos::ScalarTraits<Scalar>::coordinateType;
  using RealValuedMultiVector = Xpetra::MultiVector<coordinate_type, LO, GO, NO>;
  using CoordsType            = typename RealValuedMultiVector::dual_view_type::t_dev_const_um;
  using MeanCoordsType        = Kokkos::View<typename RealValuedMultiVector::impl_scalar_type*, typename Node::memory_space>;
};

}  // namespace MueLu

#define MUELU_RTCFACTORY_SHORT
#endif  // MUELU_RTCFACTORY_DECL_HPP
