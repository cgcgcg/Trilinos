// @HEADER
// *****************************************************************************
//           Galeri: Finite Element and Matrix Generation Package
//
// Copyright 2006 ETHZ/NTESS and the Galeri contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER
/*
  Direct translation of Galeri coordinate generator.
*/
#ifndef GALERI_XPETRAUTILS_DECL_HPP
#define GALERI_XPETRAUTILS_DECL_HPP

#include "Teuchos_ParameterList.hpp"

namespace Galeri::Xpetra {

class Utils {
 public:
  template <typename Scalar, typename LocalOrdinal, typename GlobalOrdinal, typename Map, typename RealValuedMultiVector>
  static Teuchos::RCP<RealValuedMultiVector>
  CreateCartesianCoordinates(std::string const& coordType, Teuchos::RCP<const Map> const& map, Teuchos::ParameterList& list);

  template <typename GlobalOrdinal>
  static void getSubdomainData(GlobalOrdinal n, GlobalOrdinal m, GlobalOrdinal i, GlobalOrdinal& start, GlobalOrdinal& end);

};  // class Utils
} // namespace Galeri::Xpetra


#endif  // ifndef GALERI_XPETRAUTILS_HPP
