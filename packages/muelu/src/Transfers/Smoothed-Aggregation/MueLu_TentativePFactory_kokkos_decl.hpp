// @HEADER
// *****************************************************************************
//        MueLu: A package for multigrid based preconditioning
//
// Copyright 2012 NTESS and the MueLu contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef MUELU_TENTATIVEPFACTORY_KOKKOS_DECL_HPP
#define MUELU_TENTATIVEPFACTORY_KOKKOS_DECL_HPP

#include "MueLu_TentativePFactory.hpp"

namespace MueLu {

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
class [[deprecated]] TentativePFactory_kokkos : public TentativePFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node> {};

}  // namespace MueLu

#define MUELU_TENTATIVEPFACTORY_KOKKOS_SHORT
#endif  // MUELU_TENTATIVEPFACTORY_KOKKOS_DECL_HPP
