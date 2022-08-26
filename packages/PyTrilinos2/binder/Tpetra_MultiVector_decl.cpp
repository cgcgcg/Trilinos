#include <KokkosCompat_ClassicNodeAPI_Wrapper.hpp> // Kokkos::Compat::KokkosDeviceWrapperNode
#include <KokkosSparse_CrsMatrix.hpp> // KokkosSparse::CrsMatrix
#include <KokkosSparse_CrsMatrix.hpp> // KokkosSparse::SparseRowView
#include <KokkosSparse_CrsMatrix.hpp> // KokkosSparse::SparseRowViewConst
#include <Kokkos_AnonymousSpace.hpp> // Kokkos::AnonymousSpace
#include <Kokkos_Concepts.hpp> // Kokkos::Device
#include <Kokkos_DualView.hpp> // Kokkos::DualView
#include <Kokkos_HostSpace.hpp> // 
#include <Kokkos_HostSpace.hpp> // Kokkos::HostSpace
#include <Kokkos_Layout.hpp> // Kokkos::LayoutLeft
#include <Kokkos_Layout.hpp> // Kokkos::LayoutRight
#include <Kokkos_Pair.hpp> // Kokkos::pair
#include <Kokkos_ScratchSpace.hpp> // Kokkos::ScratchMemorySpace
#include <Kokkos_Serial.hpp> // Kokkos::Impl::SerialInternal
#include <Kokkos_Serial.hpp> // Kokkos::Serial
#include <Kokkos_View.hpp> // Kokkos::View
#include <Kokkos_View.hpp> // Kokkos::ViewTraits
#include <Teuchos_ArrayRCPDecl.hpp> // Teuchos::ArrayRCP
#include <Teuchos_ArrayView.hpp> // Teuchos::ArrayView
#include <Teuchos_ArrayViewDecl.hpp> // Teuchos::ArrayView
#include <Teuchos_BLAS_types.hpp> // Teuchos::ETransp
#include <Teuchos_Comm.hpp> // Teuchos::Comm
#include <Teuchos_DataAccess.hpp> // Teuchos::DataAccess
#include <Teuchos_ENull.hpp> // Teuchos::ENull
#include <Teuchos_FancyOStream.hpp> // Teuchos::basic_FancyOStream
#include <Teuchos_RCPDecl.hpp> // Teuchos::ERCPUndefinedWeakNoDealloc
#include <Teuchos_RCPDecl.hpp> // Teuchos::ERCPWeakNoDealloc
#include <Teuchos_RCPDecl.hpp> // Teuchos::RCP
#include <Teuchos_RCPNode.hpp> // Teuchos::ERCPNodeLookup
#include <Teuchos_RCPNode.hpp> // Teuchos::ERCPStrength
#include <Teuchos_RCPNode.hpp> // Teuchos::RCPNodeHandle
#include <Teuchos_Range1D.hpp> // 
#include <Teuchos_Range1D.hpp> // Teuchos::Range1D
#include <Teuchos_VerbosityLevel.hpp> // Teuchos::EVerbosityLevel
#include <Tpetra_Access.hpp> // Tpetra::Access::OverwriteAllStruct
#include <Tpetra_Access.hpp> // Tpetra::Access::ReadOnlyStruct
#include <Tpetra_Access.hpp> // Tpetra::Access::ReadWriteStruct
#include <Tpetra_ConfigDefs.hpp> // Tpetra::LocalGlobal
#include <Tpetra_ConfigDefs.hpp> // Tpetra::LookupStatus
#include <Tpetra_Details_LocalMap.hpp> // Tpetra::Details::LocalMap
#include <Tpetra_Details_WrappedDualView.hpp> // Tpetra::Details::WrappedDualView
#include <Tpetra_LocalCrsMatrixOperator_decl.hpp> // Tpetra::LocalCrsMatrixOperator
#include <Tpetra_LocalOperator.hpp> // Tpetra::LocalOperator
#include <Tpetra_Map_decl.hpp> // Tpetra::Map
#include <Tpetra_MultiVector_decl.hpp> // Tpetra::MultiVector
#include <Tpetra_MultiVector_decl.hpp> // Tpetra::createCopy
#include <Tpetra_MultiVector_decl.hpp> // Tpetra::deep_copy
#include <Tpetra_MultiVector_decl.hpp> // Tpetra::getMultiVectorWhichVectors
#include <Tpetra_Vector_decl.hpp> // Tpetra::Vector
#include <impl/Kokkos_SharedAlloc.hpp> // Kokkos::Impl::SharedAllocationRecord
#include <impl/Kokkos_SharedAlloc.hpp> // Kokkos::Impl::SharedAllocationTracker
#include <impl/Kokkos_ViewCtor.hpp> // Kokkos::Impl::ViewCtorProp
#include <impl/Kokkos_ViewMapping.hpp> // Kokkos::Impl::ALL_t
#include <impl/Kokkos_ViewMapping.hpp> // Kokkos::Impl::ViewMapping
#include <iterator> // __gnu_cxx::__normal_iterator
#include <memory> // std::allocator
#include <memory> // std::shared_ptr
#include <ostream> // std::basic_ostream
#include <sstream> // __str__
#include <string> // std::basic_string
#include <string> // std::char_traits
#include <vector> // std::vector

#include <functional>
#include <pybind11/pybind11.h>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <Teuchos_RCP.hpp>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, Teuchos::RCP<T>)
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>)
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*)
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>)
#endif

void bind_Tpetra_MultiVector_decl(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	// Tpetra::deep_copy(class Tpetra::MultiVector<double, int, long long> &, const class Tpetra::MultiVector<double, int, long long> &) file:Tpetra_MultiVector_decl.hpp line:2228
	M("Tpetra").def("deep_copy", (void (*)(class Tpetra::MultiVector<double, int, long long> &, const class Tpetra::MultiVector<double, int, long long> &)) &Tpetra::deep_copy<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>,double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>, "C++: Tpetra::deep_copy(class Tpetra::MultiVector<double, int, long long> &, const class Tpetra::MultiVector<double, int, long long> &) --> void", pybind11::arg("dst"), pybind11::arg("src"));

	// Tpetra::createCopy(const class Tpetra::MultiVector<double, int, long long> &) file:Tpetra_MultiVector_decl.hpp line:137
	M("Tpetra").def("createCopy", (class Tpetra::MultiVector<double, int, long long> (*)(const class Tpetra::MultiVector<double, int, long long> &)) &Tpetra::createCopy<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>, "C++: Tpetra::createCopy(const class Tpetra::MultiVector<double, int, long long> &) --> class Tpetra::MultiVector<double, int, long long>", pybind11::arg("src"));

	// Tpetra::getMultiVectorWhichVectors(const class Tpetra::MultiVector<double, int, long long> &) file:Tpetra_MultiVector_decl.hpp line:2254
	M("Tpetra").def("getMultiVectorWhichVectors", (class Teuchos::ArrayView<const unsigned long> (*)(const class Tpetra::MultiVector<double, int, long long> &)) &Tpetra::getMultiVectorWhichVectors<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>, "C++: Tpetra::getMultiVectorWhichVectors(const class Tpetra::MultiVector<double, int, long long> &) --> class Teuchos::ArrayView<const unsigned long>", pybind11::arg("X"));

	// Tpetra::deep_copy(class Tpetra::MultiVector<double, int, long long> &, const class Tpetra::MultiVector<double, int, long long> &) file:Tpetra_MultiVector_decl.hpp line:2428
	M("Tpetra").def("deep_copy", (void (*)(class Tpetra::MultiVector<double, int, long long> &, const class Tpetra::MultiVector<double, int, long long> &)) &Tpetra::deep_copy<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>, "C++: Tpetra::deep_copy(class Tpetra::MultiVector<double, int, long long> &, const class Tpetra::MultiVector<double, int, long long> &) --> void", pybind11::arg("dst"), pybind11::arg("src"));

}
