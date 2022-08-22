#include <KokkosCompat_ClassicNodeAPI_Wrapper.hpp>
#include <Kokkos_AnonymousSpace.hpp>
#include <Kokkos_Concepts.hpp>
#include <Kokkos_DualView.hpp>
#include <Kokkos_HostSpace.hpp>
#include <Kokkos_Layout.hpp>
#include <Kokkos_Pair.hpp>
#include <Kokkos_ScratchSpace.hpp>
#include <Kokkos_Serial.hpp>
#include <Kokkos_View.hpp>
#include <Teuchos_ArrayRCPDecl.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_ArrayViewDecl.hpp>
#include <Teuchos_BLAS_types.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_DataAccess.hpp>
#include <Teuchos_ENull.hpp>
#include <Teuchos_FancyOStream.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <Teuchos_RCPNode.hpp>
#include <Teuchos_Range1D.hpp>
#include <Teuchos_VerbosityLevel.hpp>
#include <Tpetra_Access.hpp>
#include <Tpetra_ConfigDefs.hpp>
#include <Tpetra_Details_LocalMap.hpp>
#include <Tpetra_Details_WrappedDualView.hpp>
#include <Tpetra_Map_decl.hpp>
#include <Tpetra_MultiVector_decl.hpp>
#include <Tpetra_Vector_decl.hpp>
#include <impl/Kokkos_SharedAlloc.hpp>
#include <impl/Kokkos_ViewCtor.hpp>
#include <impl/Kokkos_ViewMapping.hpp>
#include <iterator>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

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
	// Tpetra::deep_copy(class Tpetra::MultiVector<double, int, long long> &, const class Tpetra::MultiVector<double, int, long long> &) file:Tpetra_MultiVector_decl.hpp line:2441
	M("Tpetra").def("deep_copy", (void (*)(class Tpetra::MultiVector<double, int, long long> &, const class Tpetra::MultiVector<double, int, long long> &)) &Tpetra::deep_copy<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>,double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>, "C++: Tpetra::deep_copy(class Tpetra::MultiVector<double, int, long long> &, const class Tpetra::MultiVector<double, int, long long> &) --> void", pybind11::arg("dst"), pybind11::arg("src"));

	// Tpetra::createCopy(const class Tpetra::MultiVector<double, int, long long> &) file:Tpetra_MultiVector_decl.hpp line:137
	M("Tpetra").def("createCopy", (class Tpetra::MultiVector<double, int, long long> (*)(const class Tpetra::MultiVector<double, int, long long> &)) &Tpetra::createCopy<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>, "C++: Tpetra::createCopy(const class Tpetra::MultiVector<double, int, long long> &) --> class Tpetra::MultiVector<double, int, long long>", pybind11::arg("src"));

	// Tpetra::getMultiVectorWhichVectors(const class Tpetra::MultiVector<double, int, long long> &) file:Tpetra_MultiVector_decl.hpp line:2418
	M("Tpetra").def("getMultiVectorWhichVectors", (class Teuchos::ArrayView<const unsigned long> (*)(const class Tpetra::MultiVector<double, int, long long> &)) &Tpetra::getMultiVectorWhichVectors<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>, "C++: Tpetra::getMultiVectorWhichVectors(const class Tpetra::MultiVector<double, int, long long> &) --> class Teuchos::ArrayView<const unsigned long>", pybind11::arg("X"));

	// Tpetra::deep_copy(class Tpetra::MultiVector<double, int, long long> &, const class Tpetra::MultiVector<double, int, long long> &) file:Tpetra_MultiVector_decl.hpp line:2428
	M("Tpetra").def("deep_copy", (void (*)(class Tpetra::MultiVector<double, int, long long> &, const class Tpetra::MultiVector<double, int, long long> &)) &Tpetra::deep_copy<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>, "C++: Tpetra::deep_copy(class Tpetra::MultiVector<double, int, long long> &, const class Tpetra::MultiVector<double, int, long long> &) --> void", pybind11::arg("dst"), pybind11::arg("src"));

}
