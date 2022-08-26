#include <KokkosCompat_ClassicNodeAPI_Wrapper.hpp> // Kokkos::Compat::KokkosDeviceWrapperNode
#include <Kokkos_Concepts.hpp> // Kokkos::Device
#include <Kokkos_HostSpace.hpp> // 
#include <Kokkos_HostSpace.hpp> // Kokkos::HostSpace
#include <Kokkos_Serial.hpp> // Kokkos::Impl::SerialInternal
#include <Kokkos_Serial.hpp> // Kokkos::Serial
#include <Kokkos_View.hpp> // Kokkos::View
#include <Teuchos_ArrayViewDecl.hpp> // Teuchos::ArrayView
#include <Teuchos_Comm.hpp> // Teuchos::Comm
#include <Teuchos_ENull.hpp> // Teuchos::ENull
#include <Teuchos_FancyOStream.hpp> // Teuchos::basic_FancyOStream
#include <Teuchos_RCPDecl.hpp> // Teuchos::ERCPUndefinedWeakNoDealloc
#include <Teuchos_RCPDecl.hpp> // Teuchos::ERCPWeakNoDealloc
#include <Teuchos_RCPDecl.hpp> // Teuchos::RCP
#include <Teuchos_RCPNode.hpp> // Teuchos::EPrePostDestruction
#include <Teuchos_RCPNode.hpp> // Teuchos::ERCPStrength
#include <Teuchos_RCPNode.hpp> // Teuchos::RCPNode
#include <Teuchos_RCPNode.hpp> // Teuchos::RCPNodeHandle
#include <Teuchos_VerbosityLevel.hpp> // Teuchos::EVerbosityLevel
#include <Teuchos_any.hpp> // Teuchos::any
#include <Tpetra_ConfigDefs.hpp> // Tpetra::LocalGlobal
#include <Tpetra_ConfigDefs.hpp> // Tpetra::LookupStatus
#include <Tpetra_Details_LocalMap.hpp> // Tpetra::Details::LocalMap
#include <Tpetra_Map_decl.hpp> // Tpetra::Map
#include <Tpetra_Map_def.hpp> // Tpetra::createOneToOne
#include <iterator> // __gnu_cxx::__normal_iterator
#include <memory> // std::allocator
#include <ostream> // std::basic_ostream
#include <string> // std::basic_string
#include <string> // std::char_traits

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

void bind_Tpetra_Map_def(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	// Tpetra::createOneToOne(const class Teuchos::RCP<const class Tpetra::Map<int, long long> > &) file:Tpetra_Map_def.hpp line:2469
	M("Tpetra").def("createOneToOne", (class Teuchos::RCP<const class Tpetra::Map<int, long long> > (*)(const class Teuchos::RCP<const class Tpetra::Map<int, long long> > &)) &Tpetra::createOneToOne<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>, "C++: Tpetra::createOneToOne(const class Teuchos::RCP<const class Tpetra::Map<int, long long> > &) --> class Teuchos::RCP<const class Tpetra::Map<int, long long> >", pybind11::arg("M"));

}
