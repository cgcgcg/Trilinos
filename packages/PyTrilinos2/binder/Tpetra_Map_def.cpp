#include <KokkosCompat_ClassicNodeAPI_Wrapper.hpp>
#include <Kokkos_Concepts.hpp>
#include <Kokkos_HostSpace.hpp>
#include <Kokkos_Serial.hpp>
#include <Kokkos_View.hpp>
#include <Teuchos_ArrayViewDecl.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_ENull.hpp>
#include <Teuchos_FancyOStream.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <Teuchos_RCPNode.hpp>
#include <Teuchos_VerbosityLevel.hpp>
#include <Teuchos_any.hpp>
#include <Tpetra_ConfigDefs.hpp>
#include <Tpetra_Details_LocalMap.hpp>
#include <Tpetra_Map_decl.hpp>
#include <Tpetra_Map_def.hpp>
#include <iterator>
#include <memory>
#include <ostream>
#include <string>

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
