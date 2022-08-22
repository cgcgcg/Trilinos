#include <Teuchos_Array.hpp>
#include <Tpetra_Distributor.hpp>
#include <iterator>
#include <memory>
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

void bind_Tpetra_Distributor(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	// Tpetra::distributorSendTypes() file:Tpetra_Distributor.hpp line:68
	M("Tpetra").def("distributorSendTypes", (class Teuchos::Array<std::string > (*)()) &Tpetra::distributorSendTypes, "Valid values for Distributor's \"Send type\" parameter.\n\n This is mainly useful as an implementation detail of\n Distributor.  You may use it if you would like a programmatic\n way to get all possible values of the \"Send type\" parameter of\n Distributor.\n\nC++: Tpetra::distributorSendTypes() --> class Teuchos::Array<std::string >");

}
