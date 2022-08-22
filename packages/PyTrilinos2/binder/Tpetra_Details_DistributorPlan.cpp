#include <Teuchos_ArrayViewDecl.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_ENull.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_PtrDecl.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <Teuchos_RCPNode.hpp>
#include <Tpetra_Details_DistributorPlan.hpp>
#include <iterator>
#include <memory>
#include <sstream> // __str__
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

void bind_Tpetra_Details_DistributorPlan(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	// Tpetra::Details::EDistributorHowInitialized file:Tpetra_Details_DistributorPlan.hpp line:72
	pybind11::enum_<Tpetra::Details::EDistributorHowInitialized>(M("Tpetra::Details"), "EDistributorHowInitialized", pybind11::arithmetic(), "Enum indicating how and whether a Distributor was initialized.\n\n This is an implementation detail of Distributor.  Please do\n not rely on these values in your code.")
		.value("DISTRIBUTOR_NOT_INITIALIZED", Tpetra::Details::DISTRIBUTOR_NOT_INITIALIZED)
		.value("DISTRIBUTOR_INITIALIZED_BY_CREATE_FROM_SENDS", Tpetra::Details::DISTRIBUTOR_INITIALIZED_BY_CREATE_FROM_SENDS)
		.value("DISTRIBUTOR_INITIALIZED_BY_CREATE_FROM_RECVS", Tpetra::Details::DISTRIBUTOR_INITIALIZED_BY_CREATE_FROM_RECVS)
		.value("DISTRIBUTOR_INITIALIZED_BY_CREATE_FROM_SENDS_N_RECVS", Tpetra::Details::DISTRIBUTOR_INITIALIZED_BY_CREATE_FROM_SENDS_N_RECVS)
		.value("DISTRIBUTOR_INITIALIZED_BY_REVERSE", Tpetra::Details::DISTRIBUTOR_INITIALIZED_BY_REVERSE)
		.value("DISTRIBUTOR_INITIALIZED_BY_COPY", Tpetra::Details::DISTRIBUTOR_INITIALIZED_BY_COPY)
		.export_values();

;

}
