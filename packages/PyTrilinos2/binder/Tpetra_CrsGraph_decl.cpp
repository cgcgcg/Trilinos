#include <Tpetra_CrsGraph_decl.hpp> // Tpetra::ELocalGlobal
#include <Tpetra_CrsGraph_decl.hpp> // Tpetra::RowInfo
#include <sstream> // __str__

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

void bind_Tpetra_CrsGraph_decl(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // Tpetra::RowInfo file:Tpetra_CrsGraph_decl.hpp line:106
		pybind11::class_<Tpetra::RowInfo, Teuchos::RCP<Tpetra::RowInfo>> cl(M("Tpetra"), "RowInfo", "Allocation information for a locally owned row in a\n   CrsGraph or CrsMatrix\n\n A RowInfo instance identifies a locally owned row uniquely by\n its local index, and contains other information useful for\n inserting entries into the row.  It is the return value of\n CrsGraph's getRowInfo() or updateAllocAndValues() methods.");
		cl.def( pybind11::init( [](){ return new Tpetra::RowInfo(); } ) );
		cl.def( pybind11::init( [](Tpetra::RowInfo const &o){ return new Tpetra::RowInfo(o); } ) );
		cl.def_readwrite("localRow", &Tpetra::RowInfo::localRow);
		cl.def_readwrite("allocSize", &Tpetra::RowInfo::allocSize);
		cl.def_readwrite("numEntries", &Tpetra::RowInfo::numEntries);
		cl.def_readwrite("offset1D", &Tpetra::RowInfo::offset1D);
		cl.def("assign", (struct Tpetra::RowInfo & (Tpetra::RowInfo::*)(const struct Tpetra::RowInfo &)) &Tpetra::RowInfo::operator=, "C++: Tpetra::RowInfo::operator=(const struct Tpetra::RowInfo &) --> struct Tpetra::RowInfo &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	// Tpetra::ELocalGlobal file:Tpetra_CrsGraph_decl.hpp line:113
	pybind11::enum_<Tpetra::ELocalGlobal>(M("Tpetra"), "ELocalGlobal", pybind11::arithmetic(), "")
		.value("LocalIndices", Tpetra::LocalIndices)
		.value("GlobalIndices", Tpetra::GlobalIndices)
		.export_values();

;

}
