#include <Tpetra_Access.hpp> // Tpetra::Access::OverwriteAllStruct
#include <Tpetra_Access.hpp> // Tpetra::Access::ReadOnlyStruct
#include <Tpetra_Access.hpp> // Tpetra::Access::ReadWriteStruct
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

void bind_Tpetra_Access(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // Tpetra::Access::ReadOnlyStruct file:Tpetra_Access.hpp line:50
		pybind11::class_<Tpetra::Access::ReadOnlyStruct, Teuchos::RCP<Tpetra::Access::ReadOnlyStruct>> cl(M("Tpetra::Access"), "ReadOnlyStruct", "");
		cl.def( pybind11::init( [](){ return new Tpetra::Access::ReadOnlyStruct(); } ) );
		cl.def( pybind11::init( [](Tpetra::Access::ReadOnlyStruct const &o){ return new Tpetra::Access::ReadOnlyStruct(o); } ) );
	}
	{ // Tpetra::Access::OverwriteAllStruct file:Tpetra_Access.hpp line:51
		pybind11::class_<Tpetra::Access::OverwriteAllStruct, Teuchos::RCP<Tpetra::Access::OverwriteAllStruct>> cl(M("Tpetra::Access"), "OverwriteAllStruct", "");
		cl.def( pybind11::init( [](){ return new Tpetra::Access::OverwriteAllStruct(); } ) );
		cl.def( pybind11::init( [](Tpetra::Access::OverwriteAllStruct const &o){ return new Tpetra::Access::OverwriteAllStruct(o); } ) );
	}
	{ // Tpetra::Access::ReadWriteStruct file:Tpetra_Access.hpp line:52
		pybind11::class_<Tpetra::Access::ReadWriteStruct, Teuchos::RCP<Tpetra::Access::ReadWriteStruct>> cl(M("Tpetra::Access"), "ReadWriteStruct", "");
		cl.def( pybind11::init( [](){ return new Tpetra::Access::ReadWriteStruct(); } ) );
		cl.def( pybind11::init( [](Tpetra::Access::ReadWriteStruct const &o){ return new Tpetra::Access::ReadWriteStruct(o); } ) );
	}
}
