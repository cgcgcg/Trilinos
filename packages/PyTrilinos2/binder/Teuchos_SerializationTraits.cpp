#include <PyTrilinos2_Teuchos_Custom.hpp>
#include <Teuchos_SerializationTraits.hpp>
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

void bind_Teuchos_SerializationTraits(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // Teuchos::SerializationTraits file:Teuchos_SerializationTraits.hpp line:403
		pybind11::class_<Teuchos::SerializationTraits<int,unsigned long>, Teuchos::RCP<Teuchos::SerializationTraits<int,unsigned long>>, Teuchos::DirectSerializationTraits<int,unsigned long>> cl(M("Teuchos"), "SerializationTraits_int_unsigned_long_t", "");
		cl.def( pybind11::init( [](){ return new Teuchos::SerializationTraits<int,unsigned long>(); } ) );
		cl.def( pybind11::init( [](Teuchos::SerializationTraits<int,unsigned long> const &o){ return new Teuchos::SerializationTraits<int,unsigned long>(o); } ) );
		cl.def_static("fromCountToDirectBytes", (int (*)(const int)) &Teuchos::DirectSerializationTraits<int, unsigned long>::fromCountToDirectBytes, "C++: Teuchos::DirectSerializationTraits<int, unsigned long>::fromCountToDirectBytes(const int) --> int", pybind11::arg("count"));
		cl.def_static("convertToCharPtr", (char * (*)(unsigned long *)) &Teuchos::DirectSerializationTraits<int, unsigned long>::convertToCharPtr, "C++: Teuchos::DirectSerializationTraits<int, unsigned long>::convertToCharPtr(unsigned long *) --> char *", pybind11::return_value_policy::automatic, pybind11::arg("ptr"));
		cl.def_static("convertToCharPtr", (const char * (*)(const unsigned long *)) &Teuchos::DirectSerializationTraits<int, unsigned long>::convertToCharPtr, "C++: Teuchos::DirectSerializationTraits<int, unsigned long>::convertToCharPtr(const unsigned long *) --> const char *", pybind11::return_value_policy::automatic, pybind11::arg("ptr"));
		cl.def_static("fromDirectBytesToCount", (int (*)(const int)) &Teuchos::DirectSerializationTraits<int, unsigned long>::fromDirectBytesToCount, "C++: Teuchos::DirectSerializationTraits<int, unsigned long>::fromDirectBytesToCount(const int) --> int", pybind11::arg("count"));
		cl.def_static("convertFromCharPtr", (unsigned long * (*)(char *)) &Teuchos::DirectSerializationTraits<int, unsigned long>::convertFromCharPtr, "C++: Teuchos::DirectSerializationTraits<int, unsigned long>::convertFromCharPtr(char *) --> unsigned long *", pybind11::return_value_policy::automatic, pybind11::arg("ptr"));
		cl.def_static("convertFromCharPtr", (const unsigned long * (*)(const char *)) &Teuchos::DirectSerializationTraits<int, unsigned long>::convertFromCharPtr, "C++: Teuchos::DirectSerializationTraits<int, unsigned long>::convertFromCharPtr(const char *) --> const unsigned long *", pybind11::return_value_policy::automatic, pybind11::arg("ptr"));
	}
	{ // Teuchos::SerializationTraits file:Teuchos_SerializationTraits.hpp line:470
		pybind11::class_<Teuchos::SerializationTraits<int,long long>, Teuchos::RCP<Teuchos::SerializationTraits<int,long long>>, Teuchos::DirectSerializationTraits<int,long long>> cl(M("Teuchos"), "SerializationTraits_int_long_long_t", "");
		cl.def( pybind11::init( [](){ return new Teuchos::SerializationTraits<int,long long>(); } ) );
		cl.def( pybind11::init( [](Teuchos::SerializationTraits<int,long long> const &o){ return new Teuchos::SerializationTraits<int,long long>(o); } ) );
		cl.def_static("fromCountToDirectBytes", (int (*)(const int)) &Teuchos::DirectSerializationTraits<int, long long>::fromCountToDirectBytes, "C++: Teuchos::DirectSerializationTraits<int, long long>::fromCountToDirectBytes(const int) --> int", pybind11::arg("count"));
		cl.def_static("convertToCharPtr", (char * (*)(long long *)) &Teuchos::DirectSerializationTraits<int, long long>::convertToCharPtr, "C++: Teuchos::DirectSerializationTraits<int, long long>::convertToCharPtr(long long *) --> char *", pybind11::return_value_policy::automatic, pybind11::arg("ptr"));
		cl.def_static("convertToCharPtr", (const char * (*)(const long long *)) &Teuchos::DirectSerializationTraits<int, long long>::convertToCharPtr, "C++: Teuchos::DirectSerializationTraits<int, long long>::convertToCharPtr(const long long *) --> const char *", pybind11::return_value_policy::automatic, pybind11::arg("ptr"));
		cl.def_static("fromDirectBytesToCount", (int (*)(const int)) &Teuchos::DirectSerializationTraits<int, long long>::fromDirectBytesToCount, "C++: Teuchos::DirectSerializationTraits<int, long long>::fromDirectBytesToCount(const int) --> int", pybind11::arg("count"));
		cl.def_static("convertFromCharPtr", (long long * (*)(char *)) &Teuchos::DirectSerializationTraits<int, long long>::convertFromCharPtr, "C++: Teuchos::DirectSerializationTraits<int, long long>::convertFromCharPtr(char *) --> long long *", pybind11::return_value_policy::automatic, pybind11::arg("ptr"));
		cl.def_static("convertFromCharPtr", (const long long * (*)(const char *)) &Teuchos::DirectSerializationTraits<int, long long>::convertFromCharPtr, "C++: Teuchos::DirectSerializationTraits<int, long long>::convertFromCharPtr(const char *) --> const long long *", pybind11::return_value_policy::automatic, pybind11::arg("ptr"));
	}
	{ // Teuchos::SerializationTraits file:Teuchos_SerializationTraits.hpp line:363
		pybind11::class_<Teuchos::SerializationTraits<int,char>, Teuchos::RCP<Teuchos::SerializationTraits<int,char>>, Teuchos::DirectSerializationTraits<int,char>> cl(M("Teuchos"), "SerializationTraits_int_char_t", "");
		cl.def( pybind11::init( [](){ return new Teuchos::SerializationTraits<int,char>(); } ) );
		cl.def( pybind11::init( [](Teuchos::SerializationTraits<int,char> const &o){ return new Teuchos::SerializationTraits<int,char>(o); } ) );
		cl.def_static("fromCountToDirectBytes", (int (*)(const int)) &Teuchos::DirectSerializationTraits<int, char>::fromCountToDirectBytes, "C++: Teuchos::DirectSerializationTraits<int, char>::fromCountToDirectBytes(const int) --> int", pybind11::arg("count"));
		cl.def_static("convertToCharPtr", (char * (*)(char *)) &Teuchos::DirectSerializationTraits<int, char>::convertToCharPtr, "C++: Teuchos::DirectSerializationTraits<int, char>::convertToCharPtr(char *) --> char *", pybind11::return_value_policy::automatic, pybind11::arg("ptr"));
		cl.def_static("convertToCharPtr", (const char * (*)(const char *)) &Teuchos::DirectSerializationTraits<int, char>::convertToCharPtr, "C++: Teuchos::DirectSerializationTraits<int, char>::convertToCharPtr(const char *) --> const char *", pybind11::return_value_policy::automatic, pybind11::arg("ptr"));
		cl.def_static("fromDirectBytesToCount", (int (*)(const int)) &Teuchos::DirectSerializationTraits<int, char>::fromDirectBytesToCount, "C++: Teuchos::DirectSerializationTraits<int, char>::fromDirectBytesToCount(const int) --> int", pybind11::arg("count"));
		cl.def_static("convertFromCharPtr", (char * (*)(char *)) &Teuchos::DirectSerializationTraits<int, char>::convertFromCharPtr, "C++: Teuchos::DirectSerializationTraits<int, char>::convertFromCharPtr(char *) --> char *", pybind11::return_value_policy::automatic, pybind11::arg("ptr"));
		cl.def_static("convertFromCharPtr", (const char * (*)(const char *)) &Teuchos::DirectSerializationTraits<int, char>::convertFromCharPtr, "C++: Teuchos::DirectSerializationTraits<int, char>::convertFromCharPtr(const char *) --> const char *", pybind11::return_value_policy::automatic, pybind11::arg("ptr"));
	}
}
