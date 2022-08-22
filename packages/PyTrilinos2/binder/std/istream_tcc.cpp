#include <cwchar>
#include <ios>
#include <istream>
#include <locale>
#include <sstream> // __str__
#include <streambuf>
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

void bind_std_istream_tcc(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // std::basic_istream file:bits/istream.tcc line:1048
		pybind11::class_<std::istream, std::shared_ptr<std::istream>> cl(M("std"), "istream", "");
		cl.def( pybind11::init<class std::basic_streambuf<char> *>(), pybind11::arg("__sb") );

		cl.def("gcount", (long (std::istream::*)() const) &std::basic_istream<char, std::char_traits<char> >::gcount, "C++: std::basic_istream<char, std::char_traits<char> >::gcount() const --> long");
		cl.def("get", (int (std::istream::*)()) &std::basic_istream<char, std::char_traits<char> >::get, "C++: std::basic_istream<char, std::char_traits<char> >::get() --> int");
		cl.def("get", (std::istream & (std::istream::*)(char &)) &std::basic_istream<char, std::char_traits<char> >::get, "C++: std::basic_istream<char, std::char_traits<char> >::get(char &) --> std::istream &", pybind11::return_value_policy::automatic, pybind11::arg("__c"));
		cl.def("get", (std::istream & (std::istream::*)(char *, long, char)) &std::basic_istream<char, std::char_traits<char> >::get, "C++: std::basic_istream<char, std::char_traits<char> >::get(char *, long, char) --> std::istream &", pybind11::return_value_policy::automatic, pybind11::arg("__s"), pybind11::arg("__n"), pybind11::arg("__delim"));
		cl.def("get", (std::istream & (std::istream::*)(char *, long)) &std::basic_istream<char, std::char_traits<char> >::get, "C++: std::basic_istream<char, std::char_traits<char> >::get(char *, long) --> std::istream &", pybind11::return_value_policy::automatic, pybind11::arg("__s"), pybind11::arg("__n"));
		cl.def("get", (std::istream & (std::istream::*)(class std::basic_streambuf<char> &, char)) &std::basic_istream<char, std::char_traits<char> >::get, "C++: std::basic_istream<char, std::char_traits<char> >::get(class std::basic_streambuf<char> &, char) --> std::istream &", pybind11::return_value_policy::automatic, pybind11::arg("__sb"), pybind11::arg("__delim"));
		cl.def("get", (std::istream & (std::istream::*)(class std::basic_streambuf<char> &)) &std::basic_istream<char, std::char_traits<char> >::get, "C++: std::basic_istream<char, std::char_traits<char> >::get(class std::basic_streambuf<char> &) --> std::istream &", pybind11::return_value_policy::automatic, pybind11::arg("__sb"));
		cl.def("getline", (std::istream & (std::istream::*)(char *, long, char)) &std::basic_istream<char, std::char_traits<char> >::getline, "C++: std::basic_istream<char, std::char_traits<char> >::getline(char *, long, char) --> std::istream &", pybind11::return_value_policy::automatic, pybind11::arg("__s"), pybind11::arg("__n"), pybind11::arg("__delim"));
		cl.def("getline", (std::istream & (std::istream::*)(char *, long)) &std::basic_istream<char, std::char_traits<char> >::getline, "C++: std::basic_istream<char, std::char_traits<char> >::getline(char *, long) --> std::istream &", pybind11::return_value_policy::automatic, pybind11::arg("__s"), pybind11::arg("__n"));
		cl.def("ignore", (std::istream & (std::istream::*)(long, int)) &std::basic_istream<char, std::char_traits<char> >::ignore, "C++: std::basic_istream<char, std::char_traits<char> >::ignore(long, int) --> std::istream &", pybind11::return_value_policy::automatic, pybind11::arg("__n"), pybind11::arg("__delim"));
		cl.def("ignore", (std::istream & (std::istream::*)(long)) &std::basic_istream<char, std::char_traits<char> >::ignore, "C++: std::basic_istream<char, std::char_traits<char> >::ignore(long) --> std::istream &", pybind11::return_value_policy::automatic, pybind11::arg("__n"));
		cl.def("ignore", (std::istream & (std::istream::*)()) &std::basic_istream<char, std::char_traits<char> >::ignore, "C++: std::basic_istream<char, std::char_traits<char> >::ignore() --> std::istream &", pybind11::return_value_policy::automatic);
		cl.def("peek", (int (std::istream::*)()) &std::basic_istream<char, std::char_traits<char> >::peek, "C++: std::basic_istream<char, std::char_traits<char> >::peek() --> int");
		cl.def("read", (std::istream & (std::istream::*)(char *, long)) &std::basic_istream<char, std::char_traits<char> >::read, "C++: std::basic_istream<char, std::char_traits<char> >::read(char *, long) --> std::istream &", pybind11::return_value_policy::automatic, pybind11::arg("__s"), pybind11::arg("__n"));
		cl.def("readsome", (long (std::istream::*)(char *, long)) &std::basic_istream<char, std::char_traits<char> >::readsome, "C++: std::basic_istream<char, std::char_traits<char> >::readsome(char *, long) --> long", pybind11::arg("__s"), pybind11::arg("__n"));
		cl.def("putback", (std::istream & (std::istream::*)(char)) &std::basic_istream<char, std::char_traits<char> >::putback, "C++: std::basic_istream<char, std::char_traits<char> >::putback(char) --> std::istream &", pybind11::return_value_policy::automatic, pybind11::arg("__c"));
		cl.def("unget", (std::istream & (std::istream::*)()) &std::basic_istream<char, std::char_traits<char> >::unget, "C++: std::basic_istream<char, std::char_traits<char> >::unget() --> std::istream &", pybind11::return_value_policy::automatic);
		cl.def("sync", (int (std::istream::*)()) &std::basic_istream<char, std::char_traits<char> >::sync, "C++: std::basic_istream<char, std::char_traits<char> >::sync() --> int");
		cl.def("tellg", (class std::fpos<__mbstate_t> (std::istream::*)()) &std::basic_istream<char, std::char_traits<char> >::tellg, "C++: std::basic_istream<char, std::char_traits<char> >::tellg() --> class std::fpos<__mbstate_t>");
		cl.def("seekg", (std::istream & (std::istream::*)(class std::fpos<__mbstate_t>)) &std::basic_istream<char, std::char_traits<char> >::seekg, "C++: std::basic_istream<char, std::char_traits<char> >::seekg(class std::fpos<__mbstate_t>) --> std::istream &", pybind11::return_value_policy::automatic, pybind11::arg(""));
		cl.def("seekg", (std::istream & (std::istream::*)(long, enum std::_Ios_Seekdir)) &std::basic_istream<char, std::char_traits<char> >::seekg, "C++: std::basic_istream<char, std::char_traits<char> >::seekg(long, enum std::_Ios_Seekdir) --> std::istream &", pybind11::return_value_policy::automatic, pybind11::arg(""), pybind11::arg(""));

		{ // std::basic_istream<char, std::char_traits<char> >::sentry file:istream line:107
			auto & enclosing_class = cl;
			pybind11::class_<std::basic_istream<char, std::char_traits<char> >::sentry, std::shared_ptr<std::basic_istream<char, std::char_traits<char> >::sentry>> cl(enclosing_class, "sentry", "");
			cl.def( pybind11::init( [](std::istream & a0){ return new std::basic_istream<char, std::char_traits<char> >::sentry(a0); } ), "doc" , pybind11::arg("__is"));
			cl.def( pybind11::init<std::istream &, bool>(), pybind11::arg("__is"), pybind11::arg("__noskipws") );

		}

	}
}
