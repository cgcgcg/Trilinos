#include <KokkosCompat_ClassicNodeAPI_Wrapper.hpp> // Kokkos::Compat::KokkosDeviceWrapperNode
#include <Kokkos_HostSpace.hpp> // Kokkos::HostSpace
#include <Kokkos_Serial.hpp> // Kokkos::Serial
#include <PyTrilinos2_Teuchos_Custom.hpp>
#include <Teuchos_Array.hpp> // Teuchos::Array
#include <Teuchos_ArrayViewDecl.hpp> // Teuchos::ArrayView
#include <Teuchos_BLAS.hpp> // Teuchos::BLAS
#include <Teuchos_BLAS_types.hpp> // Teuchos::EDiag
#include <Teuchos_BLAS_types.hpp> // Teuchos::ESide
#include <Teuchos_BLAS_types.hpp> // Teuchos::ETransp
#include <Teuchos_BLAS_types.hpp> // Teuchos::EUplo
#include <Teuchos_CompObject.hpp> // Teuchos::CompObject
#include <Teuchos_ENull.hpp> // Teuchos::ENull
#include <Teuchos_Flops.hpp> // Teuchos::Flops
#include <Teuchos_HashUtils.hpp> // Teuchos::HashUtils
#include <Teuchos_HashUtils.hpp> // Teuchos::hashCode
#include <Teuchos_Hashtable.hpp> // Teuchos::HashPair
#include <Teuchos_Hashtable.hpp> // Teuchos::Hashtable
#include <Teuchos_Object.hpp> // Teuchos::Object
#include <Teuchos_Object.hpp> // Teuchos::operator<<
#include <Teuchos_RCPDecl.hpp> // Teuchos::ERCPUndefinedWeakNoDealloc
#include <Teuchos_RCPDecl.hpp> // Teuchos::ERCPWeakNoDealloc
#include <Teuchos_RCPDecl.hpp> // Teuchos::RCP
#include <Teuchos_RCPNode.hpp> // Teuchos::ERCPStrength
#include <Teuchos_RCPNode.hpp> // Teuchos::RCPNodeHandle
#include <Xpetra_MatrixView.hpp> // Xpetra::MatrixView
#include <complex> // std::complex
#include <cwchar> // (anonymous)
#include <ios> // std::_Ios_Openmode
#include <ios> // std::_Ios_Seekdir
#include <ios> // std::fpos
#include <iterator> // __gnu_cxx::__normal_iterator
#include <locale> // std::locale
#include <memory> // std::allocator
#include <ostream> // std::basic_ostream
#include <sstream> // __str__
#include <streambuf> // std::basic_streambuf
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

// Teuchos::Object file:Teuchos_Object.hpp line:68
struct PyCallBack_Teuchos_Object : public Teuchos::Object {
	using Teuchos::Object::Object;

	void setLabel(const char * a0) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::Object *>(this), "setLabel");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Object::setLabel(a0);
	}
	const char * label() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::Object *>(this), "label");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<const char *>::value) {
				static pybind11::detail::override_caster_t<const char *> caster;
				return pybind11::detail::cast_ref<const char *>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<const char *>(std::move(o));
		}
		return Object::label();
	}
	int reportError(const std::string a0, int a1) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::Object *>(this), "reportError");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<int>::value) {
				static pybind11::detail::override_caster_t<int> caster;
				return pybind11::detail::cast_ref<int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<int>(std::move(o));
		}
		return Object::reportError(a0, a1);
	}
};

void bind_Teuchos_Object(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // Teuchos::Object file:Teuchos_Object.hpp line:68
		pybind11::class_<Teuchos::Object, Teuchos::RCP<Teuchos::Object>, PyCallBack_Teuchos_Object> cl(M("Teuchos"), "Object", "");
		cl.def( pybind11::init( [](){ return new Teuchos::Object(); }, [](){ return new PyCallBack_Teuchos_Object(); } ), "doc");
		cl.def( pybind11::init<int>(), pybind11::arg("tracebackModeIn") );

		cl.def( pybind11::init( [](const char * a0){ return new Teuchos::Object(a0); }, [](const char * a0){ return new PyCallBack_Teuchos_Object(a0); } ), "doc");
		cl.def( pybind11::init<const char *, int>(), pybind11::arg("label"), pybind11::arg("tracebackModeIn") );

		cl.def( pybind11::init( [](const std::string & a0){ return new Teuchos::Object(a0); }, [](const std::string & a0){ return new PyCallBack_Teuchos_Object(a0); } ), "doc");
		cl.def( pybind11::init<const std::string &, int>(), pybind11::arg("label"), pybind11::arg("tracebackModeIn") );

		cl.def( pybind11::init( [](PyCallBack_Teuchos_Object const &o){ return new PyCallBack_Teuchos_Object(o); } ) );
		cl.def( pybind11::init( [](Teuchos::Object const &o){ return new Teuchos::Object(o); } ) );
		cl.def("setLabel", (void (Teuchos::Object::*)(const char *)) &Teuchos::Object::setLabel, "C++: Teuchos::Object::setLabel(const char *) --> void", pybind11::arg("theLabel"));
		cl.def_static("setTracebackMode", (void (*)(int)) &Teuchos::Object::setTracebackMode, "Set the value of the Object error traceback report mode.\n\n TracebackMode controls whether or not traceback information is\n printed when run time integer errors are detected:\n\n <= 0 - No information report\n\n = 1 - Fatal (negative) values are reported\n\n >= 2 - All values (except zero) reported.\n\n \n Default is set to -1 when object is constructed.\n\nC++: Teuchos::Object::setTracebackMode(int) --> void", pybind11::arg("tracebackModeValue"));
		cl.def("label", (const char * (Teuchos::Object::*)() const) &Teuchos::Object::label, "Access the object's label (LEGACY; return std::string instead).\n\nC++: Teuchos::Object::label() const --> const char *", pybind11::return_value_policy::automatic);
		cl.def_static("getTracebackMode", (int (*)()) &Teuchos::Object::getTracebackMode, "Get the value of the Object error traceback report mode.\n\nC++: Teuchos::Object::getTracebackMode() --> int");
		cl.def("reportError", (int (Teuchos::Object::*)(const std::string, int) const) &Teuchos::Object::reportError, "Report an error with this Object.\n\nC++: Teuchos::Object::reportError(const std::string, int) const --> int", pybind11::arg("message"), pybind11::arg("errorCode"));
		cl.def("assign", (class Teuchos::Object & (Teuchos::Object::*)(const class Teuchos::Object &)) &Teuchos::Object::operator=, "C++: Teuchos::Object::operator=(const class Teuchos::Object &) --> class Teuchos::Object &", pybind11::return_value_policy::automatic, pybind11::arg(""));

		cl.def("__str__", [](Teuchos::Object const &o) -> std::string { std::ostringstream s; s << o; return s.str(); } );
	}
	{ // Teuchos::Flops file:Teuchos_Flops.hpp line:66
		pybind11::class_<Teuchos::Flops, Teuchos::RCP<Teuchos::Flops>> cl(M("Teuchos"), "Flops", "");
		cl.def( pybind11::init( [](){ return new Teuchos::Flops(); } ) );
		cl.def( pybind11::init( [](Teuchos::Flops const &o){ return new Teuchos::Flops(o); } ) );
		cl.def("flops", (double (Teuchos::Flops::*)() const) &Teuchos::Flops::flops, "Returns the number of floating point operations with  object and resets the count.\n\nC++: Teuchos::Flops::flops() const --> double");
		cl.def("resetFlops", (void (Teuchos::Flops::*)()) &Teuchos::Flops::resetFlops, "Resets the number of floating point operations to zero for  multi-std::vector.\n\nC++: Teuchos::Flops::resetFlops() --> void");
		cl.def("assign", (class Teuchos::Flops & (Teuchos::Flops::*)(const class Teuchos::Flops &)) &Teuchos::Flops::operator=, "C++: Teuchos::Flops::operator=(const class Teuchos::Flops &) --> class Teuchos::Flops &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // Teuchos::CompObject file:Teuchos_CompObject.hpp line:65
		pybind11::class_<Teuchos::CompObject, Teuchos::RCP<Teuchos::CompObject>> cl(M("Teuchos"), "CompObject", "");
		cl.def( pybind11::init( [](){ return new Teuchos::CompObject(); } ) );
		cl.def( pybind11::init( [](Teuchos::CompObject const &o){ return new Teuchos::CompObject(o); } ) );
		cl.def("setFlopCounter", (void (Teuchos::CompObject::*)(const class Teuchos::Flops &)) &Teuchos::CompObject::setFlopCounter, "Set the internal Teuchos::Flops() pointer.\n\nC++: Teuchos::CompObject::setFlopCounter(const class Teuchos::Flops &) --> void", pybind11::arg("FlopCounter"));
		cl.def("setFlopCounter", (void (Teuchos::CompObject::*)(const class Teuchos::CompObject &)) &Teuchos::CompObject::setFlopCounter, "Set the internal Teuchos::Flops() pointer to the flop counter of another Teuchos::CompObject.\n\nC++: Teuchos::CompObject::setFlopCounter(const class Teuchos::CompObject &) --> void", pybind11::arg("compObject"));
		cl.def("unsetFlopCounter", (void (Teuchos::CompObject::*)()) &Teuchos::CompObject::unsetFlopCounter, "Set the internal Teuchos::Flops() pointer to 0 (no flops counted).\n\nC++: Teuchos::CompObject::unsetFlopCounter() --> void");
		cl.def("getFlopCounter", (class Teuchos::Flops * (Teuchos::CompObject::*)() const) &Teuchos::CompObject::getFlopCounter, "Get the pointer to the Teuchos::Flops() object associated with this object, returns 0 if none.\n\nC++: Teuchos::CompObject::getFlopCounter() const --> class Teuchos::Flops *", pybind11::return_value_policy::automatic);
		cl.def("resetFlops", (void (Teuchos::CompObject::*)() const) &Teuchos::CompObject::resetFlops, "Resets the number of floating point operations to zero for  multi-std::vector.\n\nC++: Teuchos::CompObject::resetFlops() const --> void");
		cl.def("getFlops", (double (Teuchos::CompObject::*)() const) &Teuchos::CompObject::getFlops, "Returns the number of floating point operations with  multi-std::vector.\n\nC++: Teuchos::CompObject::getFlops() const --> double");
		cl.def("updateFlops", (void (Teuchos::CompObject::*)(int) const) &Teuchos::CompObject::updateFlops, "Increment Flop count for  object\n\nC++: Teuchos::CompObject::updateFlops(int) const --> void", pybind11::arg("addflops"));
		cl.def("updateFlops", (void (Teuchos::CompObject::*)(long) const) &Teuchos::CompObject::updateFlops, "Increment Flop count for  object\n\nC++: Teuchos::CompObject::updateFlops(long) const --> void", pybind11::arg("addflops"));
		cl.def("updateFlops", (void (Teuchos::CompObject::*)(double) const) &Teuchos::CompObject::updateFlops, "Increment Flop count for  object\n\nC++: Teuchos::CompObject::updateFlops(double) const --> void", pybind11::arg("addflops"));
		cl.def("updateFlops", (void (Teuchos::CompObject::*)(float) const) &Teuchos::CompObject::updateFlops, "Increment Flop count for  object\n\nC++: Teuchos::CompObject::updateFlops(float) const --> void", pybind11::arg("addflops"));
		cl.def("assign", (class Teuchos::CompObject & (Teuchos::CompObject::*)(const class Teuchos::CompObject &)) &Teuchos::CompObject::operator=, "C++: Teuchos::CompObject::operator=(const class Teuchos::CompObject &) --> class Teuchos::CompObject &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // Teuchos::HashUtils file:Teuchos_HashUtils.hpp line:66
		pybind11::class_<Teuchos::HashUtils, Teuchos::RCP<Teuchos::HashUtils>> cl(M("Teuchos"), "HashUtils", "Utilities for generating hashcodes.\n This is not a true hash ! For all ints and types less than ints\n it returns the i/p typecasted as an int. For every type more than the\n size of int it is only slightly more smarter where it adds the bits\n into int size chunks and calls that an hash. Used with a capacity\n limited array this will lead to one of the simplest hashes.\n Ideally this should be deprecated and not used at all.");
		cl.def( pybind11::init( [](){ return new Teuchos::HashUtils(); } ) );
		cl.def_static("nextPrime", (int (*)(int)) &Teuchos::HashUtils::nextPrime, "C++: Teuchos::HashUtils::nextPrime(int) --> int", pybind11::arg("newCapacity"));
		cl.def_static("getHashCode", (int (*)(const unsigned char *, unsigned long)) &Teuchos::HashUtils::getHashCode, "C++: Teuchos::HashUtils::getHashCode(const unsigned char *, unsigned long) --> int", pybind11::arg("a"), pybind11::arg("len"));
	}
	// Teuchos::hashCode(const int &) file:Teuchos_HashUtils.hpp line:95
	M("Teuchos").def("hashCode", (int (*)(const int &)) &Teuchos::hashCode<int>, "Get the hash code of an int\n\nC++: Teuchos::hashCode(const int &) --> int", pybind11::arg("x"));

	// Teuchos::hashCode(const unsigned int &) file:Teuchos_HashUtils.hpp line:103
	M("Teuchos").def("hashCode", (int (*)(const unsigned int &)) &Teuchos::hashCode<unsigned int>, "Get the hash code of an unsigned\n\nC++: Teuchos::hashCode(const unsigned int &) --> int", pybind11::arg("x"));

	// Teuchos::hashCode(const double &) file:Teuchos_HashUtils.hpp line:112
	M("Teuchos").def("hashCode", (int (*)(const double &)) &Teuchos::hashCode<double>, "Get the hash code of a double\n\nC++: Teuchos::hashCode(const double &) --> int", pybind11::arg("x"));

	// Teuchos::hashCode(const bool &) file:Teuchos_HashUtils.hpp line:121
	M("Teuchos").def("hashCode", (int (*)(const bool &)) &Teuchos::hashCode<bool>, "Get the hash code of a bool\n\nC++: Teuchos::hashCode(const bool &) --> int", pybind11::arg("x"));

	// Teuchos::hashCode(const long long &) file:Teuchos_HashUtils.hpp line:129
	M("Teuchos").def("hashCode", (int (*)(const long long &)) &Teuchos::hashCode<long long>, "Get the hash code of a long long\n\nC++: Teuchos::hashCode(const long long &) --> int", pybind11::arg("x"));

	// Teuchos::hashCode(const long &) file:Teuchos_HashUtils.hpp line:138
	M("Teuchos").def("hashCode", (int (*)(const long &)) &Teuchos::hashCode<long>, "Get the hash code of a long\n\nC++: Teuchos::hashCode(const long &) --> int", pybind11::arg("x"));

	// Teuchos::hashCode(const std::string &) file:Teuchos_HashUtils.hpp line:147
	M("Teuchos").def("hashCode", (int (*)(const std::string &)) &Teuchos::hashCode<std::string>, "Get the hash code of a std::string\n\nC++: Teuchos::hashCode(const std::string &) --> int", pybind11::arg("x"));

}
