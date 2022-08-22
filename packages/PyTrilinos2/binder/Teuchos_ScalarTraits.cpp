#include <Teuchos_Array.hpp>
#include <Teuchos_ArrayViewDecl.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_Describable.hpp>
#include <Teuchos_ENull.hpp>
#include <Teuchos_FancyOStream.hpp>
#include <Teuchos_Include_Pybind11.hpp>
#include <Teuchos_LabeledObject.hpp>
#include <Teuchos_PerformanceMonitorBase.hpp>
#include <Teuchos_PtrDecl.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <Teuchos_RCPNode.hpp>
#include <Teuchos_ReductionOp.hpp>
#include <Teuchos_ScalarTraits.hpp>
#include <Teuchos_SerializationTraits.hpp>
#include <Teuchos_TableColumn.hpp>
#include <Teuchos_TableEntry.hpp>
#include <Teuchos_TableFormat.hpp>
#include <Teuchos_Time.hpp>
#include <Teuchos_VerbosityLevel.hpp>
#include <Teuchos_any.hpp>
#include <cwchar>
#include <ios>
#include <iterator>
#include <memory>
#include <ostream>
#include <sstream> // __str__
#include <streambuf>
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

// Teuchos::CommStatus file:Teuchos_Comm.hpp line:71
struct PyCallBack_Teuchos_CommStatus_int_t : public Teuchos::CommStatus<int> {
	using Teuchos::CommStatus<int>::CommStatus;

	int getSourceRank() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::CommStatus<int> *>(this), "getSourceRank");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<int>::value) {
				static pybind11::detail::override_caster_t<int> caster;
				return pybind11::detail::cast_ref<int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<int>(std::move(o));
		}
		pybind11::pybind11_fail("Tried to call pure virtual function \"CommStatus::getSourceRank\"");
	}
	int getTag() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::CommStatus<int> *>(this), "getTag");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<int>::value) {
				static pybind11::detail::override_caster_t<int> caster;
				return pybind11::detail::cast_ref<int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<int>(std::move(o));
		}
		pybind11::pybind11_fail("Tried to call pure virtual function \"CommStatus::getTag\"");
	}
};

// Teuchos::CommRequest file:Teuchos_Comm.hpp line:137
struct PyCallBack_Teuchos_CommRequest_int_t : public Teuchos::CommRequest<int> {
	using Teuchos::CommRequest<int>::CommRequest;

	bool isReady() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::CommRequest<int> *>(this), "isReady");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		pybind11::pybind11_fail("Tried to call pure virtual function \"CommRequest::isReady\"");
	}
	class Teuchos::RCP<class Teuchos::CommStatus<int> > wait() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::CommRequest<int> *>(this), "wait");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<class Teuchos::RCP<class Teuchos::CommStatus<int> >>::value) {
				static pybind11::detail::override_caster_t<class Teuchos::RCP<class Teuchos::CommStatus<int> >> caster;
				return pybind11::detail::cast_ref<class Teuchos::RCP<class Teuchos::CommStatus<int> >>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<class Teuchos::RCP<class Teuchos::CommStatus<int> >>(std::move(o));
		}
		pybind11::pybind11_fail("Tried to call pure virtual function \"CommRequest::wait\"");
	}
	std::string description() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::CommRequest<int> *>(this), "description");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<std::string>::value) {
				static pybind11::detail::override_caster_t<std::string> caster;
				return pybind11::detail::cast_ref<std::string>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<std::string>(std::move(o));
		}
		return Describable::description();
	}
	void describe(class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > & a0, const enum Teuchos::EVerbosityLevel a1) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::CommRequest<int> *>(this), "describe");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Describable::describe(a0, a1);
	}
	void setObjectLabel(const std::string & a0) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::CommRequest<int> *>(this), "setObjectLabel");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LabeledObject::setObjectLabel(a0);
	}
	std::string getObjectLabel() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::CommRequest<int> *>(this), "getObjectLabel");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<std::string>::value) {
				static pybind11::detail::override_caster_t<std::string> caster;
				return pybind11::detail::cast_ref<std::string>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<std::string>(std::move(o));
		}
		return LabeledObject::getObjectLabel();
	}
};

// Teuchos::TableEntry file:Teuchos_TableEntry.hpp line:69
struct PyCallBack_Teuchos_TableEntry : public Teuchos::TableEntry {
	using Teuchos::TableEntry::TableEntry;

	std::string toString() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::TableEntry *>(this), "toString");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<std::string>::value) {
				static pybind11::detail::override_caster_t<std::string> caster;
				return pybind11::detail::cast_ref<std::string>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<std::string>(std::move(o));
		}
		pybind11::pybind11_fail("Tried to call pure virtual function \"TableEntry::toString\"");
	}
	std::string toChoppedString(int a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::TableEntry *>(this), "toChoppedString");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<std::string>::value) {
				static pybind11::detail::override_caster_t<std::string> caster;
				return pybind11::detail::cast_ref<std::string>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<std::string>(std::move(o));
		}
		return TableEntry::toChoppedString(a0);
	}
};

// Teuchos::DoubleEntry file:Teuchos_TableEntry.hpp line:97
struct PyCallBack_Teuchos_DoubleEntry : public Teuchos::DoubleEntry {
	using Teuchos::DoubleEntry::DoubleEntry;

	std::string toString() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::DoubleEntry *>(this), "toString");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<std::string>::value) {
				static pybind11::detail::override_caster_t<std::string> caster;
				return pybind11::detail::cast_ref<std::string>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<std::string>(std::move(o));
		}
		return DoubleEntry::toString();
	}
	std::string toChoppedString(int a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::DoubleEntry *>(this), "toChoppedString");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<std::string>::value) {
				static pybind11::detail::override_caster_t<std::string> caster;
				return pybind11::detail::cast_ref<std::string>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<std::string>(std::move(o));
		}
		return TableEntry::toChoppedString(a0);
	}
};

// Teuchos::IntEntry file:Teuchos_TableEntry.hpp line:117
struct PyCallBack_Teuchos_IntEntry : public Teuchos::IntEntry {
	using Teuchos::IntEntry::IntEntry;

	std::string toString() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::IntEntry *>(this), "toString");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<std::string>::value) {
				static pybind11::detail::override_caster_t<std::string> caster;
				return pybind11::detail::cast_ref<std::string>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<std::string>(std::move(o));
		}
		return IntEntry::toString();
	}
	std::string toChoppedString(int a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::IntEntry *>(this), "toChoppedString");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<std::string>::value) {
				static pybind11::detail::override_caster_t<std::string> caster;
				return pybind11::detail::cast_ref<std::string>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<std::string>(std::move(o));
		}
		return TableEntry::toChoppedString(a0);
	}
};

// Teuchos::StringEntry file:Teuchos_TableEntry.hpp line:135
struct PyCallBack_Teuchos_StringEntry : public Teuchos::StringEntry {
	using Teuchos::StringEntry::StringEntry;

	std::string toString() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::StringEntry *>(this), "toString");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<std::string>::value) {
				static pybind11::detail::override_caster_t<std::string> caster;
				return pybind11::detail::cast_ref<std::string>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<std::string>(std::move(o));
		}
		return StringEntry::toString();
	}
	std::string toChoppedString(int a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::StringEntry *>(this), "toChoppedString");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<std::string>::value) {
				static pybind11::detail::override_caster_t<std::string> caster;
				return pybind11::detail::cast_ref<std::string>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<std::string>(std::move(o));
		}
		return TableEntry::toChoppedString(a0);
	}
};

// Teuchos::CompoundEntryWithParentheses file:Teuchos_TableEntry.hpp line:157
struct PyCallBack_Teuchos_CompoundEntryWithParentheses : public Teuchos::CompoundEntryWithParentheses {
	using Teuchos::CompoundEntryWithParentheses::CompoundEntryWithParentheses;

	std::string toString() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::CompoundEntryWithParentheses *>(this), "toString");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<std::string>::value) {
				static pybind11::detail::override_caster_t<std::string> caster;
				return pybind11::detail::cast_ref<std::string>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<std::string>(std::move(o));
		}
		return CompoundEntryWithParentheses::toString();
	}
	std::string toChoppedString(int a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::CompoundEntryWithParentheses *>(this), "toChoppedString");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<std::string>::value) {
				static pybind11::detail::override_caster_t<std::string> caster;
				return pybind11::detail::cast_ref<std::string>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<std::string>(std::move(o));
		}
		return TableEntry::toChoppedString(a0);
	}
};

void bind_Teuchos_ScalarTraits(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	// Teuchos::throwScalarTraitsNanInfError(const std::string &) file:Teuchos_ScalarTraits.hpp line:123
	M("Teuchos").def("throwScalarTraitsNanInfError", (void (*)(const std::string &)) &Teuchos::throwScalarTraitsNanInfError, "C++: Teuchos::throwScalarTraitsNanInfError(const std::string &) --> void", pybind11::arg("errMsg"));

	// Teuchos::generic_real_isnaninf(const float &) file:Teuchos_ScalarTraits.hpp line:127
	M("Teuchos").def("generic_real_isnaninf", (bool (*)(const float &)) &Teuchos::generic_real_isnaninf<float>, "C++: Teuchos::generic_real_isnaninf(const float &) --> bool", pybind11::arg("x"));

	// Teuchos::generic_real_isnaninf(const double &) file:Teuchos_ScalarTraits.hpp line:127
	M("Teuchos").def("generic_real_isnaninf", (bool (*)(const double &)) &Teuchos::generic_real_isnaninf<double>, "C++: Teuchos::generic_real_isnaninf(const double &) --> bool", pybind11::arg("x"));

	{ // Teuchos::DirectSerializationTraits file:Teuchos_SerializationTraits.hpp line:311
		pybind11::class_<Teuchos::DirectSerializationTraits<int,unsigned long>, Teuchos::RCP<Teuchos::DirectSerializationTraits<int,unsigned long>>> cl(M("Teuchos"), "DirectSerializationTraits_int_unsigned_long_t", "");
		cl.def( pybind11::init( [](){ return new Teuchos::DirectSerializationTraits<int,unsigned long>(); } ) );
		cl.def( pybind11::init( [](Teuchos::DirectSerializationTraits<int,unsigned long> const &o){ return new Teuchos::DirectSerializationTraits<int,unsigned long>(o); } ) );
		cl.def_static("fromCountToDirectBytes", (int (*)(const int)) &Teuchos::DirectSerializationTraits<int, unsigned long>::fromCountToDirectBytes, "C++: Teuchos::DirectSerializationTraits<int, unsigned long>::fromCountToDirectBytes(const int) --> int", pybind11::arg("count"));
		cl.def_static("convertToCharPtr", (char * (*)(unsigned long *)) &Teuchos::DirectSerializationTraits<int, unsigned long>::convertToCharPtr, "C++: Teuchos::DirectSerializationTraits<int, unsigned long>::convertToCharPtr(unsigned long *) --> char *", pybind11::return_value_policy::automatic, pybind11::arg("ptr"));
		cl.def_static("convertToCharPtr", (const char * (*)(const unsigned long *)) &Teuchos::DirectSerializationTraits<int, unsigned long>::convertToCharPtr, "C++: Teuchos::DirectSerializationTraits<int, unsigned long>::convertToCharPtr(const unsigned long *) --> const char *", pybind11::return_value_policy::automatic, pybind11::arg("ptr"));
		cl.def_static("fromDirectBytesToCount", (int (*)(const int)) &Teuchos::DirectSerializationTraits<int, unsigned long>::fromDirectBytesToCount, "C++: Teuchos::DirectSerializationTraits<int, unsigned long>::fromDirectBytesToCount(const int) --> int", pybind11::arg("count"));
		cl.def_static("convertFromCharPtr", (unsigned long * (*)(char *)) &Teuchos::DirectSerializationTraits<int, unsigned long>::convertFromCharPtr, "C++: Teuchos::DirectSerializationTraits<int, unsigned long>::convertFromCharPtr(char *) --> unsigned long *", pybind11::return_value_policy::automatic, pybind11::arg("ptr"));
		cl.def_static("convertFromCharPtr", (const unsigned long * (*)(const char *)) &Teuchos::DirectSerializationTraits<int, unsigned long>::convertFromCharPtr, "C++: Teuchos::DirectSerializationTraits<int, unsigned long>::convertFromCharPtr(const char *) --> const unsigned long *", pybind11::return_value_policy::automatic, pybind11::arg("ptr"));
	}
	{ // Teuchos::DirectSerializationTraits file:Teuchos_SerializationTraits.hpp line:311
		pybind11::class_<Teuchos::DirectSerializationTraits<int,long long>, Teuchos::RCP<Teuchos::DirectSerializationTraits<int,long long>>> cl(M("Teuchos"), "DirectSerializationTraits_int_long_long_t", "");
		cl.def( pybind11::init( [](){ return new Teuchos::DirectSerializationTraits<int,long long>(); } ) );
		cl.def( pybind11::init( [](Teuchos::DirectSerializationTraits<int,long long> const &o){ return new Teuchos::DirectSerializationTraits<int,long long>(o); } ) );
		cl.def_static("fromCountToDirectBytes", (int (*)(const int)) &Teuchos::DirectSerializationTraits<int, long long>::fromCountToDirectBytes, "C++: Teuchos::DirectSerializationTraits<int, long long>::fromCountToDirectBytes(const int) --> int", pybind11::arg("count"));
		cl.def_static("convertToCharPtr", (char * (*)(long long *)) &Teuchos::DirectSerializationTraits<int, long long>::convertToCharPtr, "C++: Teuchos::DirectSerializationTraits<int, long long>::convertToCharPtr(long long *) --> char *", pybind11::return_value_policy::automatic, pybind11::arg("ptr"));
		cl.def_static("convertToCharPtr", (const char * (*)(const long long *)) &Teuchos::DirectSerializationTraits<int, long long>::convertToCharPtr, "C++: Teuchos::DirectSerializationTraits<int, long long>::convertToCharPtr(const long long *) --> const char *", pybind11::return_value_policy::automatic, pybind11::arg("ptr"));
		cl.def_static("fromDirectBytesToCount", (int (*)(const int)) &Teuchos::DirectSerializationTraits<int, long long>::fromDirectBytesToCount, "C++: Teuchos::DirectSerializationTraits<int, long long>::fromDirectBytesToCount(const int) --> int", pybind11::arg("count"));
		cl.def_static("convertFromCharPtr", (long long * (*)(char *)) &Teuchos::DirectSerializationTraits<int, long long>::convertFromCharPtr, "C++: Teuchos::DirectSerializationTraits<int, long long>::convertFromCharPtr(char *) --> long long *", pybind11::return_value_policy::automatic, pybind11::arg("ptr"));
		cl.def_static("convertFromCharPtr", (const long long * (*)(const char *)) &Teuchos::DirectSerializationTraits<int, long long>::convertFromCharPtr, "C++: Teuchos::DirectSerializationTraits<int, long long>::convertFromCharPtr(const char *) --> const long long *", pybind11::return_value_policy::automatic, pybind11::arg("ptr"));
	}
	{ // Teuchos::DirectSerializationTraits file:Teuchos_SerializationTraits.hpp line:311
		pybind11::class_<Teuchos::DirectSerializationTraits<int,char>, Teuchos::RCP<Teuchos::DirectSerializationTraits<int,char>>> cl(M("Teuchos"), "DirectSerializationTraits_int_char_t", "");
		cl.def( pybind11::init( [](){ return new Teuchos::DirectSerializationTraits<int,char>(); } ) );
		cl.def( pybind11::init( [](Teuchos::DirectSerializationTraits<int,char> const &o){ return new Teuchos::DirectSerializationTraits<int,char>(o); } ) );
		cl.def_static("fromCountToDirectBytes", (int (*)(const int)) &Teuchos::DirectSerializationTraits<int, char>::fromCountToDirectBytes, "C++: Teuchos::DirectSerializationTraits<int, char>::fromCountToDirectBytes(const int) --> int", pybind11::arg("count"));
		cl.def_static("convertToCharPtr", (char * (*)(char *)) &Teuchos::DirectSerializationTraits<int, char>::convertToCharPtr, "C++: Teuchos::DirectSerializationTraits<int, char>::convertToCharPtr(char *) --> char *", pybind11::return_value_policy::automatic, pybind11::arg("ptr"));
		cl.def_static("convertToCharPtr", (const char * (*)(const char *)) &Teuchos::DirectSerializationTraits<int, char>::convertToCharPtr, "C++: Teuchos::DirectSerializationTraits<int, char>::convertToCharPtr(const char *) --> const char *", pybind11::return_value_policy::automatic, pybind11::arg("ptr"));
		cl.def_static("fromDirectBytesToCount", (int (*)(const int)) &Teuchos::DirectSerializationTraits<int, char>::fromDirectBytesToCount, "C++: Teuchos::DirectSerializationTraits<int, char>::fromDirectBytesToCount(const int) --> int", pybind11::arg("count"));
		cl.def_static("convertFromCharPtr", (char * (*)(char *)) &Teuchos::DirectSerializationTraits<int, char>::convertFromCharPtr, "C++: Teuchos::DirectSerializationTraits<int, char>::convertFromCharPtr(char *) --> char *", pybind11::return_value_policy::automatic, pybind11::arg("ptr"));
		cl.def_static("convertFromCharPtr", (const char * (*)(const char *)) &Teuchos::DirectSerializationTraits<int, char>::convertFromCharPtr, "C++: Teuchos::DirectSerializationTraits<int, char>::convertFromCharPtr(const char *) --> const char *", pybind11::return_value_policy::automatic, pybind11::arg("ptr"));
	}
	{ // Teuchos::ValueTypeReductionOp file:Teuchos_ReductionOp.hpp line:60
		pybind11::class_<Teuchos::ValueTypeReductionOp<int,long long>, Teuchos::RCP<Teuchos::ValueTypeReductionOp<int,long long>>> cl(M("Teuchos"), "ValueTypeReductionOp_int_long_long_t", "");
		cl.def("assign", (class Teuchos::ValueTypeReductionOp<int, long long> & (Teuchos::ValueTypeReductionOp<int,long long>::*)(const class Teuchos::ValueTypeReductionOp<int, long long> &)) &Teuchos::ValueTypeReductionOp<int, long long>::operator=, "C++: Teuchos::ValueTypeReductionOp<int, long long>::operator=(const class Teuchos::ValueTypeReductionOp<int, long long> &) --> class Teuchos::ValueTypeReductionOp<int, long long> &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // Teuchos::CommStatus file:Teuchos_Comm.hpp line:71
		pybind11::class_<Teuchos::CommStatus<int>, Teuchos::RCP<Teuchos::CommStatus<int>>, PyCallBack_Teuchos_CommStatus_int_t> cl(M("Teuchos"), "CommStatus_int_t", "");
		cl.def(pybind11::init<PyCallBack_Teuchos_CommStatus_int_t const &>());
		cl.def( pybind11::init( [](){ return new PyCallBack_Teuchos_CommStatus_int_t(); } ) );
		cl.def("getSourceRank", (int (Teuchos::CommStatus<int>::*)()) &Teuchos::CommStatus<int>::getSourceRank, "C++: Teuchos::CommStatus<int>::getSourceRank() --> int");
		cl.def("getTag", (int (Teuchos::CommStatus<int>::*)()) &Teuchos::CommStatus<int>::getTag, "C++: Teuchos::CommStatus<int>::getTag() --> int");
		cl.def("assign", (class Teuchos::CommStatus<int> & (Teuchos::CommStatus<int>::*)(const class Teuchos::CommStatus<int> &)) &Teuchos::CommStatus<int>::operator=, "C++: Teuchos::CommStatus<int>::operator=(const class Teuchos::CommStatus<int> &) --> class Teuchos::CommStatus<int> &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // Teuchos::Comm file:Teuchos_Comm.hpp line:314
		pybind11::class_<Teuchos::Comm<int>, Teuchos::RCP<Teuchos::Comm<int>>> cl(M("Teuchos"), "Comm_int_t", "");
		cl.def("getTag", (int (Teuchos::Comm<int>::*)() const) &Teuchos::Comm<int>::getTag, "C++: Teuchos::Comm<int>::getTag() const --> int");
		cl.def("getRank", (int (Teuchos::Comm<int>::*)() const) &Teuchos::Comm<int>::getRank, "C++: Teuchos::Comm<int>::getRank() const --> int");
		cl.def("getSize", (int (Teuchos::Comm<int>::*)() const) &Teuchos::Comm<int>::getSize, "C++: Teuchos::Comm<int>::getSize() const --> int");
		cl.def("barrier", (void (Teuchos::Comm<int>::*)() const) &Teuchos::Comm<int>::barrier, "C++: Teuchos::Comm<int>::barrier() const --> void");
		cl.def("readySend", (void (Teuchos::Comm<int>::*)(const class Teuchos::ArrayView<const char> &, const int) const) &Teuchos::Comm<int>::readySend, "C++: Teuchos::Comm<int>::readySend(const class Teuchos::ArrayView<const char> &, const int) const --> void", pybind11::arg("sendBuffer"), pybind11::arg("destRank"));
		cl.def("isend", (class Teuchos::RCP<class Teuchos::CommRequest<int> > (Teuchos::Comm<int>::*)(const class Teuchos::ArrayView<const char> &, const int) const) &Teuchos::Comm<int>::isend, "C++: Teuchos::Comm<int>::isend(const class Teuchos::ArrayView<const char> &, const int) const --> class Teuchos::RCP<class Teuchos::CommRequest<int> >", pybind11::arg("sendBuffer"), pybind11::arg("destRank"));
		cl.def("isend", (class Teuchos::RCP<class Teuchos::CommRequest<int> > (Teuchos::Comm<int>::*)(const class Teuchos::ArrayView<const char> &, const int, const int) const) &Teuchos::Comm<int>::isend, "C++: Teuchos::Comm<int>::isend(const class Teuchos::ArrayView<const char> &, const int, const int) const --> class Teuchos::RCP<class Teuchos::CommRequest<int> >", pybind11::arg("sendBuffer"), pybind11::arg("destRank"), pybind11::arg("tag"));
		cl.def("ireceive", (class Teuchos::RCP<class Teuchos::CommRequest<int> > (Teuchos::Comm<int>::*)(const class Teuchos::ArrayView<char> &, const int) const) &Teuchos::Comm<int>::ireceive, "C++: Teuchos::Comm<int>::ireceive(const class Teuchos::ArrayView<char> &, const int) const --> class Teuchos::RCP<class Teuchos::CommRequest<int> >", pybind11::arg("recvBuffer"), pybind11::arg("sourceRank"));
		cl.def("ireceive", (class Teuchos::RCP<class Teuchos::CommRequest<int> > (Teuchos::Comm<int>::*)(const class Teuchos::ArrayView<char> &, const int, const int) const) &Teuchos::Comm<int>::ireceive, "C++: Teuchos::Comm<int>::ireceive(const class Teuchos::ArrayView<char> &, const int, const int) const --> class Teuchos::RCP<class Teuchos::CommRequest<int> >", pybind11::arg("recvBuffer"), pybind11::arg("sourceRank"), pybind11::arg("tag"));
		cl.def("duplicate", (class Teuchos::RCP<class Teuchos::Comm<int> > (Teuchos::Comm<int>::*)() const) &Teuchos::Comm<int>::duplicate, "C++: Teuchos::Comm<int>::duplicate() const --> class Teuchos::RCP<class Teuchos::Comm<int> >");
		cl.def("split", (class Teuchos::RCP<class Teuchos::Comm<int> > (Teuchos::Comm<int>::*)(const int, const int) const) &Teuchos::Comm<int>::split, "C++: Teuchos::Comm<int>::split(const int, const int) const --> class Teuchos::RCP<class Teuchos::Comm<int> >", pybind11::arg("color"), pybind11::arg("key"));
		cl.def("createSubcommunicator", (class Teuchos::RCP<class Teuchos::Comm<int> > (Teuchos::Comm<int>::*)(const class Teuchos::ArrayView<const int> &) const) &Teuchos::Comm<int>::createSubcommunicator, "C++: Teuchos::Comm<int>::createSubcommunicator(const class Teuchos::ArrayView<const int> &) const --> class Teuchos::RCP<class Teuchos::Comm<int> >", pybind11::arg("ranks"));
		cl.def("assign", (class Teuchos::Comm<int> & (Teuchos::Comm<int>::*)(const class Teuchos::Comm<int> &)) &Teuchos::Comm<int>::operator=, "C++: Teuchos::Comm<int>::operator=(const class Teuchos::Comm<int> &) --> class Teuchos::Comm<int> &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // Teuchos::CommRequest file:Teuchos_Comm.hpp line:137
		pybind11::class_<Teuchos::CommRequest<int>, Teuchos::RCP<Teuchos::CommRequest<int>>, PyCallBack_Teuchos_CommRequest_int_t> cl(M("Teuchos"), "CommRequest_int_t", "");
		cl.def( pybind11::init( [](){ return new PyCallBack_Teuchos_CommRequest_int_t(); } ) );
		cl.def(pybind11::init<PyCallBack_Teuchos_CommRequest_int_t const &>());
		cl.def("isReady", (bool (Teuchos::CommRequest<int>::*)()) &Teuchos::CommRequest<int>::isReady, "C++: Teuchos::CommRequest<int>::isReady() --> bool");
		cl.def("wait", (class Teuchos::RCP<class Teuchos::CommStatus<int> > (Teuchos::CommRequest<int>::*)()) &Teuchos::CommRequest<int>::wait, "C++: Teuchos::CommRequest<int>::wait() --> class Teuchos::RCP<class Teuchos::CommStatus<int> >");
		cl.def("assign", (class Teuchos::CommRequest<int> & (Teuchos::CommRequest<int>::*)(const class Teuchos::CommRequest<int> &)) &Teuchos::CommRequest<int>::operator=, "C++: Teuchos::CommRequest<int>::operator=(const class Teuchos::CommRequest<int> &) --> class Teuchos::CommRequest<int> &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // Teuchos::TableEntry file:Teuchos_TableEntry.hpp line:69
		pybind11::class_<Teuchos::TableEntry, Teuchos::RCP<Teuchos::TableEntry>, PyCallBack_Teuchos_TableEntry> cl(M("Teuchos"), "TableEntry", "An entry, perhaps compound, to be written into a table.\n\n KL 30 Apr 2006 -- initial design. Can you say overengineering??\n The complexity is to support a nice interface for pair entries\n such as time/numCalls.");
		cl.def( pybind11::init( [](){ return new PyCallBack_Teuchos_TableEntry(); } ) );
		cl.def(pybind11::init<PyCallBack_Teuchos_TableEntry const &>());
		cl.def("toString", (std::string (Teuchos::TableEntry::*)() const) &Teuchos::TableEntry::toString, "Return a std::string representation of this entry \n\nC++: Teuchos::TableEntry::toString() const --> std::string");
		cl.def("toChoppedString", (std::string (Teuchos::TableEntry::*)(int) const) &Teuchos::TableEntry::toChoppedString, "Return a std::string representation of this entry,\n truncated if necessary to fit within the given column width.\n\n \n [in] the maximum width of the std::string form. Larger\n strings must be truncated in a subclass-dependent way.\n \n\n the std::string, truncated if necessary\n\nC++: Teuchos::TableEntry::toChoppedString(int) const --> std::string", pybind11::arg("maxWidth"));
		cl.def("assign", (class Teuchos::TableEntry & (Teuchos::TableEntry::*)(const class Teuchos::TableEntry &)) &Teuchos::TableEntry::operator=, "C++: Teuchos::TableEntry::operator=(const class Teuchos::TableEntry &) --> class Teuchos::TableEntry &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // Teuchos::DoubleEntry file:Teuchos_TableEntry.hpp line:97
		pybind11::class_<Teuchos::DoubleEntry, Teuchos::RCP<Teuchos::DoubleEntry>, PyCallBack_Teuchos_DoubleEntry, Teuchos::TableEntry> cl(M("Teuchos"), "DoubleEntry", "A table entry that is a simple double-precision number");
		cl.def( pybind11::init<const double &, int, const enum std::_Ios_Fmtflags &>(), pybind11::arg("value"), pybind11::arg("precision"), pybind11::arg("flags") );

		cl.def( pybind11::init( [](PyCallBack_Teuchos_DoubleEntry const &o){ return new PyCallBack_Teuchos_DoubleEntry(o); } ) );
		cl.def( pybind11::init( [](Teuchos::DoubleEntry const &o){ return new Teuchos::DoubleEntry(o); } ) );
		cl.def("toString", (std::string (Teuchos::DoubleEntry::*)() const) &Teuchos::DoubleEntry::toString, "Write the specified entry to a std::string \n\nC++: Teuchos::DoubleEntry::toString() const --> std::string");
		cl.def("assign", (class Teuchos::DoubleEntry & (Teuchos::DoubleEntry::*)(const class Teuchos::DoubleEntry &)) &Teuchos::DoubleEntry::operator=, "C++: Teuchos::DoubleEntry::operator=(const class Teuchos::DoubleEntry &) --> class Teuchos::DoubleEntry &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // Teuchos::IntEntry file:Teuchos_TableEntry.hpp line:117
		pybind11::class_<Teuchos::IntEntry, Teuchos::RCP<Teuchos::IntEntry>, PyCallBack_Teuchos_IntEntry, Teuchos::TableEntry> cl(M("Teuchos"), "IntEntry", "A table entry that is a simple integer");
		cl.def( pybind11::init<int, const enum std::_Ios_Fmtflags &>(), pybind11::arg("value"), pybind11::arg("flags") );

		cl.def( pybind11::init( [](PyCallBack_Teuchos_IntEntry const &o){ return new PyCallBack_Teuchos_IntEntry(o); } ) );
		cl.def( pybind11::init( [](Teuchos::IntEntry const &o){ return new Teuchos::IntEntry(o); } ) );
		cl.def("toString", (std::string (Teuchos::IntEntry::*)() const) &Teuchos::IntEntry::toString, "Write the specified entry to a std::string \n\nC++: Teuchos::IntEntry::toString() const --> std::string");
		cl.def("assign", (class Teuchos::IntEntry & (Teuchos::IntEntry::*)(const class Teuchos::IntEntry &)) &Teuchos::IntEntry::operator=, "C++: Teuchos::IntEntry::operator=(const class Teuchos::IntEntry &) --> class Teuchos::IntEntry &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // Teuchos::StringEntry file:Teuchos_TableEntry.hpp line:135
		pybind11::class_<Teuchos::StringEntry, Teuchos::RCP<Teuchos::StringEntry>, PyCallBack_Teuchos_StringEntry, Teuchos::TableEntry> cl(M("Teuchos"), "StringEntry", "A table entry that is a simple std::string");
		cl.def( pybind11::init<std::string>(), pybind11::arg("value") );

		cl.def( pybind11::init( [](PyCallBack_Teuchos_StringEntry const &o){ return new PyCallBack_Teuchos_StringEntry(o); } ) );
		cl.def( pybind11::init( [](Teuchos::StringEntry const &o){ return new Teuchos::StringEntry(o); } ) );
		cl.def("toString", (std::string (Teuchos::StringEntry::*)() const) &Teuchos::StringEntry::toString, "Write the specified entry to a std::string \n\nC++: Teuchos::StringEntry::toString() const --> std::string");
		cl.def("assign", (class Teuchos::StringEntry & (Teuchos::StringEntry::*)(const class Teuchos::StringEntry &)) &Teuchos::StringEntry::operator=, "C++: Teuchos::StringEntry::operator=(const class Teuchos::StringEntry &) --> class Teuchos::StringEntry &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // Teuchos::CompoundEntryWithParentheses file:Teuchos_TableEntry.hpp line:157
		pybind11::class_<Teuchos::CompoundEntryWithParentheses, Teuchos::RCP<Teuchos::CompoundEntryWithParentheses>, PyCallBack_Teuchos_CompoundEntryWithParentheses, Teuchos::TableEntry> cl(M("Teuchos"), "CompoundEntryWithParentheses", "An entry containing two subentries, with the second\n to be written in parentheses after the first. For example,\n \n\n\n\n The two subentries can be any type of data, each represented\n with a TableEntry derived type.");
		cl.def( pybind11::init( [](const class Teuchos::RCP<class Teuchos::TableEntry> & a0, const class Teuchos::RCP<class Teuchos::TableEntry> & a1){ return new Teuchos::CompoundEntryWithParentheses(a0, a1); }, [](const class Teuchos::RCP<class Teuchos::TableEntry> & a0, const class Teuchos::RCP<class Teuchos::TableEntry> & a1){ return new PyCallBack_Teuchos_CompoundEntryWithParentheses(a0, a1); } ), "doc");
		cl.def( pybind11::init<const class Teuchos::RCP<class Teuchos::TableEntry> &, const class Teuchos::RCP<class Teuchos::TableEntry> &, bool>(), pybind11::arg("first"), pybind11::arg("second"), pybind11::arg("spaceBeforeParens") );

		cl.def( pybind11::init( [](PyCallBack_Teuchos_CompoundEntryWithParentheses const &o){ return new PyCallBack_Teuchos_CompoundEntryWithParentheses(o); } ) );
		cl.def( pybind11::init( [](Teuchos::CompoundEntryWithParentheses const &o){ return new Teuchos::CompoundEntryWithParentheses(o); } ) );
		cl.def("toString", (std::string (Teuchos::CompoundEntryWithParentheses::*)() const) &Teuchos::CompoundEntryWithParentheses::toString, "Write the specified entry to a std::string \n\nC++: Teuchos::CompoundEntryWithParentheses::toString() const --> std::string");
		cl.def("assign", (class Teuchos::CompoundEntryWithParentheses & (Teuchos::CompoundEntryWithParentheses::*)(const class Teuchos::CompoundEntryWithParentheses &)) &Teuchos::CompoundEntryWithParentheses::operator=, "C++: Teuchos::CompoundEntryWithParentheses::operator=(const class Teuchos::CompoundEntryWithParentheses &) --> class Teuchos::CompoundEntryWithParentheses &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // Teuchos::TableColumn file:Teuchos_TableColumn.hpp line:61
		pybind11::class_<Teuchos::TableColumn, Teuchos::RCP<Teuchos::TableColumn>> cl(M("Teuchos"), "TableColumn", "KL 30 Apr 2006 -- initial design.");
		cl.def( pybind11::init( [](){ return new Teuchos::TableColumn(); } ) );
		cl.def( pybind11::init<const class Teuchos::Array<std::string > &>(), pybind11::arg("vals") );

		cl.def( pybind11::init<const class Teuchos::Array<double> &, int, const enum std::_Ios_Fmtflags &>(), pybind11::arg("vals"), pybind11::arg("precision"), pybind11::arg("flags") );

		cl.def( pybind11::init<const class Teuchos::Array<double> &, const class Teuchos::Array<double> &, int, const enum std::_Ios_Fmtflags &, bool>(), pybind11::arg("first"), pybind11::arg("second"), pybind11::arg("precision"), pybind11::arg("flags"), pybind11::arg("spaceBeforeParentheses") );

		cl.def( pybind11::init( [](Teuchos::TableColumn const &o){ return new Teuchos::TableColumn(o); } ) );
		cl.def("numRows", (int (Teuchos::TableColumn::*)() const) &Teuchos::TableColumn::numRows, "C++: Teuchos::TableColumn::numRows() const --> int");
		cl.def("addEntry", (void (Teuchos::TableColumn::*)(const class Teuchos::RCP<class Teuchos::TableEntry> &)) &Teuchos::TableColumn::addEntry, "C++: Teuchos::TableColumn::addEntry(const class Teuchos::RCP<class Teuchos::TableEntry> &) --> void", pybind11::arg("entry"));
		cl.def("entry", (const class Teuchos::RCP<class Teuchos::TableEntry> & (Teuchos::TableColumn::*)(int) const) &Teuchos::TableColumn::entry, "C++: Teuchos::TableColumn::entry(int) const --> const class Teuchos::RCP<class Teuchos::TableEntry> &", pybind11::return_value_policy::automatic, pybind11::arg("i"));
	}
	{ // Teuchos::TableFormat file:Teuchos_TableFormat.hpp line:65
		pybind11::class_<Teuchos::TableFormat, Teuchos::RCP<Teuchos::TableFormat>> cl(M("Teuchos"), "TableFormat", "Encapsulation of formatting specifications for writing data\n in a clean tabular form.\n\n Note: it is left to the programmer to avoid invalid settings such as\n negative column spaces, zero page widths, and other such potentially\n bad things.\n\n KL 30 Apr 2006 -- initial design.");
		cl.def( pybind11::init( [](){ return new Teuchos::TableFormat(); } ) );
		cl.def( pybind11::init( [](Teuchos::TableFormat const &o){ return new Teuchos::TableFormat(o); } ) );
		cl.def("pageWidth", (int (Teuchos::TableFormat::*)() const) &Teuchos::TableFormat::pageWidth, "Get the maximum number of characters per line.\n Default is 80. \n\nC++: Teuchos::TableFormat::pageWidth() const --> int");
		cl.def("precision", (int (Teuchos::TableFormat::*)() const) &Teuchos::TableFormat::precision, "Get the precision for writing doubles.\n Default is 4. \n\nC++: Teuchos::TableFormat::precision() const --> int");
		cl.def("columnSpacing", (int (Teuchos::TableFormat::*)() const) &Teuchos::TableFormat::columnSpacing, "Get the number of characters to be left as blank\n spaces in each column. Default is 4. \n\nC++: Teuchos::TableFormat::columnSpacing() const --> int");
		cl.def("setPageWidth", (void (Teuchos::TableFormat::*)(int) const) &Teuchos::TableFormat::setPageWidth, "Set the number of characters on a line.\n This quantity can be updated within the const\n method writeWholeTables() \n\nC++: Teuchos::TableFormat::setPageWidth(int) const --> void", pybind11::arg("pw"));
		cl.def("setPrecision", (void (Teuchos::TableFormat::*)(int)) &Teuchos::TableFormat::setPrecision, "Set the precision for writing doubles \n\nC++: Teuchos::TableFormat::setPrecision(int) --> void", pybind11::arg("p"));
		cl.def("setColumnSpacing", (void (Teuchos::TableFormat::*)(int)) &Teuchos::TableFormat::setColumnSpacing, "Set the number of characters to be left as blank spaces in each column \n\nC++: Teuchos::TableFormat::setColumnSpacing(int) --> void", pybind11::arg("columnSpacing_in"));
		cl.def("setRowsBetweenLines", (void (Teuchos::TableFormat::*)(int)) &Teuchos::TableFormat::setRowsBetweenLines, "Set the interval at which a horizontal line will be written between\n rows.\n\n  lineInterval [in] the number of rows between each horizontal line\n\nC++: Teuchos::TableFormat::setRowsBetweenLines(int) --> void", pybind11::arg("lineInterval"));
		cl.def("thinline", (std::string (Teuchos::TableFormat::*)() const) &Teuchos::TableFormat::thinline, "Return a horizontal line in dashes \"----\"\n the width of the page.\n\n Originally called hbar, but changed to avoid\n possible confusion for physicists expecting hbar() to return\n \n\n :-).  \n\nC++: Teuchos::TableFormat::thinline() const --> std::string");
		cl.def("thickline", (std::string (Teuchos::TableFormat::*)() const) &Teuchos::TableFormat::thickline, "Return a thick horizontal line in equal signs \"====\" the\n width of the page \n\nC++: Teuchos::TableFormat::thickline() const --> std::string");
		cl.def("blanks", (std::string (Teuchos::TableFormat::*)(int) const) &Teuchos::TableFormat::blanks, "Return a std::string full of blanks up to the requested size \n\nC++: Teuchos::TableFormat::blanks(int) const --> std::string", pybind11::arg("size"));
		cl.def("computeRequiredColumnWidth", (int (Teuchos::TableFormat::*)(const std::string &, const class Teuchos::TableColumn &) const) &Teuchos::TableFormat::computeRequiredColumnWidth, "Computes the column width required to write all values\n to the required precision.\n\n \n [in] the title of the column\n \n\n [in] the column data\n\n Postcondition: colString.size()==values.size()\n\nC++: Teuchos::TableFormat::computeRequiredColumnWidth(const std::string &, const class Teuchos::TableColumn &) const --> int", pybind11::arg("name"), pybind11::arg("column"));
		cl.def("setColumnWidths", (void (Teuchos::TableFormat::*)(const class Teuchos::Array<int> &)) &Teuchos::TableFormat::setColumnWidths, "Set the column widths to be used for subsequent rows \n\nC++: Teuchos::TableFormat::setColumnWidths(const class Teuchos::Array<int> &) --> void", pybind11::arg("colWidths"));
		cl.def("assign", (class Teuchos::TableFormat & (Teuchos::TableFormat::*)(const class Teuchos::TableFormat &)) &Teuchos::TableFormat::operator=, "C++: Teuchos::TableFormat::operator=(const class Teuchos::TableFormat &) --> class Teuchos::TableFormat &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	// Teuchos::ECounterSetOp file:Teuchos_PerformanceMonitorBase.hpp line:63
	pybind11::enum_<Teuchos::ECounterSetOp>(M("Teuchos"), "ECounterSetOp", pybind11::arithmetic(), "Set operation type for  to perform.\n\n The  function merges sets of counter names\n over all MPI processes in a communicator.  Different MPI\n processes may have created different sets of counters.  This\n enum allows the caller to specify how mergeCounterNames() picks\n the global set of timers.")
		.value("Intersection", Teuchos::Intersection)
		.value("Union", Teuchos::Union)
		.export_values();

;

	// Teuchos::mergeCounterNames(const class Teuchos::Comm<int> &, const class Teuchos::Array<std::string > &, class Teuchos::Array<std::string > &, const enum Teuchos::ECounterSetOp) file:Teuchos_PerformanceMonitorBase.hpp line:90
	M("Teuchos").def("mergeCounterNames", (void (*)(const class Teuchos::Comm<int> &, const class Teuchos::Array<std::string > &, class Teuchos::Array<std::string > &, const enum Teuchos::ECounterSetOp)) &Teuchos::mergeCounterNames, "Merge counter names over all processors.\n\n Different MPI processes may have created different sets of\n counters.  Use this function to reconcile the sets among\n processes, either by computing their intersection or their\n union.  This is done using a reduction to MPI Rank 0 (relative\n to the given communicator) and a broadcast to all processes\n participating in the communicator.  We use a\n reduce-and-broadcast rather than just a reduction, so that all\n participating processes can use the resulting list of global\n names as lookup keys for computing global statistics.\n\n \n [in] Communicator over which to merge.\n\n \n [in] The calling MPI process' list of (local)\n   counter names.\n\n \n [out] On output, on each MPI process: the\n   results of merging the counter names.\n\n \n [in] If Intersection, globalNames on output\n   contains the intersection of all sets of counter names.  If\n   Union, globalNames on output contains the union of all sets of\n   counter names.\n\nC++: Teuchos::mergeCounterNames(const class Teuchos::Comm<int> &, const class Teuchos::Array<std::string > &, class Teuchos::Array<std::string > &, const enum Teuchos::ECounterSetOp) --> void", pybind11::arg("comm"), pybind11::arg("localNames"), pybind11::arg("globalNames"), pybind11::arg("setOp"));

	// Teuchos::unsortedMergePair(const class Teuchos::Array<std::string > &, class Teuchos::Array<std::string > &, const enum Teuchos::ECounterSetOp) file:Teuchos_PerformanceMonitorBase.hpp line:106
	M("Teuchos").def("unsortedMergePair", (void (*)(const class Teuchos::Array<std::string > &, class Teuchos::Array<std::string > &, const enum Teuchos::ECounterSetOp)) &Teuchos::unsortedMergePair, "merge for unsorted lists.  New entries are at the bottom of the list\n \n\n - The calling MPI process' list of (local)\n counter names.\n \n\n - Global list of names\n \n\n If Intersection, globalNames on output\n   contains the intersection of all sets of counter names.  If\n   Union, globalNames on output contains the union of all sets of\n   counter names.\n\nC++: Teuchos::unsortedMergePair(const class Teuchos::Array<std::string > &, class Teuchos::Array<std::string > &, const enum Teuchos::ECounterSetOp) --> void", pybind11::arg("localNames"), pybind11::arg("globalNames"), pybind11::arg("setOp"));

	{ // Teuchos::PerformanceMonitorBase file:Teuchos_PerformanceMonitorBase.hpp line:154
		pybind11::class_<Teuchos::PerformanceMonitorBase<Teuchos::Time>, Teuchos::RCP<Teuchos::PerformanceMonitorBase<Teuchos::Time>>> cl(M("Teuchos"), "PerformanceMonitorBase_Teuchos_Time_t", "");
		cl.def( pybind11::init( [](class Teuchos::Time & a0){ return new Teuchos::PerformanceMonitorBase<Teuchos::Time>(a0); } ), "doc" , pybind11::arg("counter_in"));
		cl.def( pybind11::init<class Teuchos::Time &, bool>(), pybind11::arg("counter_in"), pybind11::arg("reset") );

		cl.def( pybind11::init( [](Teuchos::PerformanceMonitorBase<Teuchos::Time> const &o){ return new Teuchos::PerformanceMonitorBase<Teuchos::Time>(o); } ) );
		cl.def_static("getNewCounter", (class Teuchos::RCP<class Teuchos::Time> (*)(const std::string &)) &Teuchos::PerformanceMonitorBase<Teuchos::Time>::getNewCounter, "C++: Teuchos::PerformanceMonitorBase<Teuchos::Time>::getNewCounter(const std::string &) --> class Teuchos::RCP<class Teuchos::Time>", pybind11::arg("name"));
		cl.def_static("format", (class Teuchos::TableFormat & (*)()) &Teuchos::PerformanceMonitorBase<Teuchos::Time>::format, "C++: Teuchos::PerformanceMonitorBase<Teuchos::Time>::format() --> class Teuchos::TableFormat &", pybind11::return_value_policy::automatic);
		cl.def_static("lookupCounter", (class Teuchos::RCP<class Teuchos::Time> (*)(const std::string &)) &Teuchos::PerformanceMonitorBase<Teuchos::Time>::lookupCounter, "C++: Teuchos::PerformanceMonitorBase<Teuchos::Time>::lookupCounter(const std::string &) --> class Teuchos::RCP<class Teuchos::Time>", pybind11::arg("name"));
		cl.def_static("clearCounters", (void (*)()) &Teuchos::PerformanceMonitorBase<Teuchos::Time>::clearCounters, "C++: Teuchos::PerformanceMonitorBase<Teuchos::Time>::clearCounters() --> void");
		cl.def_static("clearCounter", (void (*)(const std::string &)) &Teuchos::PerformanceMonitorBase<Teuchos::Time>::clearCounter, "C++: Teuchos::PerformanceMonitorBase<Teuchos::Time>::clearCounter(const std::string &) --> void", pybind11::arg("name"));
	}
}
