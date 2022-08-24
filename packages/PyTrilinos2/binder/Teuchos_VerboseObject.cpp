#include <PyTrilinos2_Teuchos_Custom.hpp>
#include <Teuchos_ENull.hpp>
#include <Teuchos_FancyOStream.hpp>
#include <Teuchos_PtrDecl.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <Teuchos_RCPNode.hpp>
#include <Teuchos_VerboseObject.hpp>
#include <Teuchos_any.hpp>
#include <cwchar>
#include <ios>
#include <iterator>
#include <memory>
#include <ostream>
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

// Teuchos::VerboseObjectBase file:Teuchos_VerboseObject.hpp line:66
struct PyCallBack_Teuchos_VerboseObjectBase : public Teuchos::VerboseObjectBase {
	using Teuchos::VerboseObjectBase::VerboseObjectBase;

	const class Teuchos::VerboseObjectBase & setOStream(const class Teuchos::RCP<class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > > & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::VerboseObjectBase *>(this), "setOStream");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<const class Teuchos::VerboseObjectBase &>::value) {
				static pybind11::detail::override_caster_t<const class Teuchos::VerboseObjectBase &> caster;
				return pybind11::detail::cast_ref<const class Teuchos::VerboseObjectBase &>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<const class Teuchos::VerboseObjectBase &>(std::move(o));
		}
		return VerboseObjectBase::setOStream(a0);
	}
	const class Teuchos::VerboseObjectBase & setOverridingOStream(const class Teuchos::RCP<class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > > & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::VerboseObjectBase *>(this), "setOverridingOStream");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<const class Teuchos::VerboseObjectBase &>::value) {
				static pybind11::detail::override_caster_t<const class Teuchos::VerboseObjectBase &> caster;
				return pybind11::detail::cast_ref<const class Teuchos::VerboseObjectBase &>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<const class Teuchos::VerboseObjectBase &>(std::move(o));
		}
		return VerboseObjectBase::setOverridingOStream(a0);
	}
	class Teuchos::VerboseObjectBase & setLinePrefix(const std::string & a0) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::VerboseObjectBase *>(this), "setLinePrefix");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<class Teuchos::VerboseObjectBase &>::value) {
				static pybind11::detail::override_caster_t<class Teuchos::VerboseObjectBase &> caster;
				return pybind11::detail::cast_ref<class Teuchos::VerboseObjectBase &>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<class Teuchos::VerboseObjectBase &>(std::move(o));
		}
		return VerboseObjectBase::setLinePrefix(a0);
	}
	using _binder_ret_0 = class Teuchos::RCP<class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > >;
	_binder_ret_0 getOStream() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::VerboseObjectBase *>(this), "getOStream");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_0>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_0> caster;
				return pybind11::detail::cast_ref<_binder_ret_0>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_0>(std::move(o));
		}
		return VerboseObjectBase::getOStream();
	}
	using _binder_ret_1 = class Teuchos::RCP<class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > >;
	_binder_ret_1 getOverridingOStream() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::VerboseObjectBase *>(this), "getOverridingOStream");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_1>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_1> caster;
				return pybind11::detail::cast_ref<_binder_ret_1>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_1>(std::move(o));
		}
		return VerboseObjectBase::getOverridingOStream();
	}
	std::string getLinePrefix() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::VerboseObjectBase *>(this), "getLinePrefix");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<std::string>::value) {
				static pybind11::detail::override_caster_t<std::string> caster;
				return pybind11::detail::cast_ref<std::string>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<std::string>(std::move(o));
		}
		return VerboseObjectBase::getLinePrefix();
	}
	using _binder_ret_2 = class Teuchos::basic_OSTab<char, struct std::char_traits<char> >;
	_binder_ret_2 getOSTab(const int a0, const std::string & a1) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::VerboseObjectBase *>(this), "getOSTab");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_2>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_2> caster;
				return pybind11::detail::cast_ref<_binder_ret_2>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_2>(std::move(o));
		}
		return VerboseObjectBase::getOSTab(a0, a1);
	}
	void informUpdatedVerbosityState() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::VerboseObjectBase *>(this), "informUpdatedVerbosityState");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return VerboseObjectBase::informUpdatedVerbosityState();
	}
};

void bind_Teuchos_VerboseObject(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // Teuchos::VerboseObjectBase file:Teuchos_VerboseObject.hpp line:66
		pybind11::class_<Teuchos::VerboseObjectBase, Teuchos::RCP<Teuchos::VerboseObjectBase>, PyCallBack_Teuchos_VerboseObjectBase> cl(M("Teuchos"), "VerboseObjectBase", "Non-templated base class for objects that can print their\n activities to a stream.\n\n \n\n Objects that derive from this interface print to a default class-owned\n (i.e. static) output stream object (set using setDefaultOStream())\n or the output stream can be set on an object-by-object basis using\n setOStream().\n\n The output stream type is FancyOStream which allows for automated\n indentation (using the OSTab class) and has other useful features.");
		cl.def( pybind11::init( [](){ return new Teuchos::VerboseObjectBase(); }, [](){ return new PyCallBack_Teuchos_VerboseObjectBase(); } ), "doc");
		cl.def( pybind11::init<const class Teuchos::RCP<class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > > &>(), pybind11::arg("oStream") );

		cl.def( pybind11::init( [](PyCallBack_Teuchos_VerboseObjectBase const &o){ return new PyCallBack_Teuchos_VerboseObjectBase(o); } ) );
		cl.def( pybind11::init( [](Teuchos::VerboseObjectBase const &o){ return new Teuchos::VerboseObjectBase(o); } ) );
		cl.def_static("setDefaultOStream", (void (*)(const class Teuchos::RCP<class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > > &)) &Teuchos::VerboseObjectBase::setDefaultOStream, "Set the default output stream object.\n\n If this function is not called, then a default stream based on\n std::cout is used.\n\nC++: Teuchos::VerboseObjectBase::setDefaultOStream(const class Teuchos::RCP<class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > > &) --> void", pybind11::arg("defaultOStream"));
		cl.def_static("getDefaultOStream", (class Teuchos::RCP<class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > > (*)()) &Teuchos::VerboseObjectBase::getDefaultOStream, "Get the default output stream object. \n\nC++: Teuchos::VerboseObjectBase::getDefaultOStream() --> class Teuchos::RCP<class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > >");
		cl.def("setOStream", (const class Teuchos::VerboseObjectBase & (Teuchos::VerboseObjectBase::*)(const class Teuchos::RCP<class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > > &) const) &Teuchos::VerboseObjectBase::setOStream, "The output stream for *this object.\n\n This function is supposed by called by general clients to set the output\n stream according to some general logic in the code.\n\nC++: Teuchos::VerboseObjectBase::setOStream(const class Teuchos::RCP<class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > > &) const --> const class Teuchos::VerboseObjectBase &", pybind11::return_value_policy::automatic, pybind11::arg("oStream"));
		cl.def("setOverridingOStream", (const class Teuchos::VerboseObjectBase & (Teuchos::VerboseObjectBase::*)(const class Teuchos::RCP<class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > > &) const) &Teuchos::VerboseObjectBase::setOverridingOStream, "Set the overriding the output stream for *this object.\n\n This function is supposed to be called by special clients that want to\n set the output stream in a way that will not be overridden by\n setOStream().\n\nC++: Teuchos::VerboseObjectBase::setOverridingOStream(const class Teuchos::RCP<class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > > &) const --> const class Teuchos::VerboseObjectBase &", pybind11::return_value_policy::automatic, pybind11::arg("oStream"));
		cl.def("setLinePrefix", (class Teuchos::VerboseObjectBase & (Teuchos::VerboseObjectBase::*)(const std::string &)) &Teuchos::VerboseObjectBase::setLinePrefix, "Set line prefix name for this object \n\nC++: Teuchos::VerboseObjectBase::setLinePrefix(const std::string &) --> class Teuchos::VerboseObjectBase &", pybind11::return_value_policy::automatic, pybind11::arg("linePrefix"));
		cl.def("getOStream", (class Teuchos::RCP<class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > > (Teuchos::VerboseObjectBase::*)() const) &Teuchos::VerboseObjectBase::getOStream, "Return the output stream to be used for out for *this\n object.\n\nC++: Teuchos::VerboseObjectBase::getOStream() const --> class Teuchos::RCP<class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > >");
		cl.def("getOverridingOStream", (class Teuchos::RCP<class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > > (Teuchos::VerboseObjectBase::*)() const) &Teuchos::VerboseObjectBase::getOverridingOStream, "Return the the overriding output stream if set.\n\n This is the output stream that will be returned from\n getOStream() regardless that stream is set by\n setOStream().\n\nC++: Teuchos::VerboseObjectBase::getOverridingOStream() const --> class Teuchos::RCP<class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > >");
		cl.def("getLinePrefix", (std::string (Teuchos::VerboseObjectBase::*)() const) &Teuchos::VerboseObjectBase::getLinePrefix, "Get the line prefix for this object \n\nC++: Teuchos::VerboseObjectBase::getLinePrefix() const --> std::string");
		cl.def("getOSTab", [](Teuchos::VerboseObjectBase const &o) -> Teuchos::basic_OSTab<char, struct std::char_traits<char> > { return o.getOSTab(); }, "");
		cl.def("getOSTab", [](Teuchos::VerboseObjectBase const &o, const int & a0) -> Teuchos::basic_OSTab<char, struct std::char_traits<char> > { return o.getOSTab(a0); }, "", pybind11::arg("tabs"));
		cl.def("getOSTab", (class Teuchos::basic_OSTab<char, struct std::char_traits<char> > (Teuchos::VerboseObjectBase::*)(const int, const std::string &) const) &Teuchos::VerboseObjectBase::getOSTab, "Create a tab object which sets the number of tabs and optionally the line prefix.\n\n \n  [in] The number of relative tabs to add (if tabs > 0) or remove (if tabs < 0).\n              If tabs == OSTab::DISABLE_TABBING then tabbing will be turned off temporarily.\n\n \n\n              [in] Sets a line prefix that overrides this->getLinePrefix().\n\n The side effects of these changes go away as soon as the returned\n OSTab object is destroyed at the end of the block of code.\n\n Returns OSTab( this->getOStream(), tabs, linePrefix.length() ? linePrefix : this->getLinePrefix() )\n\nC++: Teuchos::VerboseObjectBase::getOSTab(const int, const std::string &) const --> class Teuchos::basic_OSTab<char, struct std::char_traits<char> >", pybind11::arg("tabs"), pybind11::arg("linePrefix"));
		cl.def("assign", (class Teuchos::VerboseObjectBase & (Teuchos::VerboseObjectBase::*)(const class Teuchos::VerboseObjectBase &)) &Teuchos::VerboseObjectBase::operator=, "C++: Teuchos::VerboseObjectBase::operator=(const class Teuchos::VerboseObjectBase &) --> class Teuchos::VerboseObjectBase &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
}
