#include <PyTrilinos2_Teuchos_Custom.hpp>
#include <Teuchos_ENull.hpp>
#include <Teuchos_FilteredIterator.hpp>
#include <Teuchos_ParameterEntry.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_ParameterListAcceptor.hpp>
#include <Teuchos_ParameterListAcceptorDefaultBase.hpp>
#include <Teuchos_ParameterListModifier.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <Teuchos_RCPNode.hpp>
#include <Teuchos_StringIndexedOrderedValueObjectContainer.hpp>
#include <Teuchos_any.hpp>
#include <deque>
#include <memory>
#include <ostream>
#include <sstream> // __str__
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

// Teuchos::ParameterListAcceptor file:Teuchos_ParameterListAcceptor.hpp line:152
struct PyCallBack_Teuchos_ParameterListAcceptor : public Teuchos::ParameterListAcceptor {
	using Teuchos::ParameterListAcceptor::ParameterListAcceptor;

	void setParameterList(const class Teuchos::RCP<class Teuchos::ParameterList> & a0) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::ParameterListAcceptor *>(this), "setParameterList");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		pybind11::pybind11_fail("Tried to call pure virtual function \"ParameterListAcceptor::setParameterList\"");
	}
	class Teuchos::RCP<class Teuchos::ParameterList> getNonconstParameterList() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::ParameterListAcceptor *>(this), "getNonconstParameterList");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<class Teuchos::RCP<class Teuchos::ParameterList>>::value) {
				static pybind11::detail::override_caster_t<class Teuchos::RCP<class Teuchos::ParameterList>> caster;
				return pybind11::detail::cast_ref<class Teuchos::RCP<class Teuchos::ParameterList>>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<class Teuchos::RCP<class Teuchos::ParameterList>>(std::move(o));
		}
		pybind11::pybind11_fail("Tried to call pure virtual function \"ParameterListAcceptor::getNonconstParameterList\"");
	}
	class Teuchos::RCP<class Teuchos::ParameterList> unsetParameterList() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::ParameterListAcceptor *>(this), "unsetParameterList");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<class Teuchos::RCP<class Teuchos::ParameterList>>::value) {
				static pybind11::detail::override_caster_t<class Teuchos::RCP<class Teuchos::ParameterList>> caster;
				return pybind11::detail::cast_ref<class Teuchos::RCP<class Teuchos::ParameterList>>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<class Teuchos::RCP<class Teuchos::ParameterList>>(std::move(o));
		}
		pybind11::pybind11_fail("Tried to call pure virtual function \"ParameterListAcceptor::unsetParameterList\"");
	}
	class Teuchos::RCP<const class Teuchos::ParameterList> getParameterList() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::ParameterListAcceptor *>(this), "getParameterList");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<class Teuchos::RCP<const class Teuchos::ParameterList>>::value) {
				static pybind11::detail::override_caster_t<class Teuchos::RCP<const class Teuchos::ParameterList>> caster;
				return pybind11::detail::cast_ref<class Teuchos::RCP<const class Teuchos::ParameterList>>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<class Teuchos::RCP<const class Teuchos::ParameterList>>(std::move(o));
		}
		return ParameterListAcceptor::getParameterList();
	}
	class Teuchos::RCP<const class Teuchos::ParameterList> getValidParameters() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::ParameterListAcceptor *>(this), "getValidParameters");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<class Teuchos::RCP<const class Teuchos::ParameterList>>::value) {
				static pybind11::detail::override_caster_t<class Teuchos::RCP<const class Teuchos::ParameterList>> caster;
				return pybind11::detail::cast_ref<class Teuchos::RCP<const class Teuchos::ParameterList>>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<class Teuchos::RCP<const class Teuchos::ParameterList>>(std::move(o));
		}
		return ParameterListAcceptor::getValidParameters();
	}
};

// Teuchos::ParameterListAcceptorDefaultBase file:Teuchos_ParameterListAcceptorDefaultBase.hpp line:61
struct PyCallBack_Teuchos_ParameterListAcceptorDefaultBase : public Teuchos::ParameterListAcceptorDefaultBase {
	using Teuchos::ParameterListAcceptorDefaultBase::ParameterListAcceptorDefaultBase;

	class Teuchos::RCP<class Teuchos::ParameterList> getNonconstParameterList() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::ParameterListAcceptorDefaultBase *>(this), "getNonconstParameterList");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<class Teuchos::RCP<class Teuchos::ParameterList>>::value) {
				static pybind11::detail::override_caster_t<class Teuchos::RCP<class Teuchos::ParameterList>> caster;
				return pybind11::detail::cast_ref<class Teuchos::RCP<class Teuchos::ParameterList>>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<class Teuchos::RCP<class Teuchos::ParameterList>>(std::move(o));
		}
		return ParameterListAcceptorDefaultBase::getNonconstParameterList();
	}
	class Teuchos::RCP<class Teuchos::ParameterList> unsetParameterList() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::ParameterListAcceptorDefaultBase *>(this), "unsetParameterList");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<class Teuchos::RCP<class Teuchos::ParameterList>>::value) {
				static pybind11::detail::override_caster_t<class Teuchos::RCP<class Teuchos::ParameterList>> caster;
				return pybind11::detail::cast_ref<class Teuchos::RCP<class Teuchos::ParameterList>>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<class Teuchos::RCP<class Teuchos::ParameterList>>(std::move(o));
		}
		return ParameterListAcceptorDefaultBase::unsetParameterList();
	}
	class Teuchos::RCP<const class Teuchos::ParameterList> getParameterList() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::ParameterListAcceptorDefaultBase *>(this), "getParameterList");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<class Teuchos::RCP<const class Teuchos::ParameterList>>::value) {
				static pybind11::detail::override_caster_t<class Teuchos::RCP<const class Teuchos::ParameterList>> caster;
				return pybind11::detail::cast_ref<class Teuchos::RCP<const class Teuchos::ParameterList>>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<class Teuchos::RCP<const class Teuchos::ParameterList>>(std::move(o));
		}
		return ParameterListAcceptorDefaultBase::getParameterList();
	}
	void setParameterList(const class Teuchos::RCP<class Teuchos::ParameterList> & a0) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::ParameterListAcceptorDefaultBase *>(this), "setParameterList");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		pybind11::pybind11_fail("Tried to call pure virtual function \"ParameterListAcceptor::setParameterList\"");
	}
	class Teuchos::RCP<const class Teuchos::ParameterList> getValidParameters() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::ParameterListAcceptorDefaultBase *>(this), "getValidParameters");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<class Teuchos::RCP<const class Teuchos::ParameterList>>::value) {
				static pybind11::detail::override_caster_t<class Teuchos::RCP<const class Teuchos::ParameterList>> caster;
				return pybind11::detail::cast_ref<class Teuchos::RCP<const class Teuchos::ParameterList>>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<class Teuchos::RCP<const class Teuchos::ParameterList>>(std::move(o));
		}
		return ParameterListAcceptor::getValidParameters();
	}
};

void bind_Teuchos_ParameterListAcceptor(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // Teuchos::ParameterListAcceptor file:Teuchos_ParameterListAcceptor.hpp line:152
		pybind11::class_<Teuchos::ParameterListAcceptor, Teuchos::RCP<Teuchos::ParameterListAcceptor>, PyCallBack_Teuchos_ParameterListAcceptor> cl(M("Teuchos"), "ParameterListAcceptor", "Interface for objects that can accept a ParameterList.\n\n \n\n Most users only need to know about two methods:\n   - setParameterList()\n   - getValidParameters()\n\n setParameterList() lets users set this object's parameters.\n getValidParameters() returns a default list of parameters,\n including any documentation and/or validators that the subclass may\n provide.  If you call setParameterList(), implementations will fill\n in any missing parameters with their default values, and do\n validation.  That's all you really need to know!  If you want more\n details about semantics, though, please read on.\n\n \n\n \n\n This interface does not define the semantics of calling\n setParametersList() twice with different lists.  For example,\n suppose that the class  takes two parameters:\n \n  An  parameter \"Integer parameter\" \n  A  parameter \"Boolean parameter\" \n \n The default value of the first is 0, and the default value of the\n second is false.  In the following code sample, what is the final\n state of x's parameters?\n \n\n\n\n\n\n\n\n\n\n\n\n\n The answer is that we can't tell without knowing more about\n SomeClass.  There are at least two possibilities:\n \n  \"Integer parameter\" is 0, and \"Boolean parameter\" is true \n  \"Integer parameter\" is 42, and \"Boolean parameter\" is true \n \n\n The first possibility says that the input ParameterList expresses\n the complete state of the object.  Any missing parameters in\n subsequent calls get filled in with their default values.  The\n second possibility says that the input ParameterList expresses a\n \"delta,\" a difference from its current state.  You must read the\n subclass' documentation to determine which of these it implements.\n\n \n\n Developers who would like a simpler interface from which to inherit\n may prefer the subclass ParameterListAcceptorDefaultBase.  That\n class provides default implementations of all but two of this\n class' methods.\n\n It's tempting to begin setParameterList() as follows:\n \n\n\n\n That's correct, but be aware that this can only be used to\n implement \"complete state\" semantics, not \"delta\" semantics.\n This is because validateParametersAndSetDefaults() fills in\n default values, as its name suggests.\n\n Before ParameterList had the validation feature, many\n implementations of setParameterList() would use the two-argument\n version of ParameterList::get(), and supply the current\n value of the parameter as the default if that parameter didn't\n exist in the input list.  This implemented delta semantics.  It is\n unclear whether implementers knew what semantics they were\n implementing, but that was the effect of their code.\n\n If you want to implement delta semantics, and also want to exploit\n the validation feature, you have at least two options.  First, you\n could use the validation method that does not set defaults:\n \n\n\n\n and then use the two-argument version of\n ParameterList::get() in the way discussed above, so that\n existing parameter values don't get replaced with defaults.\n\n The second option is to keep a copy of the ParameterList from the\n previous call to setParameterList().  This must be a deep copy,\n because users might have changed parameters since then.  (It is\n likely that they have just one ParameterList, which they change as\n necessary.)  You may then use that list -- not the result of\n getValidParameters() -- as the input argument of\n validateParametersAndSetDefaults().");
		cl.def(pybind11::init<PyCallBack_Teuchos_ParameterListAcceptor const &>());
		cl.def( pybind11::init( [](){ return new PyCallBack_Teuchos_ParameterListAcceptor(); } ) );
		cl.def("setParameterList", (void (Teuchos::ParameterListAcceptor::*)(const class Teuchos::RCP<class Teuchos::ParameterList> &)) &Teuchos::ParameterListAcceptor::setParameterList, "Set parameters from a parameter list and return with default values.\n\n \n [in/out] On input: contains the parameters set\n   by the client.  On output: the same list, possibly filled with\n   default values, depending on the implementation.\n\n Implementations of this method generally read parameters out of\n  and use them to modify the state or behavior of\n this object.  Implementations may validate input parameters, and\n throw an exception or set an error state if any of them are\n invalid.  \"Validation\n\n \n  ! paramList.is_null () \n \n\n this->getParameterList().get() == paramList.get()\n\n This object \"remembers\"  until it is \"unset\" using\n unsetParameterList().  When the input ParameterList is passed in,\n we assume that the client has finished setting parameters in the\n ParameterList.  If the client changes  after calling\n this method, this object's behavior is undefined.  This is\n because the object may read the options from  at any\n time.  It may either do so in this method, or it may wait to read\n them at some later time.  Users should not expect that if they\n change a parameter, that this object will automatically recognize\n the change.  To change even one parameter, this method must be\n called again.\n\nC++: Teuchos::ParameterListAcceptor::setParameterList(const class Teuchos::RCP<class Teuchos::ParameterList> &) --> void", pybind11::arg("paramList"));
		cl.def("getNonconstParameterList", (class Teuchos::RCP<class Teuchos::ParameterList> (Teuchos::ParameterListAcceptor::*)()) &Teuchos::ParameterListAcceptor::getNonconstParameterList, "Get a nonconst version of the parameter list that was set\n    using setParameterList().\n\n The returned ParameterList should be the same object (pointer\n equality) as the object given to setParameterList().  If\n setParameterList() has not yet been called on this object, the\n returned RCP may be null, but need not necessarily be.  If\n unsetParameterList()\n\nC++: Teuchos::ParameterListAcceptor::getNonconstParameterList() --> class Teuchos::RCP<class Teuchos::ParameterList>");
		cl.def("unsetParameterList", (class Teuchos::RCP<class Teuchos::ParameterList> (Teuchos::ParameterListAcceptor::*)()) &Teuchos::ParameterListAcceptor::unsetParameterList, "Unset the parameter list that was set using setParameterList().\n\n This does not undo the effect of setting the parameters\n via a call to setParameterList().  It merely \"forgets\" the RCP,\n so that getParameterList() and getNonconstParameterList() both\n return null.\n\n \n  this->getParameter().is_null () \n \n\n  this->getNonconstParameter().is_null () \n\nC++: Teuchos::ParameterListAcceptor::unsetParameterList() --> class Teuchos::RCP<class Teuchos::ParameterList>");
		cl.def("getParameterList", (class Teuchos::RCP<const class Teuchos::ParameterList> (Teuchos::ParameterListAcceptor::*)() const) &Teuchos::ParameterListAcceptor::getParameterList, "Get const version of the parameter list that was set\n    using setParameterList().\n\n The default implementation returns:\n   \n\n\n\n   \n\nC++: Teuchos::ParameterListAcceptor::getParameterList() const --> class Teuchos::RCP<const class Teuchos::ParameterList>");
		cl.def("getValidParameters", (class Teuchos::RCP<const class Teuchos::ParameterList> (Teuchos::ParameterListAcceptor::*)() const) &Teuchos::ParameterListAcceptor::getValidParameters, "Return a ParameterList containing all of the valid\n   parameters that this->setParameterList(...) will\n   accept, along with any validators.\n\n Implementations of setParameterList() may use the list returned\n by getValidParameters() to validate the input ParameterList.\n\n The default implementation returns null.\n\nC++: Teuchos::ParameterListAcceptor::getValidParameters() const --> class Teuchos::RCP<const class Teuchos::ParameterList>");
		cl.def("assign", (class Teuchos::ParameterListAcceptor & (Teuchos::ParameterListAcceptor::*)(const class Teuchos::ParameterListAcceptor &)) &Teuchos::ParameterListAcceptor::operator=, "C++: Teuchos::ParameterListAcceptor::operator=(const class Teuchos::ParameterListAcceptor &) --> class Teuchos::ParameterListAcceptor &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // Teuchos::ParameterListAcceptorDefaultBase file:Teuchos_ParameterListAcceptorDefaultBase.hpp line:61
		pybind11::class_<Teuchos::ParameterListAcceptorDefaultBase, Teuchos::RCP<Teuchos::ParameterListAcceptorDefaultBase>, PyCallBack_Teuchos_ParameterListAcceptorDefaultBase, Teuchos::ParameterListAcceptor> cl(M("Teuchos"), "ParameterListAcceptorDefaultBase", "Intermediate node base class for objects that accept parameter lists\n that implements some of the needed behavior automatically.\n\n Subclasses just need to implement setParameterList() and\n getValidParameters().  The underlying parameter list is accessed\n using the non-virtual protected members setMyParamList() and\n getMyParamList().");
		cl.def(pybind11::init<PyCallBack_Teuchos_ParameterListAcceptorDefaultBase const &>());
		cl.def( pybind11::init( [](){ return new PyCallBack_Teuchos_ParameterListAcceptorDefaultBase(); } ) );
		cl.def("getNonconstParameterList", (class Teuchos::RCP<class Teuchos::ParameterList> (Teuchos::ParameterListAcceptorDefaultBase::*)()) &Teuchos::ParameterListAcceptorDefaultBase::getNonconstParameterList, ". \n\nC++: Teuchos::ParameterListAcceptorDefaultBase::getNonconstParameterList() --> class Teuchos::RCP<class Teuchos::ParameterList>");
		cl.def("unsetParameterList", (class Teuchos::RCP<class Teuchos::ParameterList> (Teuchos::ParameterListAcceptorDefaultBase::*)()) &Teuchos::ParameterListAcceptorDefaultBase::unsetParameterList, ". \n\nC++: Teuchos::ParameterListAcceptorDefaultBase::unsetParameterList() --> class Teuchos::RCP<class Teuchos::ParameterList>");
		cl.def("getParameterList", (class Teuchos::RCP<const class Teuchos::ParameterList> (Teuchos::ParameterListAcceptorDefaultBase::*)() const) &Teuchos::ParameterListAcceptorDefaultBase::getParameterList, ". \n\nC++: Teuchos::ParameterListAcceptorDefaultBase::getParameterList() const --> class Teuchos::RCP<const class Teuchos::ParameterList>");
		cl.def("assign", (class Teuchos::ParameterListAcceptorDefaultBase & (Teuchos::ParameterListAcceptorDefaultBase::*)(const class Teuchos::ParameterListAcceptorDefaultBase &)) &Teuchos::ParameterListAcceptorDefaultBase::operator=, "C++: Teuchos::ParameterListAcceptorDefaultBase::operator=(const class Teuchos::ParameterListAcceptorDefaultBase &) --> class Teuchos::ParameterListAcceptorDefaultBase &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
}
