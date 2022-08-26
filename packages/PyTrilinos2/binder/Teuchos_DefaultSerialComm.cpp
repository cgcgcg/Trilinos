#include <PyTrilinos2_Teuchos_Custom.hpp>
#include <Teuchos_ArrayViewDecl.hpp> // Teuchos::ArrayView
#include <Teuchos_Comm.hpp> // Teuchos::Comm
#include <Teuchos_Comm.hpp> // Teuchos::CommRequest
#include <Teuchos_Comm.hpp> // Teuchos::CommStatus
#include <Teuchos_DefaultSerialComm.hpp> // Teuchos::SerialComm
#include <Teuchos_DefaultSerialComm.hpp> // Teuchos::SerialCommStatus
#include <Teuchos_Describable.hpp> // Teuchos::Describable
#include <Teuchos_ENull.hpp> // Teuchos::ENull
#include <Teuchos_EReductionType.hpp> // Teuchos::EReductionType
#include <Teuchos_EReductionType.hpp> // Teuchos::toString
#include <Teuchos_FancyOStream.hpp> // Teuchos::basic_FancyOStream
#include <Teuchos_LabeledObject.hpp> // Teuchos::LabeledObject
#include <Teuchos_PtrDecl.hpp> // Teuchos::Ptr
#include <Teuchos_RCPDecl.hpp> // Teuchos::ERCPUndefinedWeakNoDealloc
#include <Teuchos_RCPDecl.hpp> // Teuchos::ERCPWeakNoDealloc
#include <Teuchos_RCPDecl.hpp> // Teuchos::RCP
#include <Teuchos_RCPNode.hpp> // Teuchos::EPrePostDestruction
#include <Teuchos_RCPNode.hpp> // Teuchos::ERCPNodeLookup
#include <Teuchos_RCPNode.hpp> // Teuchos::ERCPStrength
#include <Teuchos_RCPNode.hpp> // Teuchos::RCPNode
#include <Teuchos_RCPNode.hpp> // Teuchos::RCPNodeHandle
#include <Teuchos_VerbosityLevel.hpp> // Teuchos::EVerbosityLevel
#include <Teuchos_any.hpp> // Teuchos::any
#include <iterator> // __gnu_cxx::__normal_iterator
#include <memory> // std::allocator
#include <ostream> // std::basic_ostream
#include <sstream> // __str__
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

// Teuchos::SerialCommStatus file:Teuchos_DefaultSerialComm.hpp line:58
struct PyCallBack_Teuchos_SerialCommStatus_int_t : public Teuchos::SerialCommStatus<int> {
	using Teuchos::SerialCommStatus<int>::SerialCommStatus;

	int getSourceRank() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::SerialCommStatus<int> *>(this), "getSourceRank");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<int>::value) {
				static pybind11::detail::override_caster_t<int> caster;
				return pybind11::detail::cast_ref<int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<int>(std::move(o));
		}
		return SerialCommStatus::getSourceRank();
	}
	int getTag() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::SerialCommStatus<int> *>(this), "getTag");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<int>::value) {
				static pybind11::detail::override_caster_t<int> caster;
				return pybind11::detail::cast_ref<int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<int>(std::move(o));
		}
		return SerialCommStatus::getTag();
	}
};

// Teuchos::SerialComm file:Teuchos_DefaultSerialComm.hpp line:76
struct PyCallBack_Teuchos_SerialComm_int_t : public Teuchos::SerialComm<int> {
	using Teuchos::SerialComm<int>::SerialComm;

	int getTag() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::SerialComm<int> *>(this), "getTag");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<int>::value) {
				static pybind11::detail::override_caster_t<int> caster;
				return pybind11::detail::cast_ref<int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<int>(std::move(o));
		}
		return SerialComm::getTag();
	}
	int getRank() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::SerialComm<int> *>(this), "getRank");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<int>::value) {
				static pybind11::detail::override_caster_t<int> caster;
				return pybind11::detail::cast_ref<int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<int>(std::move(o));
		}
		return SerialComm::getRank();
	}
	int getSize() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::SerialComm<int> *>(this), "getSize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<int>::value) {
				static pybind11::detail::override_caster_t<int> caster;
				return pybind11::detail::cast_ref<int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<int>(std::move(o));
		}
		return SerialComm::getSize();
	}
	void barrier() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::SerialComm<int> *>(this), "barrier");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return SerialComm::barrier();
	}
	void readySend(const class Teuchos::ArrayView<const char> & a0, const int a1) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::SerialComm<int> *>(this), "readySend");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return SerialComm::readySend(a0, a1);
	}
	class Teuchos::RCP<class Teuchos::CommRequest<int> > isend(const class Teuchos::ArrayView<const char> & a0, const int a1) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::SerialComm<int> *>(this), "isend");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<class Teuchos::RCP<class Teuchos::CommRequest<int> >>::value) {
				static pybind11::detail::override_caster_t<class Teuchos::RCP<class Teuchos::CommRequest<int> >> caster;
				return pybind11::detail::cast_ref<class Teuchos::RCP<class Teuchos::CommRequest<int> >>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<class Teuchos::RCP<class Teuchos::CommRequest<int> >>(std::move(o));
		}
		return SerialComm::isend(a0, a1);
	}
	class Teuchos::RCP<class Teuchos::CommRequest<int> > isend(const class Teuchos::ArrayView<const char> & a0, const int a1, const int a2) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::SerialComm<int> *>(this), "isend");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<class Teuchos::RCP<class Teuchos::CommRequest<int> >>::value) {
				static pybind11::detail::override_caster_t<class Teuchos::RCP<class Teuchos::CommRequest<int> >> caster;
				return pybind11::detail::cast_ref<class Teuchos::RCP<class Teuchos::CommRequest<int> >>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<class Teuchos::RCP<class Teuchos::CommRequest<int> >>(std::move(o));
		}
		return SerialComm::isend(a0, a1, a2);
	}
	class Teuchos::RCP<class Teuchos::CommRequest<int> > ireceive(const class Teuchos::ArrayView<char> & a0, const int a1) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::SerialComm<int> *>(this), "ireceive");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<class Teuchos::RCP<class Teuchos::CommRequest<int> >>::value) {
				static pybind11::detail::override_caster_t<class Teuchos::RCP<class Teuchos::CommRequest<int> >> caster;
				return pybind11::detail::cast_ref<class Teuchos::RCP<class Teuchos::CommRequest<int> >>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<class Teuchos::RCP<class Teuchos::CommRequest<int> >>(std::move(o));
		}
		return SerialComm::ireceive(a0, a1);
	}
	class Teuchos::RCP<class Teuchos::CommRequest<int> > ireceive(const class Teuchos::ArrayView<char> & a0, const int a1, const int a2) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::SerialComm<int> *>(this), "ireceive");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<class Teuchos::RCP<class Teuchos::CommRequest<int> >>::value) {
				static pybind11::detail::override_caster_t<class Teuchos::RCP<class Teuchos::CommRequest<int> >> caster;
				return pybind11::detail::cast_ref<class Teuchos::RCP<class Teuchos::CommRequest<int> >>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<class Teuchos::RCP<class Teuchos::CommRequest<int> >>(std::move(o));
		}
		return SerialComm::ireceive(a0, a1, a2);
	}
	class Teuchos::RCP<class Teuchos::Comm<int> > duplicate() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::SerialComm<int> *>(this), "duplicate");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<class Teuchos::RCP<class Teuchos::Comm<int> >>::value) {
				static pybind11::detail::override_caster_t<class Teuchos::RCP<class Teuchos::Comm<int> >> caster;
				return pybind11::detail::cast_ref<class Teuchos::RCP<class Teuchos::Comm<int> >>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<class Teuchos::RCP<class Teuchos::Comm<int> >>(std::move(o));
		}
		return SerialComm::duplicate();
	}
	class Teuchos::RCP<class Teuchos::Comm<int> > split(const int a0, const int a1) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::SerialComm<int> *>(this), "split");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<class Teuchos::RCP<class Teuchos::Comm<int> >>::value) {
				static pybind11::detail::override_caster_t<class Teuchos::RCP<class Teuchos::Comm<int> >> caster;
				return pybind11::detail::cast_ref<class Teuchos::RCP<class Teuchos::Comm<int> >>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<class Teuchos::RCP<class Teuchos::Comm<int> >>(std::move(o));
		}
		return SerialComm::split(a0, a1);
	}
	class Teuchos::RCP<class Teuchos::Comm<int> > createSubcommunicator(const class Teuchos::ArrayView<const int> & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::SerialComm<int> *>(this), "createSubcommunicator");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<class Teuchos::RCP<class Teuchos::Comm<int> >>::value) {
				static pybind11::detail::override_caster_t<class Teuchos::RCP<class Teuchos::Comm<int> >> caster;
				return pybind11::detail::cast_ref<class Teuchos::RCP<class Teuchos::Comm<int> >>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<class Teuchos::RCP<class Teuchos::Comm<int> >>(std::move(o));
		}
		return SerialComm::createSubcommunicator(a0);
	}
	std::string description() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::SerialComm<int> *>(this), "description");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<std::string>::value) {
				static pybind11::detail::override_caster_t<std::string> caster;
				return pybind11::detail::cast_ref<std::string>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<std::string>(std::move(o));
		}
		return SerialComm::description();
	}
	void describe(class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > & a0, const enum Teuchos::EVerbosityLevel a1) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::SerialComm<int> *>(this), "describe");
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
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::SerialComm<int> *>(this), "setObjectLabel");
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
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::SerialComm<int> *>(this), "getObjectLabel");
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

void bind_Teuchos_DefaultSerialComm(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // Teuchos::SerialCommStatus file:Teuchos_DefaultSerialComm.hpp line:58
		pybind11::class_<Teuchos::SerialCommStatus<int>, Teuchos::RCP<Teuchos::SerialCommStatus<int>>, PyCallBack_Teuchos_SerialCommStatus_int_t, Teuchos::CommStatus<int>> cl(M("Teuchos"), "SerialCommStatus_int_t", "");
		cl.def( pybind11::init( [](){ return new Teuchos::SerialCommStatus<int>(); }, [](){ return new PyCallBack_Teuchos_SerialCommStatus_int_t(); } ) );
		cl.def( pybind11::init( [](PyCallBack_Teuchos_SerialCommStatus_int_t const &o){ return new PyCallBack_Teuchos_SerialCommStatus_int_t(o); } ) );
		cl.def( pybind11::init( [](Teuchos::SerialCommStatus<int> const &o){ return new Teuchos::SerialCommStatus<int>(o); } ) );
		cl.def("getSourceRank", (int (Teuchos::SerialCommStatus<int>::*)()) &Teuchos::SerialCommStatus<int>::getSourceRank, "C++: Teuchos::SerialCommStatus<int>::getSourceRank() --> int");
		cl.def("getTag", (int (Teuchos::SerialCommStatus<int>::*)()) &Teuchos::SerialCommStatus<int>::getTag, "C++: Teuchos::SerialCommStatus<int>::getTag() --> int");
		cl.def("assign", (class Teuchos::SerialCommStatus<int> & (Teuchos::SerialCommStatus<int>::*)(const class Teuchos::SerialCommStatus<int> &)) &Teuchos::SerialCommStatus<int>::operator=, "C++: Teuchos::SerialCommStatus<int>::operator=(const class Teuchos::SerialCommStatus<int> &) --> class Teuchos::SerialCommStatus<int> &", pybind11::return_value_policy::automatic, pybind11::arg(""));
		cl.def("getSourceRank", (int (Teuchos::CommStatus<int>::*)()) &Teuchos::CommStatus<int>::getSourceRank, "C++: Teuchos::CommStatus<int>::getSourceRank() --> int");
		cl.def("getTag", (int (Teuchos::CommStatus<int>::*)()) &Teuchos::CommStatus<int>::getTag, "C++: Teuchos::CommStatus<int>::getTag() --> int");
		cl.def("assign", (class Teuchos::CommStatus<int> & (Teuchos::CommStatus<int>::*)(const class Teuchos::CommStatus<int> &)) &Teuchos::CommStatus<int>::operator=, "C++: Teuchos::CommStatus<int>::operator=(const class Teuchos::CommStatus<int> &) --> class Teuchos::CommStatus<int> &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // Teuchos::SerialComm file:Teuchos_DefaultSerialComm.hpp line:76
		pybind11::class_<Teuchos::SerialComm<int>, Teuchos::RCP<Teuchos::SerialComm<int>>, PyCallBack_Teuchos_SerialComm_int_t, Teuchos::Comm<int>> cl(M("Teuchos"), "SerialComm_int_t", "");
		cl.def( pybind11::init( [](){ return new Teuchos::SerialComm<int>(); }, [](){ return new PyCallBack_Teuchos_SerialComm_int_t(); } ) );
		cl.def( pybind11::init( [](PyCallBack_Teuchos_SerialComm_int_t const &o){ return new PyCallBack_Teuchos_SerialComm_int_t(o); } ) );
		cl.def( pybind11::init( [](Teuchos::SerialComm<int> const &o){ return new Teuchos::SerialComm<int>(o); } ) );
		cl.def("getTag", (int (Teuchos::SerialComm<int>::*)() const) &Teuchos::SerialComm<int>::getTag, "C++: Teuchos::SerialComm<int>::getTag() const --> int");
		cl.def("getRank", (int (Teuchos::SerialComm<int>::*)() const) &Teuchos::SerialComm<int>::getRank, "C++: Teuchos::SerialComm<int>::getRank() const --> int");
		cl.def("getSize", (int (Teuchos::SerialComm<int>::*)() const) &Teuchos::SerialComm<int>::getSize, "C++: Teuchos::SerialComm<int>::getSize() const --> int");
		cl.def("barrier", (void (Teuchos::SerialComm<int>::*)() const) &Teuchos::SerialComm<int>::barrier, "C++: Teuchos::SerialComm<int>::barrier() const --> void");
		cl.def("readySend", (void (Teuchos::SerialComm<int>::*)(const class Teuchos::ArrayView<const char> &, const int) const) &Teuchos::SerialComm<int>::readySend, "C++: Teuchos::SerialComm<int>::readySend(const class Teuchos::ArrayView<const char> &, const int) const --> void", pybind11::arg(""), pybind11::arg(""));
		cl.def("isend", (class Teuchos::RCP<class Teuchos::CommRequest<int> > (Teuchos::SerialComm<int>::*)(const class Teuchos::ArrayView<const char> &, const int) const) &Teuchos::SerialComm<int>::isend, "C++: Teuchos::SerialComm<int>::isend(const class Teuchos::ArrayView<const char> &, const int) const --> class Teuchos::RCP<class Teuchos::CommRequest<int> >", pybind11::arg(""), pybind11::arg(""));
		cl.def("isend", (class Teuchos::RCP<class Teuchos::CommRequest<int> > (Teuchos::SerialComm<int>::*)(const class Teuchos::ArrayView<const char> &, const int, const int) const) &Teuchos::SerialComm<int>::isend, "C++: Teuchos::SerialComm<int>::isend(const class Teuchos::ArrayView<const char> &, const int, const int) const --> class Teuchos::RCP<class Teuchos::CommRequest<int> >", pybind11::arg(""), pybind11::arg(""), pybind11::arg(""));
		cl.def("ireceive", (class Teuchos::RCP<class Teuchos::CommRequest<int> > (Teuchos::SerialComm<int>::*)(const class Teuchos::ArrayView<char> &, const int) const) &Teuchos::SerialComm<int>::ireceive, "C++: Teuchos::SerialComm<int>::ireceive(const class Teuchos::ArrayView<char> &, const int) const --> class Teuchos::RCP<class Teuchos::CommRequest<int> >", pybind11::arg(""), pybind11::arg(""));
		cl.def("ireceive", (class Teuchos::RCP<class Teuchos::CommRequest<int> > (Teuchos::SerialComm<int>::*)(const class Teuchos::ArrayView<char> &, const int, const int) const) &Teuchos::SerialComm<int>::ireceive, "C++: Teuchos::SerialComm<int>::ireceive(const class Teuchos::ArrayView<char> &, const int, const int) const --> class Teuchos::RCP<class Teuchos::CommRequest<int> >", pybind11::arg(""), pybind11::arg(""), pybind11::arg(""));
		cl.def("duplicate", (class Teuchos::RCP<class Teuchos::Comm<int> > (Teuchos::SerialComm<int>::*)() const) &Teuchos::SerialComm<int>::duplicate, "C++: Teuchos::SerialComm<int>::duplicate() const --> class Teuchos::RCP<class Teuchos::Comm<int> >");
		cl.def("split", (class Teuchos::RCP<class Teuchos::Comm<int> > (Teuchos::SerialComm<int>::*)(const int, const int) const) &Teuchos::SerialComm<int>::split, "C++: Teuchos::SerialComm<int>::split(const int, const int) const --> class Teuchos::RCP<class Teuchos::Comm<int> >", pybind11::arg("color"), pybind11::arg(""));
		cl.def("createSubcommunicator", (class Teuchos::RCP<class Teuchos::Comm<int> > (Teuchos::SerialComm<int>::*)(const class Teuchos::ArrayView<const int> &) const) &Teuchos::SerialComm<int>::createSubcommunicator, "C++: Teuchos::SerialComm<int>::createSubcommunicator(const class Teuchos::ArrayView<const int> &) const --> class Teuchos::RCP<class Teuchos::Comm<int> >", pybind11::arg("ranks"));
		cl.def("description", (std::string (Teuchos::SerialComm<int>::*)() const) &Teuchos::SerialComm<int>::description, "C++: Teuchos::SerialComm<int>::description() const --> std::string");
		cl.def("assign", (class Teuchos::SerialComm<int> & (Teuchos::SerialComm<int>::*)(const class Teuchos::SerialComm<int> &)) &Teuchos::SerialComm<int>::operator=, "C++: Teuchos::SerialComm<int>::operator=(const class Teuchos::SerialComm<int> &) --> class Teuchos::SerialComm<int> &", pybind11::return_value_policy::automatic, pybind11::arg(""));
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
	// Teuchos::EReductionType file:Teuchos_EReductionType.hpp line:71
	pybind11::enum_<Teuchos::EReductionType>(M("Teuchos"), "EReductionType", pybind11::arithmetic(), "Predefined reduction operations that Teuchos::Comm understands\n \n\n\n Teuchos' Comm class wraps MPI (the Message Passing Interface for\n distributed-memory parallel programming).  If you do not have MPI,\n it imitates MPI's functionality, as if you were running with a\n single \"parallel\" process.  This means that Teuchos must wrap a\n subset of MPI's functionality, so that it can build without MPI.\n\n Comm provides wrappers for   and\n other collectives that take a reduction operator \n Teuchos wraps  in two different ways.  The first way is\n this enum, which lets users pick from a set of common predefined\n   The second way is through Teuchos' wrappers for custom\n  namely ValueTypeReductionOp and ValueTypeReductionOp.\n Most users should find the reduction operators below sufficient.")
		.value("REDUCE_SUM", Teuchos::REDUCE_SUM)
		.value("REDUCE_MIN", Teuchos::REDUCE_MIN)
		.value("REDUCE_MAX", Teuchos::REDUCE_MAX)
		.value("REDUCE_AND", Teuchos::REDUCE_AND)
		.value("REDUCE_BOR", Teuchos::REDUCE_BOR)
		.export_values();

;

	// Teuchos::toString(const enum Teuchos::EReductionType) file:Teuchos_EReductionType.hpp line:81
	M("Teuchos").def("toString", (const char * (*)(const enum Teuchos::EReductionType)) &Teuchos::toString, "Convert EReductionType to string representation.\n \n\n\nC++: Teuchos::toString(const enum Teuchos::EReductionType) --> const char *", pybind11::return_value_policy::automatic, pybind11::arg("reductType"));

}
