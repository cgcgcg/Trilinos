#include <KokkosCompat_ClassicNodeAPI_Wrapper.hpp>
#include <Kokkos_AnonymousSpace.hpp>
#include <Kokkos_HostSpace.hpp>
#include <Kokkos_Layout.hpp>
#include <Kokkos_ScratchSpace.hpp>
#include <Kokkos_Serial.hpp>
#include <Kokkos_View.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_ArrayViewDecl.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_Describable.hpp>
#include <Teuchos_ENull.hpp>
#include <Teuchos_FancyOStream.hpp>
#include <Teuchos_FilteredIterator.hpp>
#include <Teuchos_LabeledObject.hpp>
#include <Teuchos_ParameterEntry.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_ParameterListAcceptorDefaultBase.hpp>
#include <Teuchos_ParameterListModifier.hpp>
#include <Teuchos_PtrDecl.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <Teuchos_RCPNode.hpp>
#include <Teuchos_StringIndexedOrderedValueObjectContainer.hpp>
#include <Teuchos_VerbosityLevel.hpp>
#include <Teuchos_any.hpp>
#include <Tpetra_Details_DistributorPlan.hpp>
#include <Tpetra_Distributor.hpp>
#include <Tpetra_Export_decl.hpp>
#include <Tpetra_Import_decl.hpp>
#include <Tpetra_RowGraph_decl.hpp>
#include <cwchar>
#include <deque>
#include <impl/Kokkos_SharedAlloc.hpp>
#include <impl/Kokkos_ViewCtor.hpp>
#include <impl/Kokkos_ViewMapping.hpp>
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

// Tpetra::Import file:Tpetra_Import_decl.hpp line:109
struct PyCallBack_Tpetra_Import_int_long_long_Kokkos_Compat_KokkosDeviceWrapperNode_Kokkos_Serial_Kokkos_HostSpace_t : public Tpetra::Import<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> {
	using Tpetra::Import<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::Import;

	void describe(class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > & a0, const enum Teuchos::EVerbosityLevel a1) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::Import<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> *>(this), "describe");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Import::describe(a0, a1);
	}
	std::string description() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::Import<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> *>(this), "description");
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
	void setObjectLabel(const std::string & a0) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::Import<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> *>(this), "setObjectLabel");
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
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::Import<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> *>(this), "getObjectLabel");
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

// Tpetra::Export file:Tpetra_Export_decl.hpp line:117
struct PyCallBack_Tpetra_Export_int_long_long_Kokkos_Compat_KokkosDeviceWrapperNode_Kokkos_Serial_Kokkos_HostSpace_t : public Tpetra::Export<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> {
	using Tpetra::Export<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::Export;

	void describe(class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > & a0, const enum Teuchos::EVerbosityLevel a1) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::Export<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> *>(this), "describe");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Export::describe(a0, a1);
	}
	std::string description() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::Export<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> *>(this), "description");
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
	void setObjectLabel(const std::string & a0) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::Export<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> *>(this), "setObjectLabel");
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
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::Export<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> *>(this), "getObjectLabel");
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

// Tpetra::Distributor file:Tpetra_Distributor.hpp line:132
struct PyCallBack_Tpetra_Distributor : public Tpetra::Distributor {
	using Tpetra::Distributor::Distributor;

	void setParameterList(const class Teuchos::RCP<class Teuchos::ParameterList> & a0) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::Distributor *>(this), "setParameterList");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Distributor::setParameterList(a0);
	}
	class Teuchos::RCP<const class Teuchos::ParameterList> getValidParameters() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::Distributor *>(this), "getValidParameters");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<class Teuchos::RCP<const class Teuchos::ParameterList>>::value) {
				static pybind11::detail::override_caster_t<class Teuchos::RCP<const class Teuchos::ParameterList>> caster;
				return pybind11::detail::cast_ref<class Teuchos::RCP<const class Teuchos::ParameterList>>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<class Teuchos::RCP<const class Teuchos::ParameterList>>(std::move(o));
		}
		return Distributor::getValidParameters();
	}
	std::string description() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::Distributor *>(this), "description");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<std::string>::value) {
				static pybind11::detail::override_caster_t<std::string> caster;
				return pybind11::detail::cast_ref<std::string>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<std::string>(std::move(o));
		}
		return Distributor::description();
	}
	void describe(class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > & a0, const enum Teuchos::EVerbosityLevel a1) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::Distributor *>(this), "describe");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Distributor::describe(a0, a1);
	}
	void setObjectLabel(const std::string & a0) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::Distributor *>(this), "setObjectLabel");
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
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::Distributor *>(this), "getObjectLabel");
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
	class Teuchos::RCP<class Teuchos::ParameterList> getNonconstParameterList() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::Distributor *>(this), "getNonconstParameterList");
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
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::Distributor *>(this), "unsetParameterList");
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
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::Distributor *>(this), "getParameterList");
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
};

void bind_Tpetra_Import_decl(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // Tpetra::Import file:Tpetra_Import_decl.hpp line:109
		pybind11::class_<Tpetra::Import<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>, Teuchos::RCP<Tpetra::Import<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>>, PyCallBack_Tpetra_Import_int_long_long_Kokkos_Compat_KokkosDeviceWrapperNode_Kokkos_Serial_Kokkos_HostSpace_t> cl(M("Tpetra"), "Import_int_long_long_Kokkos_Compat_KokkosDeviceWrapperNode_Kokkos_Serial_Kokkos_HostSpace_t", "");
		cl.def( pybind11::init<const class Teuchos::RCP<const class Tpetra::Map<int, long long> > &, const class Teuchos::RCP<const class Tpetra::Map<int, long long> > &>(), pybind11::arg("source"), pybind11::arg("target") );

		cl.def( pybind11::init<const class Teuchos::RCP<const class Tpetra::Map<int, long long> > &, const class Teuchos::RCP<const class Tpetra::Map<int, long long> > &, const class Teuchos::RCP<class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > > &>(), pybind11::arg("source"), pybind11::arg("target"), pybind11::arg("out") );

		cl.def( pybind11::init<const class Teuchos::RCP<const class Tpetra::Map<int, long long> > &, const class Teuchos::RCP<const class Tpetra::Map<int, long long> > &, const class Teuchos::RCP<class Teuchos::ParameterList> &>(), pybind11::arg("source"), pybind11::arg("target"), pybind11::arg("plist") );

		cl.def( pybind11::init<const class Teuchos::RCP<const class Tpetra::Map<int, long long> > &, const class Teuchos::RCP<const class Tpetra::Map<int, long long> > &, const class Teuchos::RCP<class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > > &, const class Teuchos::RCP<class Teuchos::ParameterList> &>(), pybind11::arg("source"), pybind11::arg("target"), pybind11::arg("out"), pybind11::arg("plist") );

		cl.def( pybind11::init( [](const class Teuchos::RCP<const class Tpetra::Map<int, long long> > & a0, const class Teuchos::RCP<const class Tpetra::Map<int, long long> > & a1, class Teuchos::Array<int> & a2){ return new Tpetra::Import<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>(a0, a1, a2); }, [](const class Teuchos::RCP<const class Tpetra::Map<int, long long> > & a0, const class Teuchos::RCP<const class Tpetra::Map<int, long long> > & a1, class Teuchos::Array<int> & a2){ return new PyCallBack_Tpetra_Import_int_long_long_Kokkos_Compat_KokkosDeviceWrapperNode_Kokkos_Serial_Kokkos_HostSpace_t(a0, a1, a2); } ), "doc");
		cl.def( pybind11::init<const class Teuchos::RCP<const class Tpetra::Map<int, long long> > &, const class Teuchos::RCP<const class Tpetra::Map<int, long long> > &, class Teuchos::Array<int> &, const class Teuchos::RCP<class Teuchos::ParameterList> &>(), pybind11::arg("source"), pybind11::arg("target"), pybind11::arg("remotePIDs"), pybind11::arg("plist") );

		cl.def( pybind11::init( [](PyCallBack_Tpetra_Import_int_long_long_Kokkos_Compat_KokkosDeviceWrapperNode_Kokkos_Serial_Kokkos_HostSpace_t const &o){ return new PyCallBack_Tpetra_Import_int_long_long_Kokkos_Compat_KokkosDeviceWrapperNode_Kokkos_Serial_Kokkos_HostSpace_t(o); } ) );
		cl.def( pybind11::init( [](Tpetra::Import<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> const &o){ return new Tpetra::Import<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>(o); } ) );
		cl.def( pybind11::init<const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &>(), pybind11::arg("exporter") );

		cl.def( pybind11::init( [](const class Teuchos::RCP<const class Tpetra::Map<int, long long> > & a0, const class Teuchos::RCP<const class Tpetra::Map<int, long long> > & a1, const class Teuchos::ArrayView<int> & a2, const class Teuchos::ArrayView<const int> & a3, const class Teuchos::ArrayView<const int> & a4){ return new Tpetra::Import<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>(a0, a1, a2, a3, a4); }, [](const class Teuchos::RCP<const class Tpetra::Map<int, long long> > & a0, const class Teuchos::RCP<const class Tpetra::Map<int, long long> > & a1, const class Teuchos::ArrayView<int> & a2, const class Teuchos::ArrayView<const int> & a3, const class Teuchos::ArrayView<const int> & a4){ return new PyCallBack_Tpetra_Import_int_long_long_Kokkos_Compat_KokkosDeviceWrapperNode_Kokkos_Serial_Kokkos_HostSpace_t(a0, a1, a2, a3, a4); } ), "doc");
		cl.def( pybind11::init( [](const class Teuchos::RCP<const class Tpetra::Map<int, long long> > & a0, const class Teuchos::RCP<const class Tpetra::Map<int, long long> > & a1, const class Teuchos::ArrayView<int> & a2, const class Teuchos::ArrayView<const int> & a3, const class Teuchos::ArrayView<const int> & a4, const class Teuchos::RCP<class Teuchos::ParameterList> & a5){ return new Tpetra::Import<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>(a0, a1, a2, a3, a4, a5); }, [](const class Teuchos::RCP<const class Tpetra::Map<int, long long> > & a0, const class Teuchos::RCP<const class Tpetra::Map<int, long long> > & a1, const class Teuchos::ArrayView<int> & a2, const class Teuchos::ArrayView<const int> & a3, const class Teuchos::ArrayView<const int> & a4, const class Teuchos::RCP<class Teuchos::ParameterList> & a5){ return new PyCallBack_Tpetra_Import_int_long_long_Kokkos_Compat_KokkosDeviceWrapperNode_Kokkos_Serial_Kokkos_HostSpace_t(a0, a1, a2, a3, a4, a5); } ), "doc");
		cl.def( pybind11::init<const class Teuchos::RCP<const class Tpetra::Map<int, long long> > &, const class Teuchos::RCP<const class Tpetra::Map<int, long long> > &, const class Teuchos::ArrayView<int> &, const class Teuchos::ArrayView<const int> &, const class Teuchos::ArrayView<const int> &, const class Teuchos::RCP<class Teuchos::ParameterList> &, const class Teuchos::RCP<class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > > &>(), pybind11::arg("source"), pybind11::arg("target"), pybind11::arg("remotePIDs"), pybind11::arg("userExportLIDs"), pybind11::arg("userExportPIDs"), pybind11::arg("plist"), pybind11::arg("out") );

		cl.def("assign", (class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & (Tpetra::Import<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &)) &Tpetra::Import<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::operator=, "C++: Tpetra::Import<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::operator=(const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &) --> class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &", pybind11::return_value_policy::automatic, pybind11::arg("Source"));
		cl.def("setUnion", (class Teuchos::RCP<const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > > (Tpetra::Import<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &) const) &Tpetra::Import<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::setUnion, "C++: Tpetra::Import<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::setUnion(const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &) const --> class Teuchos::RCP<const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > >", pybind11::arg("rhs"));
		cl.def("setUnion", (class Teuchos::RCP<const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > > (Tpetra::Import<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)() const) &Tpetra::Import<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::setUnion, "C++: Tpetra::Import<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::setUnion() const --> class Teuchos::RCP<const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > >");
		cl.def("createRemoteOnlyImport", (class Teuchos::RCP<const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > > (Tpetra::Import<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Teuchos::RCP<const class Tpetra::Map<int, long long> > &) const) &Tpetra::Import<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::createRemoteOnlyImport, "C++: Tpetra::Import<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::createRemoteOnlyImport(const class Teuchos::RCP<const class Tpetra::Map<int, long long> > &) const --> class Teuchos::RCP<const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > >", pybind11::arg("remoteTarget"));
		cl.def("describe", [](Tpetra::Import<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> const &o, class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > & a0) -> void { return o.describe(a0); }, "", pybind11::arg("out"));
		cl.def("describe", (void (Tpetra::Import<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > &, const enum Teuchos::EVerbosityLevel) const) &Tpetra::Import<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::describe, "C++: Tpetra::Import<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::describe(class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > &, const enum Teuchos::EVerbosityLevel) const --> void", pybind11::arg("out"), pybind11::arg("verbLevel"));
	}
	{ // Tpetra::Export file:Tpetra_Export_decl.hpp line:117
		pybind11::class_<Tpetra::Export<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>, Teuchos::RCP<Tpetra::Export<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>>, PyCallBack_Tpetra_Export_int_long_long_Kokkos_Compat_KokkosDeviceWrapperNode_Kokkos_Serial_Kokkos_HostSpace_t> cl(M("Tpetra"), "Export_int_long_long_Kokkos_Compat_KokkosDeviceWrapperNode_Kokkos_Serial_Kokkos_HostSpace_t", "");
		cl.def( pybind11::init<const class Teuchos::RCP<const class Tpetra::Map<int, long long> > &, const class Teuchos::RCP<const class Tpetra::Map<int, long long> > &>(), pybind11::arg("source"), pybind11::arg("target") );

		cl.def( pybind11::init<const class Teuchos::RCP<const class Tpetra::Map<int, long long> > &, const class Teuchos::RCP<const class Tpetra::Map<int, long long> > &, const class Teuchos::RCP<class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > > &>(), pybind11::arg("source"), pybind11::arg("target"), pybind11::arg("out") );

		cl.def( pybind11::init<const class Teuchos::RCP<const class Tpetra::Map<int, long long> > &, const class Teuchos::RCP<const class Tpetra::Map<int, long long> > &, const class Teuchos::RCP<class Teuchos::ParameterList> &>(), pybind11::arg("source"), pybind11::arg("target"), pybind11::arg("plist") );

		cl.def( pybind11::init<const class Teuchos::RCP<const class Tpetra::Map<int, long long> > &, const class Teuchos::RCP<const class Tpetra::Map<int, long long> > &, const class Teuchos::RCP<class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > > &, const class Teuchos::RCP<class Teuchos::ParameterList> &>(), pybind11::arg("source"), pybind11::arg("target"), pybind11::arg("out"), pybind11::arg("plist") );

		cl.def( pybind11::init( [](PyCallBack_Tpetra_Export_int_long_long_Kokkos_Compat_KokkosDeviceWrapperNode_Kokkos_Serial_Kokkos_HostSpace_t const &o){ return new PyCallBack_Tpetra_Export_int_long_long_Kokkos_Compat_KokkosDeviceWrapperNode_Kokkos_Serial_Kokkos_HostSpace_t(o); } ) );
		cl.def( pybind11::init( [](Tpetra::Export<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> const &o){ return new Tpetra::Export<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>(o); } ) );
		cl.def( pybind11::init<const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &>(), pybind11::arg("importer") );

		cl.def("assign", (class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & (Tpetra::Export<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &)) &Tpetra::Export<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::operator=, "C++: Tpetra::Export<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::operator=(const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &) --> class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &", pybind11::return_value_policy::automatic, pybind11::arg("rhs"));
		cl.def("describe", [](Tpetra::Export<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> const &o, class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > & a0) -> void { return o.describe(a0); }, "", pybind11::arg("out"));
		cl.def("describe", (void (Tpetra::Export<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > &, const enum Teuchos::EVerbosityLevel) const) &Tpetra::Export<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::describe, "C++: Tpetra::Export<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::describe(class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > &, const enum Teuchos::EVerbosityLevel) const --> void", pybind11::arg("out"), pybind11::arg("verbLevel"));
	}
	{ // Tpetra::RowGraph file:Tpetra_RowGraph_decl.hpp line:69
		pybind11::class_<Tpetra::RowGraph<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>, Teuchos::RCP<Tpetra::RowGraph<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>>, Tpetra::Packable<long long,int>> cl(M("Tpetra"), "RowGraph_int_long_long_Kokkos_Compat_KokkosDeviceWrapperNode_Kokkos_Serial_Kokkos_HostSpace_t", "");
		cl.def("getComm", (class Teuchos::RCP<const class Teuchos::Comm<int> > (Tpetra::RowGraph<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)() const) &Tpetra::RowGraph<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::getComm, "C++: Tpetra::RowGraph<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::getComm() const --> class Teuchos::RCP<const class Teuchos::Comm<int> >");
		cl.def("getRowMap", (class Teuchos::RCP<const class Tpetra::Map<int, long long> > (Tpetra::RowGraph<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)() const) &Tpetra::RowGraph<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::getRowMap, "C++: Tpetra::RowGraph<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::getRowMap() const --> class Teuchos::RCP<const class Tpetra::Map<int, long long> >");
		cl.def("getColMap", (class Teuchos::RCP<const class Tpetra::Map<int, long long> > (Tpetra::RowGraph<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)() const) &Tpetra::RowGraph<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::getColMap, "C++: Tpetra::RowGraph<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::getColMap() const --> class Teuchos::RCP<const class Tpetra::Map<int, long long> >");
		cl.def("getDomainMap", (class Teuchos::RCP<const class Tpetra::Map<int, long long> > (Tpetra::RowGraph<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)() const) &Tpetra::RowGraph<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::getDomainMap, "C++: Tpetra::RowGraph<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::getDomainMap() const --> class Teuchos::RCP<const class Tpetra::Map<int, long long> >");
		cl.def("getRangeMap", (class Teuchos::RCP<const class Tpetra::Map<int, long long> > (Tpetra::RowGraph<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)() const) &Tpetra::RowGraph<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::getRangeMap, "C++: Tpetra::RowGraph<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::getRangeMap() const --> class Teuchos::RCP<const class Tpetra::Map<int, long long> >");
		cl.def("getImporter", (class Teuchos::RCP<const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > > (Tpetra::RowGraph<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)() const) &Tpetra::RowGraph<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::getImporter, "C++: Tpetra::RowGraph<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::getImporter() const --> class Teuchos::RCP<const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > >");
		cl.def("getExporter", (class Teuchos::RCP<const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > > (Tpetra::RowGraph<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)() const) &Tpetra::RowGraph<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::getExporter, "C++: Tpetra::RowGraph<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::getExporter() const --> class Teuchos::RCP<const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > >");
		cl.def("getGlobalNumRows", (unsigned long (Tpetra::RowGraph<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)() const) &Tpetra::RowGraph<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::getGlobalNumRows, "C++: Tpetra::RowGraph<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::getGlobalNumRows() const --> unsigned long");
		cl.def("getGlobalNumCols", (unsigned long (Tpetra::RowGraph<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)() const) &Tpetra::RowGraph<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::getGlobalNumCols, "C++: Tpetra::RowGraph<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::getGlobalNumCols() const --> unsigned long");
		cl.def("getLocalNumRows", (unsigned long (Tpetra::RowGraph<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)() const) &Tpetra::RowGraph<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::getLocalNumRows, "C++: Tpetra::RowGraph<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::getLocalNumRows() const --> unsigned long");
		cl.def("getLocalNumCols", (unsigned long (Tpetra::RowGraph<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)() const) &Tpetra::RowGraph<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::getLocalNumCols, "C++: Tpetra::RowGraph<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::getLocalNumCols() const --> unsigned long");
		cl.def("getIndexBase", (long long (Tpetra::RowGraph<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)() const) &Tpetra::RowGraph<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::getIndexBase, "C++: Tpetra::RowGraph<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::getIndexBase() const --> long long");
		cl.def("getGlobalNumEntries", (unsigned long (Tpetra::RowGraph<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)() const) &Tpetra::RowGraph<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::getGlobalNumEntries, "C++: Tpetra::RowGraph<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::getGlobalNumEntries() const --> unsigned long");
		cl.def("getLocalNumEntries", (unsigned long (Tpetra::RowGraph<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)() const) &Tpetra::RowGraph<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::getLocalNumEntries, "C++: Tpetra::RowGraph<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::getLocalNumEntries() const --> unsigned long");
		cl.def("getNumEntriesInGlobalRow", (unsigned long (Tpetra::RowGraph<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(long long) const) &Tpetra::RowGraph<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::getNumEntriesInGlobalRow, "C++: Tpetra::RowGraph<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::getNumEntriesInGlobalRow(long long) const --> unsigned long", pybind11::arg("globalRow"));
		cl.def("getNumEntriesInLocalRow", (unsigned long (Tpetra::RowGraph<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(int) const) &Tpetra::RowGraph<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::getNumEntriesInLocalRow, "C++: Tpetra::RowGraph<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::getNumEntriesInLocalRow(int) const --> unsigned long", pybind11::arg("localRow"));
		cl.def("getGlobalMaxNumRowEntries", (unsigned long (Tpetra::RowGraph<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)() const) &Tpetra::RowGraph<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::getGlobalMaxNumRowEntries, "C++: Tpetra::RowGraph<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::getGlobalMaxNumRowEntries() const --> unsigned long");
		cl.def("getLocalMaxNumRowEntries", (unsigned long (Tpetra::RowGraph<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)() const) &Tpetra::RowGraph<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::getLocalMaxNumRowEntries, "C++: Tpetra::RowGraph<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::getLocalMaxNumRowEntries() const --> unsigned long");
		cl.def("hasColMap", (bool (Tpetra::RowGraph<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)() const) &Tpetra::RowGraph<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::hasColMap, "C++: Tpetra::RowGraph<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::hasColMap() const --> bool");
		cl.def("isLocallyIndexed", (bool (Tpetra::RowGraph<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)() const) &Tpetra::RowGraph<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::isLocallyIndexed, "C++: Tpetra::RowGraph<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::isLocallyIndexed() const --> bool");
		cl.def("isGloballyIndexed", (bool (Tpetra::RowGraph<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)() const) &Tpetra::RowGraph<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::isGloballyIndexed, "C++: Tpetra::RowGraph<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::isGloballyIndexed() const --> bool");
		cl.def("isFillComplete", (bool (Tpetra::RowGraph<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)() const) &Tpetra::RowGraph<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::isFillComplete, "C++: Tpetra::RowGraph<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::isFillComplete() const --> bool");
		cl.def("supportsRowViews", (bool (Tpetra::RowGraph<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)() const) &Tpetra::RowGraph<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::supportsRowViews, "C++: Tpetra::RowGraph<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::supportsRowViews() const --> bool");
		cl.def("pack", (void (Tpetra::RowGraph<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Teuchos::ArrayView<const int> &, class Teuchos::Array<long long> &, const class Teuchos::ArrayView<unsigned long> &, unsigned long &) const) &Tpetra::RowGraph<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::pack, "C++: Tpetra::RowGraph<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::pack(const class Teuchos::ArrayView<const int> &, class Teuchos::Array<long long> &, const class Teuchos::ArrayView<unsigned long> &, unsigned long &) const --> void", pybind11::arg("exportLIDs"), pybind11::arg("exports"), pybind11::arg("numPacketsPerLID"), pybind11::arg("constantNumPackets"));
		cl.def("assign", (class Tpetra::RowGraph<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & (Tpetra::RowGraph<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::RowGraph<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &)) &Tpetra::RowGraph<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::operator=, "C++: Tpetra::RowGraph<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::operator=(const class Tpetra::RowGraph<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &) --> class Tpetra::RowGraph<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &", pybind11::return_value_policy::automatic, pybind11::arg(""));
		cl.def("pack", (void (Tpetra::Packable<long long,int>::*)(const class Teuchos::ArrayView<const int> &, class Teuchos::Array<long long> &, const class Teuchos::ArrayView<unsigned long> &, unsigned long &) const) &Tpetra::Packable<long long, int>::pack, "C++: Tpetra::Packable<long long, int>::pack(const class Teuchos::ArrayView<const int> &, class Teuchos::Array<long long> &, const class Teuchos::ArrayView<unsigned long> &, unsigned long &) const --> void", pybind11::arg("exportLIDs"), pybind11::arg("exports"), pybind11::arg("numPacketsPerLID"), pybind11::arg("constantNumPackets"));
		cl.def("assign", (class Tpetra::Packable<long long, int> & (Tpetra::Packable<long long,int>::*)(const class Tpetra::Packable<long long, int> &)) &Tpetra::Packable<long long, int>::operator=, "C++: Tpetra::Packable<long long, int>::operator=(const class Tpetra::Packable<long long, int> &) --> class Tpetra::Packable<long long, int> &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // Tpetra::Distributor file:Tpetra_Distributor.hpp line:132
		pybind11::class_<Tpetra::Distributor, Teuchos::RCP<Tpetra::Distributor>, PyCallBack_Tpetra_Distributor, Teuchos::ParameterListAcceptorDefaultBase> cl(M("Tpetra"), "Distributor", "Sets up and executes a communication plan for a Tpetra DistObject.\n\n \n Most Tpetra users do not need to know about this class.\n\n This class encapsulates the general information and\n communication services needed for subclasses of \n (such as CrsMatrix and MultiVector) to do data redistribution\n (Import and Export) operations.  It is an implementation detail\n of Import and Export; in particular; it actually does the\n communication.\n\n Here is the typical way to use this class:\n 1. Create a Distributor.  (The constructor is inexpensive.)\n 2. Set up the Distributor once, using one of the two \"plan\n    creation\" methods: either createFromSends(), or\n    createFromRecvs().  This may be more expensive and\n    communication-intensive than Step 3.\n 3. Communicate the data by calling doPostsAndWaits() (forward\n    mode), or doReversePostsAndWaits() (reverse mode).  You may\n    do this multiple times with the same Distributor instance.\n\n Step 2 is expensive, but you can amortize its cost over multiple\n uses of the Distributor for communication (Step 3).  You may\n also separate out \"posts\" (invoking nonblocking communication)\n and \"waits\" (waiting for that communication to complete), by\n calling doPosts() (resp. doReversePosts()), then doWaits()\n (resp. doReverseWaits()).  This is useful if you have local work\n to do between the posts and waits, because it may overlap\n communication with computation.  Whether it actually does\n overlap, depends on both the MPI implementation and your choice\n of parameters for the Distributor.\n\n Instances of Distributor take the following parameters that\n control communication and debug output:\n - \"Send type\" (std::string): When using MPI, the\n   variant of MPI_Send to use in do[Reverse]Posts().  Valid\n   values include \"Isend\", \n   and \"Send\".  The\n   default is \"Send\".  (The receive type is always MPI_Irecv, a\n   nonblocking receive.  Since we post receives first before\n   sends, this prevents deadlock, even if MPI_Send blocks and\n   does not buffer.)\n - \"Debug\" ( If true, print copious debugging output on\n   all processes in the Distributor's communicator.  This is\n   useful only for debugging Distributor and other Tpetra classes\n   that use it (like Import and Export).  If the Distributor was\n   created using one of the constructors that takes a\n   Teuchos::FancyOStream, it will write debugging output to that\n   stream.  Otherwise, it will write debugging output to stderr.\n   Currently, the \"Debug\" parameter overrides \"VerboseObject\"\n   (see below).\n - \"VerboseObject\" (sublist): Optional sublist for controlling\n   behavior of Distributor as a Teuchos::VerboseObject.  This is\n   currently useful only for debugging.  This sublist takes\n   optional parameters \"Verbosity Level\" (std::string) and\n   \"Output File\" (std::string).  \"Verbosity Level\" has six valid\n   values: \"VERB_DEFAULT\", \"VERB_NONE\", \"VERB_LOW\",\n   \"VERB_MEDIUM\", \"VERB_HIGH\", and \"VERB_EXTREME\", with\n   increasing verbosity starting with \"VERB_NONE\".  \"Output File\"\n   is the name of a file to use for output; \"none\" means don't\n   open a file, but write to the default output stream.");
		cl.def( pybind11::init<const class Teuchos::RCP<const class Teuchos::Comm<int> > &>(), pybind11::arg("comm") );

		cl.def( pybind11::init<const class Teuchos::RCP<const class Teuchos::Comm<int> > &, const class Teuchos::RCP<class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > > &>(), pybind11::arg("comm"), pybind11::arg("out") );

		cl.def( pybind11::init<const class Teuchos::RCP<const class Teuchos::Comm<int> > &, const class Teuchos::RCP<class Teuchos::ParameterList> &>(), pybind11::arg("comm"), pybind11::arg("plist") );

		cl.def( pybind11::init<const class Teuchos::RCP<const class Teuchos::Comm<int> > &, const class Teuchos::RCP<class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > > &, const class Teuchos::RCP<class Teuchos::ParameterList> &>(), pybind11::arg("comm"), pybind11::arg("out"), pybind11::arg("plist") );

		cl.def( pybind11::init( [](PyCallBack_Tpetra_Distributor const &o){ return new PyCallBack_Tpetra_Distributor(o); } ) );
		cl.def( pybind11::init( [](Tpetra::Distributor const &o){ return new Tpetra::Distributor(o); } ) );
		cl.def("swap", (void (Tpetra::Distributor::*)(class Tpetra::Distributor &)) &Tpetra::Distributor::swap, "Swap the contents of rhs with those of *this.\n\n This is useful in Import's setUnion() method.  It avoids the\n overhead of copying arrays, since it can use std::swap on the\n arrays.\n\nC++: Tpetra::Distributor::swap(class Tpetra::Distributor &) --> void", pybind11::arg("rhs"));
		cl.def("setParameterList", (void (Tpetra::Distributor::*)(const class Teuchos::RCP<class Teuchos::ParameterList> &)) &Tpetra::Distributor::setParameterList, "Set Distributor parameters.\n\n Please see the class documentation for a list of all accepted\n parameters and their default values.\n\nC++: Tpetra::Distributor::setParameterList(const class Teuchos::RCP<class Teuchos::ParameterList> &) --> void", pybind11::arg("plist"));
		cl.def("getValidParameters", (class Teuchos::RCP<const class Teuchos::ParameterList> (Tpetra::Distributor::*)() const) &Tpetra::Distributor::getValidParameters, "List of valid Distributor parameters.\n\n Please see the class documentation for a list of all accepted\n parameters and their default values.\n\nC++: Tpetra::Distributor::getValidParameters() const --> class Teuchos::RCP<const class Teuchos::ParameterList>");
		cl.def("createFromSends", (unsigned long (Tpetra::Distributor::*)(const class Teuchos::ArrayView<const int> &)) &Tpetra::Distributor::createFromSends, "Set up Distributor using list of process ranks to which\n   this process will send.\n\n Take a list of process ranks and construct a plan for\n efficiently scattering to those processes.  Return the number\n of processes which will send me (the calling process) data.\n\n \n [in] List of ranks of the processes that\n   will get the exported data.  If there is a process rank\n   greater than or equal to the number of processes, all\n   processes will throw an std::runtime_error\n   exception.  Process ranks less than zero are ignored; their\n   placement corresponds to null sends in any future\n   exports. That is, if exportProcIDs[0] == -1, then\n   the corresponding position in the export array is ignored\n   during a call to doPosts() or doPostsAndWaits().  For this\n   reason, a negative entry is sufficient to break contiguity.\n\n \n Number of imports this process will be receiving.\n\nC++: Tpetra::Distributor::createFromSends(const class Teuchos::ArrayView<const int> &) --> unsigned long", pybind11::arg("exportProcIDs"));
		cl.def("createFromSendsAndRecvs", (void (Tpetra::Distributor::*)(const class Teuchos::ArrayView<const int> &, const class Teuchos::ArrayView<const int> &)) &Tpetra::Distributor::createFromSendsAndRecvs, "Set up Distributor using list of process ranks to which\n   to send, and list of process ranks from which to receive.\n\n \n [in] List of process ranks to which this\n   process must send a message.\n \n\n [in] List of process ranks from which\n   this process must receive a message.\n\nC++: Tpetra::Distributor::createFromSendsAndRecvs(const class Teuchos::ArrayView<const int> &, const class Teuchos::ArrayView<const int> &) --> void", pybind11::arg("exportProcIDs"), pybind11::arg("remoteProcIDs"));
		cl.def("getNumReceives", (unsigned long (Tpetra::Distributor::*)() const) &Tpetra::Distributor::getNumReceives, "The number of processes from which we will receive data.\n\n The count does not include the calling process.\n\nC++: Tpetra::Distributor::getNumReceives() const --> unsigned long");
		cl.def("getNumSends", (unsigned long (Tpetra::Distributor::*)() const) &Tpetra::Distributor::getNumSends, "The number of processes to which we will send data.\n\n The count does not include the calling process.\n\nC++: Tpetra::Distributor::getNumSends() const --> unsigned long");
		cl.def("hasSelfMessage", (bool (Tpetra::Distributor::*)() const) &Tpetra::Distributor::hasSelfMessage, "Whether the calling process will send or receive messages to itself.\n\nC++: Tpetra::Distributor::hasSelfMessage() const --> bool");
		cl.def("getMaxSendLength", (unsigned long (Tpetra::Distributor::*)() const) &Tpetra::Distributor::getMaxSendLength, "Maximum number of values this process will send to another single process.\n\nC++: Tpetra::Distributor::getMaxSendLength() const --> unsigned long");
		cl.def("getTotalReceiveLength", (unsigned long (Tpetra::Distributor::*)() const) &Tpetra::Distributor::getTotalReceiveLength, "Total number of values this process will receive from other processes.\n\nC++: Tpetra::Distributor::getTotalReceiveLength() const --> unsigned long");
		cl.def("getProcsFrom", (class Teuchos::ArrayView<const int> (Tpetra::Distributor::*)() const) &Tpetra::Distributor::getProcsFrom, "Ranks of the processes sending values to this process.\n\n This is a nonpersisting view.  It will last only as long as\n this Distributor instance does.\n\nC++: Tpetra::Distributor::getProcsFrom() const --> class Teuchos::ArrayView<const int>");
		cl.def("getProcsTo", (class Teuchos::ArrayView<const int> (Tpetra::Distributor::*)() const) &Tpetra::Distributor::getProcsTo, "Ranks of the processes to which this process will send values.\n\n This is a nonpersisting view.  It will last only as long as\n this Distributor instance does.\n\nC++: Tpetra::Distributor::getProcsTo() const --> class Teuchos::ArrayView<const int>");
		cl.def("getLengthsFrom", (class Teuchos::ArrayView<const unsigned long> (Tpetra::Distributor::*)() const) &Tpetra::Distributor::getLengthsFrom, "Number of values this process will receive from each process.\n\n This process will receive getLengthsFrom[i] values\n from process getProcsFrom[i].\n\n This is a nonpersisting view.  It will last only as long as\n this Distributor instance does.\n\nC++: Tpetra::Distributor::getLengthsFrom() const --> class Teuchos::ArrayView<const unsigned long>");
		cl.def("getLengthsTo", (class Teuchos::ArrayView<const unsigned long> (Tpetra::Distributor::*)() const) &Tpetra::Distributor::getLengthsTo, "Number of values this process will send to each process.\n\n This process will send getLengthsTo[i] values to\n process getProcsTo[i].\n\n This is a nonpersisting view.  It will last only as long as\n this Distributor instance does.\n\nC++: Tpetra::Distributor::getLengthsTo() const --> class Teuchos::ArrayView<const unsigned long>");
		cl.def("howInitialized", (enum Tpetra::Details::EDistributorHowInitialized (Tpetra::Distributor::*)() const) &Tpetra::Distributor::howInitialized, "Return an enum indicating whether and how a Distributor was initialized.\n\n This is an implementation detail of Tpetra.  Please do not\n call this method or rely on it existing in your code.\n\nC++: Tpetra::Distributor::howInitialized() const --> enum Tpetra::Details::EDistributorHowInitialized");
		cl.def("getReverse", [](Tpetra::Distributor const &o) -> Teuchos::RCP<class Tpetra::Distributor> { return o.getReverse(); }, "");
		cl.def("getReverse", (class Teuchos::RCP<class Tpetra::Distributor> (Tpetra::Distributor::*)(bool) const) &Tpetra::Distributor::getReverse, "A reverse communication plan Distributor.\n\n The first time this method is called, it creates a Distributor\n with the reverse communication plan of *this.  On\n subsequent calls, it returns the cached reverse Distributor.\n\n Most users do not need to call this method.  If you invoke\n doReversePosts() or doReversePostsAndWaits(), the reverse\n Distributor will be created automatically if it does not yet\n exist.\n\nC++: Tpetra::Distributor::getReverse(bool) const --> class Teuchos::RCP<class Tpetra::Distributor>", pybind11::arg("create"));
		cl.def("doWaits", (void (Tpetra::Distributor::*)()) &Tpetra::Distributor::doWaits, "Wait on any outstanding nonblocking message requests to complete.\n\n This method is for forward mode communication only, that is,\n after calling doPosts().  For reverse mode communication\n (after calling doReversePosts()), call doReverseWaits()\n instead.\n\nC++: Tpetra::Distributor::doWaits() --> void");
		cl.def("doReverseWaits", (void (Tpetra::Distributor::*)()) &Tpetra::Distributor::doReverseWaits, "Wait on any outstanding nonblocking message requests to complete.\n\n This method is for reverse mode communication only, that is,\n after calling doReversePosts().  For forward mode\n communication (after calling doPosts()), call doWaits()\n instead.\n\nC++: Tpetra::Distributor::doReverseWaits() --> void");
		cl.def("description", (std::string (Tpetra::Distributor::*)() const) &Tpetra::Distributor::description, "Return a one-line description of this object.\n\nC++: Tpetra::Distributor::description() const --> std::string");
		cl.def("describe", [](Tpetra::Distributor const &o, class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > & a0) -> void { return o.describe(a0); }, "", pybind11::arg("out"));
		cl.def("describe", (void (Tpetra::Distributor::*)(class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > &, const enum Teuchos::EVerbosityLevel) const) &Tpetra::Distributor::describe, "Describe this object in a human-readable way to the\n   given output stream.\n\n You must call this method as a collective over all processes\n in this object's communicator.\n\n \n [out] Output stream to which to write.  Only\n   Process 0 in this object's communicator may write to the\n   output stream.\n\n \n [in] Verbosity level.  This also controls\n   whether this method does any communication.  At verbosity\n   levels higher (greater) than Teuchos::VERB_LOW, this method\n   behaves as a collective over the object's communicator.\n\n Teuchos::FancyOStream wraps std::ostream.  It adds features\n like tab levels.  If you just want to wrap std::cout, try\n this:\n \n\n\n\nC++: Tpetra::Distributor::describe(class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > &, const enum Teuchos::EVerbosityLevel) const --> void", pybind11::arg("out"), pybind11::arg("verbLevel"));
		cl.def("assign", (class Tpetra::Distributor & (Tpetra::Distributor::*)(const class Tpetra::Distributor &)) &Tpetra::Distributor::operator=, "C++: Tpetra::Distributor::operator=(const class Tpetra::Distributor &) --> class Tpetra::Distributor &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
}
