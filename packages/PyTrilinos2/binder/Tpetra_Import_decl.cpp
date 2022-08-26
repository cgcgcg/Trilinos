#include <KokkosCompat_ClassicNodeAPI_Wrapper.hpp> // Kokkos::Compat::KokkosDeviceWrapperNode
#include <Kokkos_HostSpace.hpp> // Kokkos::HostSpace
#include <Kokkos_Serial.hpp> // Kokkos::Serial
#include <Teuchos_Array.hpp> // Teuchos::Array
#include <Teuchos_ArrayViewDecl.hpp> // Teuchos::ArrayView
#include <Teuchos_Describable.hpp> // Teuchos::Describable
#include <Teuchos_ENull.hpp> // Teuchos::ENull
#include <Teuchos_FancyOStream.hpp> // Teuchos::basic_FancyOStream
#include <Teuchos_LabeledObject.hpp> // Teuchos::LabeledObject
#include <Teuchos_ParameterList.hpp> // Teuchos::ParameterList
#include <Teuchos_PtrDecl.hpp> // Teuchos::Ptr
#include <Teuchos_RCPDecl.hpp> // Teuchos::ERCPUndefinedWeakNoDealloc
#include <Teuchos_RCPDecl.hpp> // Teuchos::ERCPWeakNoDealloc
#include <Teuchos_RCPDecl.hpp> // Teuchos::RCP
#include <Teuchos_RCPNode.hpp> // Teuchos::ERCPNodeLookup
#include <Teuchos_RCPNode.hpp> // Teuchos::ERCPStrength
#include <Teuchos_RCPNode.hpp> // Teuchos::RCPNodeHandle
#include <Teuchos_VerbosityLevel.hpp> // Teuchos::EVerbosityLevel
#include <Tpetra_CombineMode.hpp> // Tpetra::CombineMode
#include <Tpetra_DistObject_decl.hpp> // Tpetra::DistObject
#include <Tpetra_Export_decl.hpp> // Tpetra::Export
#include <Tpetra_Import_decl.hpp> // Tpetra::Import
#include <Tpetra_SrcDistObject.hpp> // Tpetra::SrcDistObject
#include <cwchar> // (anonymous)
#include <ios> // std::_Ios_Seekdir
#include <ios> // std::fpos
#include <iterator> // __gnu_cxx::__normal_iterator
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
}
