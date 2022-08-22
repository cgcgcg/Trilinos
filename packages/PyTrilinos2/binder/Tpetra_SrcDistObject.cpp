#include <KokkosCompat_ClassicNodeAPI_Wrapper.hpp>
#include <Kokkos_Concepts.hpp>
#include <Kokkos_HostSpace.hpp>
#include <Kokkos_Serial.hpp>
#include <Kokkos_View.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_ArrayViewDecl.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_ENull.hpp>
#include <Teuchos_FancyOStream.hpp>
#include <Teuchos_LabeledObject.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_PtrDecl.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <Teuchos_RCPNode.hpp>
#include <Teuchos_VerbosityLevel.hpp>
#include <Teuchos_any.hpp>
#include <Tpetra_CombineMode.hpp>
#include <Tpetra_ConfigDefs.hpp>
#include <Tpetra_Details_LocalMap.hpp>
#include <Tpetra_DistObject_decl.hpp>
#include <Tpetra_Export_decl.hpp>
#include <Tpetra_Import_decl.hpp>
#include <Tpetra_Map_decl.hpp>
#include <Tpetra_SrcDistObject.hpp>
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

// Tpetra::DistObject file:Tpetra_DistObject_decl.hpp line:321
struct PyCallBack_Tpetra_DistObject_double_int_long_long_Kokkos_Compat_KokkosDeviceWrapperNode_Kokkos_Serial_Kokkos_HostSpace_t : public Tpetra::DistObject<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> {
	using Tpetra::DistObject<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::DistObject;

	using _binder_ret_0 = class Teuchos::RCP<const class Tpetra::Map<int, long long> >;
	_binder_ret_0 getMap() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::DistObject<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> *>(this), "getMap");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_0>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_0> caster;
				return pybind11::detail::cast_ref<_binder_ret_0>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_0>(std::move(o));
		}
		return DistObject::getMap();
	}
	std::string description() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::DistObject<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> *>(this), "description");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<std::string>::value) {
				static pybind11::detail::override_caster_t<std::string> caster;
				return pybind11::detail::cast_ref<std::string>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<std::string>(std::move(o));
		}
		return DistObject::description();
	}
	void describe(class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > & a0, const enum Teuchos::EVerbosityLevel a1) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::DistObject<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> *>(this), "describe");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return DistObject::describe(a0, a1);
	}
	void removeEmptyProcessesInPlace(const class Teuchos::RCP<const class Tpetra::Map<int, long long> > & a0) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::DistObject<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> *>(this), "removeEmptyProcessesInPlace");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return DistObject::removeEmptyProcessesInPlace(a0);
	}
	unsigned long constantNumberOfPackets() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::DistObject<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> *>(this), "constantNumberOfPackets");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned long>::value) {
				static pybind11::detail::override_caster_t<unsigned long> caster;
				return pybind11::detail::cast_ref<unsigned long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned long>(std::move(o));
		}
		return DistObject::constantNumberOfPackets();
	}
	bool reallocArraysForNumPacketsPerLid(const unsigned long a0, const unsigned long a1) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::DistObject<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> *>(this), "reallocArraysForNumPacketsPerLid");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return DistObject::reallocArraysForNumPacketsPerLid(a0, a1);
	}
	bool checkSizes(const class Tpetra::SrcDistObject & a0) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::DistObject<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> *>(this), "checkSizes");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		pybind11::pybind11_fail("Tried to call pure virtual function \"DistObject::checkSizes\"");
	}
	bool reallocImportsIfNeeded(const unsigned long a0, const bool a1, const std::string * a2, const bool a3, const enum Tpetra::CombineMode a4) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::DistObject<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> *>(this), "reallocImportsIfNeeded");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2, a3, a4);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return DistObject::reallocImportsIfNeeded(a0, a1, a2, a3, a4);
	}
	void setObjectLabel(const std::string & a0) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::DistObject<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> *>(this), "setObjectLabel");
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
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::DistObject<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> *>(this), "getObjectLabel");
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

// Tpetra::DistObject file:Tpetra_DistObject_decl.hpp line:321
struct PyCallBack_Tpetra_DistObject_long_long_int_long_long_Kokkos_Compat_KokkosDeviceWrapperNode_Kokkos_Serial_Kokkos_HostSpace_t : public Tpetra::DistObject<long long,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> {
	using Tpetra::DistObject<long long,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::DistObject;

	using _binder_ret_0 = class Teuchos::RCP<const class Tpetra::Map<int, long long> >;
	_binder_ret_0 getMap() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::DistObject<long long,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> *>(this), "getMap");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_0>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_0> caster;
				return pybind11::detail::cast_ref<_binder_ret_0>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_0>(std::move(o));
		}
		return DistObject::getMap();
	}
	std::string description() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::DistObject<long long,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> *>(this), "description");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<std::string>::value) {
				static pybind11::detail::override_caster_t<std::string> caster;
				return pybind11::detail::cast_ref<std::string>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<std::string>(std::move(o));
		}
		return DistObject::description();
	}
	void describe(class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > & a0, const enum Teuchos::EVerbosityLevel a1) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::DistObject<long long,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> *>(this), "describe");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return DistObject::describe(a0, a1);
	}
	void removeEmptyProcessesInPlace(const class Teuchos::RCP<const class Tpetra::Map<int, long long> > & a0) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::DistObject<long long,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> *>(this), "removeEmptyProcessesInPlace");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return DistObject::removeEmptyProcessesInPlace(a0);
	}
	unsigned long constantNumberOfPackets() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::DistObject<long long,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> *>(this), "constantNumberOfPackets");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned long>::value) {
				static pybind11::detail::override_caster_t<unsigned long> caster;
				return pybind11::detail::cast_ref<unsigned long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned long>(std::move(o));
		}
		return DistObject::constantNumberOfPackets();
	}
	bool reallocArraysForNumPacketsPerLid(const unsigned long a0, const unsigned long a1) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::DistObject<long long,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> *>(this), "reallocArraysForNumPacketsPerLid");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return DistObject::reallocArraysForNumPacketsPerLid(a0, a1);
	}
	bool checkSizes(const class Tpetra::SrcDistObject & a0) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::DistObject<long long,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> *>(this), "checkSizes");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		pybind11::pybind11_fail("Tried to call pure virtual function \"DistObject::checkSizes\"");
	}
	bool reallocImportsIfNeeded(const unsigned long a0, const bool a1, const std::string * a2, const bool a3, const enum Tpetra::CombineMode a4) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::DistObject<long long,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> *>(this), "reallocImportsIfNeeded");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2, a3, a4);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return DistObject::reallocImportsIfNeeded(a0, a1, a2, a3, a4);
	}
	void setObjectLabel(const std::string & a0) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::DistObject<long long,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> *>(this), "setObjectLabel");
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
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::DistObject<long long,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> *>(this), "getObjectLabel");
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

// Tpetra::DistObject file:Tpetra_DistObject_decl.hpp line:321
struct PyCallBack_Tpetra_DistObject_int_int_long_long_Kokkos_Compat_KokkosDeviceWrapperNode_Kokkos_Serial_Kokkos_HostSpace_t : public Tpetra::DistObject<int,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> {
	using Tpetra::DistObject<int,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::DistObject;

	using _binder_ret_0 = class Teuchos::RCP<const class Tpetra::Map<int, long long> >;
	_binder_ret_0 getMap() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::DistObject<int,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> *>(this), "getMap");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_0>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_0> caster;
				return pybind11::detail::cast_ref<_binder_ret_0>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_0>(std::move(o));
		}
		return DistObject::getMap();
	}
	std::string description() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::DistObject<int,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> *>(this), "description");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<std::string>::value) {
				static pybind11::detail::override_caster_t<std::string> caster;
				return pybind11::detail::cast_ref<std::string>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<std::string>(std::move(o));
		}
		return DistObject::description();
	}
	void describe(class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > & a0, const enum Teuchos::EVerbosityLevel a1) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::DistObject<int,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> *>(this), "describe");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return DistObject::describe(a0, a1);
	}
	void removeEmptyProcessesInPlace(const class Teuchos::RCP<const class Tpetra::Map<int, long long> > & a0) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::DistObject<int,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> *>(this), "removeEmptyProcessesInPlace");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return DistObject::removeEmptyProcessesInPlace(a0);
	}
	unsigned long constantNumberOfPackets() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::DistObject<int,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> *>(this), "constantNumberOfPackets");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned long>::value) {
				static pybind11::detail::override_caster_t<unsigned long> caster;
				return pybind11::detail::cast_ref<unsigned long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned long>(std::move(o));
		}
		return DistObject::constantNumberOfPackets();
	}
	bool reallocArraysForNumPacketsPerLid(const unsigned long a0, const unsigned long a1) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::DistObject<int,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> *>(this), "reallocArraysForNumPacketsPerLid");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return DistObject::reallocArraysForNumPacketsPerLid(a0, a1);
	}
	bool checkSizes(const class Tpetra::SrcDistObject & a0) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::DistObject<int,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> *>(this), "checkSizes");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		pybind11::pybind11_fail("Tried to call pure virtual function \"DistObject::checkSizes\"");
	}
	bool reallocImportsIfNeeded(const unsigned long a0, const bool a1, const std::string * a2, const bool a3, const enum Tpetra::CombineMode a4) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::DistObject<int,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> *>(this), "reallocImportsIfNeeded");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2, a3, a4);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return DistObject::reallocImportsIfNeeded(a0, a1, a2, a3, a4);
	}
	void setObjectLabel(const std::string & a0) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::DistObject<int,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> *>(this), "setObjectLabel");
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
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::DistObject<int,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> *>(this), "getObjectLabel");
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

// Tpetra::DistObject file:Tpetra_DistObject_decl.hpp line:321
struct PyCallBack_Tpetra_DistObject_char_int_long_long_Kokkos_Compat_KokkosDeviceWrapperNode_Kokkos_Serial_Kokkos_HostSpace_t : public Tpetra::DistObject<char,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> {
	using Tpetra::DistObject<char,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::DistObject;

	using _binder_ret_0 = class Teuchos::RCP<const class Tpetra::Map<int, long long> >;
	_binder_ret_0 getMap() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::DistObject<char,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> *>(this), "getMap");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_0>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_0> caster;
				return pybind11::detail::cast_ref<_binder_ret_0>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_0>(std::move(o));
		}
		return DistObject::getMap();
	}
	std::string description() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::DistObject<char,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> *>(this), "description");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<std::string>::value) {
				static pybind11::detail::override_caster_t<std::string> caster;
				return pybind11::detail::cast_ref<std::string>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<std::string>(std::move(o));
		}
		return DistObject::description();
	}
	void describe(class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > & a0, const enum Teuchos::EVerbosityLevel a1) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::DistObject<char,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> *>(this), "describe");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return DistObject::describe(a0, a1);
	}
	void removeEmptyProcessesInPlace(const class Teuchos::RCP<const class Tpetra::Map<int, long long> > & a0) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::DistObject<char,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> *>(this), "removeEmptyProcessesInPlace");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return DistObject::removeEmptyProcessesInPlace(a0);
	}
	unsigned long constantNumberOfPackets() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::DistObject<char,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> *>(this), "constantNumberOfPackets");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned long>::value) {
				static pybind11::detail::override_caster_t<unsigned long> caster;
				return pybind11::detail::cast_ref<unsigned long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned long>(std::move(o));
		}
		return DistObject::constantNumberOfPackets();
	}
	bool reallocArraysForNumPacketsPerLid(const unsigned long a0, const unsigned long a1) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::DistObject<char,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> *>(this), "reallocArraysForNumPacketsPerLid");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return DistObject::reallocArraysForNumPacketsPerLid(a0, a1);
	}
	bool checkSizes(const class Tpetra::SrcDistObject & a0) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::DistObject<char,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> *>(this), "checkSizes");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		pybind11::pybind11_fail("Tried to call pure virtual function \"DistObject::checkSizes\"");
	}
	bool reallocImportsIfNeeded(const unsigned long a0, const bool a1, const std::string * a2, const bool a3, const enum Tpetra::CombineMode a4) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::DistObject<char,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> *>(this), "reallocImportsIfNeeded");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2, a3, a4);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return DistObject::reallocImportsIfNeeded(a0, a1, a2, a3, a4);
	}
	void setObjectLabel(const std::string & a0) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::DistObject<char,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> *>(this), "setObjectLabel");
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
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::DistObject<char,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> *>(this), "getObjectLabel");
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

void bind_Tpetra_SrcDistObject(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // Tpetra::SrcDistObject file:Tpetra_SrcDistObject.hpp line:89
		pybind11::class_<Tpetra::SrcDistObject, Teuchos::RCP<Tpetra::SrcDistObject>> cl(M("Tpetra"), "SrcDistObject", "Abstract base class for objects that can be the source of\n   an Import or Export operation.\n\n Any object that may be the source of an Import or Export data\n redistribution operation must inherit from this class.  This\n class implements no methods, other than a trivial virtual\n destructor.  If a subclass X inherits from this class, that\n indicates that the subclass can be the source of an Import or\n Export, for some set of subclasses of DistObject.  A\n subclass Y of DistObject which is the target of the Import or\n Export operation will attempt to cast the input source\n SrcDistObject to a subclass which it knows how to treat as a\n source object.  The target subclass Y is responsible for knowing\n what source classes to expect, and how to interpret the\n resulting source object.\n\n DistObject inherits from this class, since a DistObject subclass\n may be either the source or the target of an Import or Export.\n A SrcDistObject subclass which does not inherit from DistObject\n need only be a valid source of an Import or Export; it need not\n be a valid target.\n\n This object compares to the Epetra class Epetra_SrcDistObject.\n Unlike in Epetra, this class in Tpetra does not include\n a getMap() method.  This is for two reasons.  First, consider\n the following inheritance hierarchy: DistObject and RowGraph\n inherit from SrcDistObject, and CrsGraph inherits from\n DistObject and RowGraph.  If SrcDistObject had a virtual getMap\n method, that would make resolution of the method ambiguous.\n Second, it is not necessary for SrcDistObject to have a getMap\n method, because a SrcDistObject alone does not suffice as the\n source of an Import or Export.  Any DistObject subclass must\n cast the SrcDistObject to a subclass which it knows how to treat\n as the source of an Import or Export.  Thus, it's not necessary\n for SrcDistObject to have a getMap method, since it needs to be\n cast anyway before use.  In general, I prefer to keep interfaces\n as simple as possible.");
		cl.def( pybind11::init( [](Tpetra::SrcDistObject const &o){ return new Tpetra::SrcDistObject(o); } ) );
		cl.def( pybind11::init( [](){ return new Tpetra::SrcDistObject(); } ) );
	}
	{ // Tpetra::DistObject file:Tpetra_DistObject_decl.hpp line:321
		pybind11::class_<Tpetra::DistObject<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>, Teuchos::RCP<Tpetra::DistObject<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>>, PyCallBack_Tpetra_DistObject_double_int_long_long_Kokkos_Compat_KokkosDeviceWrapperNode_Kokkos_Serial_Kokkos_HostSpace_t, Tpetra::SrcDistObject> cl(M("Tpetra"), "DistObject_double_int_long_long_Kokkos_Compat_KokkosDeviceWrapperNode_Kokkos_Serial_Kokkos_HostSpace_t", "");
		cl.def( pybind11::init<const class Teuchos::RCP<const class Tpetra::Map<int, long long> > &>(), pybind11::arg("map") );

		cl.def(pybind11::init<PyCallBack_Tpetra_DistObject_double_int_long_long_Kokkos_Compat_KokkosDeviceWrapperNode_Kokkos_Serial_Kokkos_HostSpace_t const &>());
		cl.def("assign", (class Tpetra::DistObject<double, int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & (Tpetra::DistObject<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::DistObject<double, int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &)) &Tpetra::DistObject<double, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::operator=, "C++: Tpetra::DistObject<double, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::operator=(const class Tpetra::DistObject<double, int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &) --> class Tpetra::DistObject<double, int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &", pybind11::return_value_policy::automatic, pybind11::arg(""));
		cl.def("doImport", [](Tpetra::DistObject<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> &o, const class Tpetra::SrcDistObject & a0, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & a1, const enum Tpetra::CombineMode & a2) -> void { return o.doImport(a0, a1, a2); }, "", pybind11::arg("source"), pybind11::arg("importer"), pybind11::arg("CM"));
		cl.def("doImport", (void (Tpetra::DistObject<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::SrcDistObject &, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool)) &Tpetra::DistObject<double, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::doImport, "C++: Tpetra::DistObject<double, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::doImport(const class Tpetra::SrcDistObject &, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool) --> void", pybind11::arg("source"), pybind11::arg("importer"), pybind11::arg("CM"), pybind11::arg("restrictedMode"));
		cl.def("doExport", [](Tpetra::DistObject<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> &o, const class Tpetra::SrcDistObject & a0, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & a1, const enum Tpetra::CombineMode & a2) -> void { return o.doExport(a0, a1, a2); }, "", pybind11::arg("source"), pybind11::arg("exporter"), pybind11::arg("CM"));
		cl.def("doExport", (void (Tpetra::DistObject<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::SrcDistObject &, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool)) &Tpetra::DistObject<double, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::doExport, "C++: Tpetra::DistObject<double, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::doExport(const class Tpetra::SrcDistObject &, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool) --> void", pybind11::arg("source"), pybind11::arg("exporter"), pybind11::arg("CM"), pybind11::arg("restrictedMode"));
		cl.def("doImport", [](Tpetra::DistObject<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> &o, const class Tpetra::SrcDistObject & a0, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & a1, const enum Tpetra::CombineMode & a2) -> void { return o.doImport(a0, a1, a2); }, "", pybind11::arg("source"), pybind11::arg("exporter"), pybind11::arg("CM"));
		cl.def("doImport", (void (Tpetra::DistObject<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::SrcDistObject &, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool)) &Tpetra::DistObject<double, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::doImport, "C++: Tpetra::DistObject<double, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::doImport(const class Tpetra::SrcDistObject &, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool) --> void", pybind11::arg("source"), pybind11::arg("exporter"), pybind11::arg("CM"), pybind11::arg("restrictedMode"));
		cl.def("doExport", [](Tpetra::DistObject<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> &o, const class Tpetra::SrcDistObject & a0, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & a1, const enum Tpetra::CombineMode & a2) -> void { return o.doExport(a0, a1, a2); }, "", pybind11::arg("source"), pybind11::arg("importer"), pybind11::arg("CM"));
		cl.def("doExport", (void (Tpetra::DistObject<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::SrcDistObject &, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool)) &Tpetra::DistObject<double, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::doExport, "C++: Tpetra::DistObject<double, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::doExport(const class Tpetra::SrcDistObject &, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool) --> void", pybind11::arg("source"), pybind11::arg("importer"), pybind11::arg("CM"), pybind11::arg("restrictedMode"));
		cl.def("beginImport", [](Tpetra::DistObject<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> &o, const class Tpetra::SrcDistObject & a0, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & a1, const enum Tpetra::CombineMode & a2) -> void { return o.beginImport(a0, a1, a2); }, "", pybind11::arg("source"), pybind11::arg("importer"), pybind11::arg("CM"));
		cl.def("beginImport", (void (Tpetra::DistObject<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::SrcDistObject &, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool)) &Tpetra::DistObject<double, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::beginImport, "C++: Tpetra::DistObject<double, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::beginImport(const class Tpetra::SrcDistObject &, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool) --> void", pybind11::arg("source"), pybind11::arg("importer"), pybind11::arg("CM"), pybind11::arg("restrictedMode"));
		cl.def("beginExport", [](Tpetra::DistObject<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> &o, const class Tpetra::SrcDistObject & a0, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & a1, const enum Tpetra::CombineMode & a2) -> void { return o.beginExport(a0, a1, a2); }, "", pybind11::arg("source"), pybind11::arg("exporter"), pybind11::arg("CM"));
		cl.def("beginExport", (void (Tpetra::DistObject<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::SrcDistObject &, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool)) &Tpetra::DistObject<double, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::beginExport, "C++: Tpetra::DistObject<double, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::beginExport(const class Tpetra::SrcDistObject &, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool) --> void", pybind11::arg("source"), pybind11::arg("exporter"), pybind11::arg("CM"), pybind11::arg("restrictedMode"));
		cl.def("beginImport", [](Tpetra::DistObject<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> &o, const class Tpetra::SrcDistObject & a0, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & a1, const enum Tpetra::CombineMode & a2) -> void { return o.beginImport(a0, a1, a2); }, "", pybind11::arg("source"), pybind11::arg("exporter"), pybind11::arg("CM"));
		cl.def("beginImport", (void (Tpetra::DistObject<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::SrcDistObject &, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool)) &Tpetra::DistObject<double, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::beginImport, "C++: Tpetra::DistObject<double, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::beginImport(const class Tpetra::SrcDistObject &, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool) --> void", pybind11::arg("source"), pybind11::arg("exporter"), pybind11::arg("CM"), pybind11::arg("restrictedMode"));
		cl.def("beginExport", [](Tpetra::DistObject<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> &o, const class Tpetra::SrcDistObject & a0, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & a1, const enum Tpetra::CombineMode & a2) -> void { return o.beginExport(a0, a1, a2); }, "", pybind11::arg("source"), pybind11::arg("importer"), pybind11::arg("CM"));
		cl.def("beginExport", (void (Tpetra::DistObject<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::SrcDistObject &, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool)) &Tpetra::DistObject<double, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::beginExport, "C++: Tpetra::DistObject<double, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::beginExport(const class Tpetra::SrcDistObject &, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool) --> void", pybind11::arg("source"), pybind11::arg("importer"), pybind11::arg("CM"), pybind11::arg("restrictedMode"));
		cl.def("endImport", [](Tpetra::DistObject<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> &o, const class Tpetra::SrcDistObject & a0, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & a1, const enum Tpetra::CombineMode & a2) -> void { return o.endImport(a0, a1, a2); }, "", pybind11::arg("source"), pybind11::arg("importer"), pybind11::arg("CM"));
		cl.def("endImport", (void (Tpetra::DistObject<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::SrcDistObject &, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool)) &Tpetra::DistObject<double, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::endImport, "C++: Tpetra::DistObject<double, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::endImport(const class Tpetra::SrcDistObject &, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool) --> void", pybind11::arg("source"), pybind11::arg("importer"), pybind11::arg("CM"), pybind11::arg("restrictedMode"));
		cl.def("endExport", [](Tpetra::DistObject<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> &o, const class Tpetra::SrcDistObject & a0, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & a1, const enum Tpetra::CombineMode & a2) -> void { return o.endExport(a0, a1, a2); }, "", pybind11::arg("source"), pybind11::arg("exporter"), pybind11::arg("CM"));
		cl.def("endExport", (void (Tpetra::DistObject<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::SrcDistObject &, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool)) &Tpetra::DistObject<double, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::endExport, "C++: Tpetra::DistObject<double, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::endExport(const class Tpetra::SrcDistObject &, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool) --> void", pybind11::arg("source"), pybind11::arg("exporter"), pybind11::arg("CM"), pybind11::arg("restrictedMode"));
		cl.def("endImport", [](Tpetra::DistObject<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> &o, const class Tpetra::SrcDistObject & a0, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & a1, const enum Tpetra::CombineMode & a2) -> void { return o.endImport(a0, a1, a2); }, "", pybind11::arg("source"), pybind11::arg("exporter"), pybind11::arg("CM"));
		cl.def("endImport", (void (Tpetra::DistObject<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::SrcDistObject &, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool)) &Tpetra::DistObject<double, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::endImport, "C++: Tpetra::DistObject<double, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::endImport(const class Tpetra::SrcDistObject &, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool) --> void", pybind11::arg("source"), pybind11::arg("exporter"), pybind11::arg("CM"), pybind11::arg("restrictedMode"));
		cl.def("endExport", [](Tpetra::DistObject<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> &o, const class Tpetra::SrcDistObject & a0, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & a1, const enum Tpetra::CombineMode & a2) -> void { return o.endExport(a0, a1, a2); }, "", pybind11::arg("source"), pybind11::arg("importer"), pybind11::arg("CM"));
		cl.def("endExport", (void (Tpetra::DistObject<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::SrcDistObject &, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool)) &Tpetra::DistObject<double, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::endExport, "C++: Tpetra::DistObject<double, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::endExport(const class Tpetra::SrcDistObject &, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool) --> void", pybind11::arg("source"), pybind11::arg("importer"), pybind11::arg("CM"), pybind11::arg("restrictedMode"));
		cl.def("transferArrived", (bool (Tpetra::DistObject<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)() const) &Tpetra::DistObject<double, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::transferArrived, "C++: Tpetra::DistObject<double, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::transferArrived() const --> bool");
		cl.def("isDistributed", (bool (Tpetra::DistObject<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)() const) &Tpetra::DistObject<double, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::isDistributed, "C++: Tpetra::DistObject<double, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::isDistributed() const --> bool");
		cl.def("getMap", (class Teuchos::RCP<const class Tpetra::Map<int, long long> > (Tpetra::DistObject<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)() const) &Tpetra::DistObject<double, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::getMap, "C++: Tpetra::DistObject<double, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::getMap() const --> class Teuchos::RCP<const class Tpetra::Map<int, long long> >");
		cl.def("description", (std::string (Tpetra::DistObject<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)() const) &Tpetra::DistObject<double, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::description, "C++: Tpetra::DistObject<double, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::description() const --> std::string");
		cl.def("describe", [](Tpetra::DistObject<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> const &o, class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > & a0) -> void { return o.describe(a0); }, "", pybind11::arg("out"));
		cl.def("describe", (void (Tpetra::DistObject<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > &, const enum Teuchos::EVerbosityLevel) const) &Tpetra::DistObject<double, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::describe, "C++: Tpetra::DistObject<double, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::describe(class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > &, const enum Teuchos::EVerbosityLevel) const --> void", pybind11::arg("out"), pybind11::arg("verbLevel"));
		cl.def("removeEmptyProcessesInPlace", (void (Tpetra::DistObject<double,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Teuchos::RCP<const class Tpetra::Map<int, long long> > &)) &Tpetra::DistObject<double, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::removeEmptyProcessesInPlace, "C++: Tpetra::DistObject<double, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::removeEmptyProcessesInPlace(const class Teuchos::RCP<const class Tpetra::Map<int, long long> > &) --> void", pybind11::arg("newMap"));
	}
	{ // Tpetra::DistObject file:Tpetra_DistObject_decl.hpp line:321
		pybind11::class_<Tpetra::DistObject<long long,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>, Teuchos::RCP<Tpetra::DistObject<long long,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>>, PyCallBack_Tpetra_DistObject_long_long_int_long_long_Kokkos_Compat_KokkosDeviceWrapperNode_Kokkos_Serial_Kokkos_HostSpace_t, Tpetra::SrcDistObject> cl(M("Tpetra"), "DistObject_long_long_int_long_long_Kokkos_Compat_KokkosDeviceWrapperNode_Kokkos_Serial_Kokkos_HostSpace_t", "");
		cl.def( pybind11::init<const class Teuchos::RCP<const class Tpetra::Map<int, long long> > &>(), pybind11::arg("map") );

		cl.def(pybind11::init<PyCallBack_Tpetra_DistObject_long_long_int_long_long_Kokkos_Compat_KokkosDeviceWrapperNode_Kokkos_Serial_Kokkos_HostSpace_t const &>());
		cl.def("assign", (class Tpetra::DistObject<long long, int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & (Tpetra::DistObject<long long,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::DistObject<long long, int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &)) &Tpetra::DistObject<long long, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::operator=, "C++: Tpetra::DistObject<long long, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::operator=(const class Tpetra::DistObject<long long, int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &) --> class Tpetra::DistObject<long long, int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &", pybind11::return_value_policy::automatic, pybind11::arg(""));
		cl.def("doImport", [](Tpetra::DistObject<long long,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> &o, const class Tpetra::SrcDistObject & a0, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & a1, const enum Tpetra::CombineMode & a2) -> void { return o.doImport(a0, a1, a2); }, "", pybind11::arg("source"), pybind11::arg("importer"), pybind11::arg("CM"));
		cl.def("doImport", (void (Tpetra::DistObject<long long,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::SrcDistObject &, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool)) &Tpetra::DistObject<long long, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::doImport, "C++: Tpetra::DistObject<long long, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::doImport(const class Tpetra::SrcDistObject &, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool) --> void", pybind11::arg("source"), pybind11::arg("importer"), pybind11::arg("CM"), pybind11::arg("restrictedMode"));
		cl.def("doExport", [](Tpetra::DistObject<long long,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> &o, const class Tpetra::SrcDistObject & a0, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & a1, const enum Tpetra::CombineMode & a2) -> void { return o.doExport(a0, a1, a2); }, "", pybind11::arg("source"), pybind11::arg("exporter"), pybind11::arg("CM"));
		cl.def("doExport", (void (Tpetra::DistObject<long long,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::SrcDistObject &, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool)) &Tpetra::DistObject<long long, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::doExport, "C++: Tpetra::DistObject<long long, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::doExport(const class Tpetra::SrcDistObject &, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool) --> void", pybind11::arg("source"), pybind11::arg("exporter"), pybind11::arg("CM"), pybind11::arg("restrictedMode"));
		cl.def("doImport", [](Tpetra::DistObject<long long,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> &o, const class Tpetra::SrcDistObject & a0, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & a1, const enum Tpetra::CombineMode & a2) -> void { return o.doImport(a0, a1, a2); }, "", pybind11::arg("source"), pybind11::arg("exporter"), pybind11::arg("CM"));
		cl.def("doImport", (void (Tpetra::DistObject<long long,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::SrcDistObject &, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool)) &Tpetra::DistObject<long long, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::doImport, "C++: Tpetra::DistObject<long long, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::doImport(const class Tpetra::SrcDistObject &, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool) --> void", pybind11::arg("source"), pybind11::arg("exporter"), pybind11::arg("CM"), pybind11::arg("restrictedMode"));
		cl.def("doExport", [](Tpetra::DistObject<long long,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> &o, const class Tpetra::SrcDistObject & a0, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & a1, const enum Tpetra::CombineMode & a2) -> void { return o.doExport(a0, a1, a2); }, "", pybind11::arg("source"), pybind11::arg("importer"), pybind11::arg("CM"));
		cl.def("doExport", (void (Tpetra::DistObject<long long,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::SrcDistObject &, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool)) &Tpetra::DistObject<long long, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::doExport, "C++: Tpetra::DistObject<long long, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::doExport(const class Tpetra::SrcDistObject &, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool) --> void", pybind11::arg("source"), pybind11::arg("importer"), pybind11::arg("CM"), pybind11::arg("restrictedMode"));
		cl.def("beginImport", [](Tpetra::DistObject<long long,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> &o, const class Tpetra::SrcDistObject & a0, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & a1, const enum Tpetra::CombineMode & a2) -> void { return o.beginImport(a0, a1, a2); }, "", pybind11::arg("source"), pybind11::arg("importer"), pybind11::arg("CM"));
		cl.def("beginImport", (void (Tpetra::DistObject<long long,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::SrcDistObject &, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool)) &Tpetra::DistObject<long long, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::beginImport, "C++: Tpetra::DistObject<long long, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::beginImport(const class Tpetra::SrcDistObject &, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool) --> void", pybind11::arg("source"), pybind11::arg("importer"), pybind11::arg("CM"), pybind11::arg("restrictedMode"));
		cl.def("beginExport", [](Tpetra::DistObject<long long,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> &o, const class Tpetra::SrcDistObject & a0, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & a1, const enum Tpetra::CombineMode & a2) -> void { return o.beginExport(a0, a1, a2); }, "", pybind11::arg("source"), pybind11::arg("exporter"), pybind11::arg("CM"));
		cl.def("beginExport", (void (Tpetra::DistObject<long long,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::SrcDistObject &, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool)) &Tpetra::DistObject<long long, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::beginExport, "C++: Tpetra::DistObject<long long, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::beginExport(const class Tpetra::SrcDistObject &, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool) --> void", pybind11::arg("source"), pybind11::arg("exporter"), pybind11::arg("CM"), pybind11::arg("restrictedMode"));
		cl.def("beginImport", [](Tpetra::DistObject<long long,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> &o, const class Tpetra::SrcDistObject & a0, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & a1, const enum Tpetra::CombineMode & a2) -> void { return o.beginImport(a0, a1, a2); }, "", pybind11::arg("source"), pybind11::arg("exporter"), pybind11::arg("CM"));
		cl.def("beginImport", (void (Tpetra::DistObject<long long,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::SrcDistObject &, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool)) &Tpetra::DistObject<long long, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::beginImport, "C++: Tpetra::DistObject<long long, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::beginImport(const class Tpetra::SrcDistObject &, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool) --> void", pybind11::arg("source"), pybind11::arg("exporter"), pybind11::arg("CM"), pybind11::arg("restrictedMode"));
		cl.def("beginExport", [](Tpetra::DistObject<long long,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> &o, const class Tpetra::SrcDistObject & a0, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & a1, const enum Tpetra::CombineMode & a2) -> void { return o.beginExport(a0, a1, a2); }, "", pybind11::arg("source"), pybind11::arg("importer"), pybind11::arg("CM"));
		cl.def("beginExport", (void (Tpetra::DistObject<long long,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::SrcDistObject &, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool)) &Tpetra::DistObject<long long, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::beginExport, "C++: Tpetra::DistObject<long long, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::beginExport(const class Tpetra::SrcDistObject &, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool) --> void", pybind11::arg("source"), pybind11::arg("importer"), pybind11::arg("CM"), pybind11::arg("restrictedMode"));
		cl.def("endImport", [](Tpetra::DistObject<long long,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> &o, const class Tpetra::SrcDistObject & a0, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & a1, const enum Tpetra::CombineMode & a2) -> void { return o.endImport(a0, a1, a2); }, "", pybind11::arg("source"), pybind11::arg("importer"), pybind11::arg("CM"));
		cl.def("endImport", (void (Tpetra::DistObject<long long,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::SrcDistObject &, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool)) &Tpetra::DistObject<long long, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::endImport, "C++: Tpetra::DistObject<long long, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::endImport(const class Tpetra::SrcDistObject &, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool) --> void", pybind11::arg("source"), pybind11::arg("importer"), pybind11::arg("CM"), pybind11::arg("restrictedMode"));
		cl.def("endExport", [](Tpetra::DistObject<long long,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> &o, const class Tpetra::SrcDistObject & a0, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & a1, const enum Tpetra::CombineMode & a2) -> void { return o.endExport(a0, a1, a2); }, "", pybind11::arg("source"), pybind11::arg("exporter"), pybind11::arg("CM"));
		cl.def("endExport", (void (Tpetra::DistObject<long long,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::SrcDistObject &, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool)) &Tpetra::DistObject<long long, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::endExport, "C++: Tpetra::DistObject<long long, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::endExport(const class Tpetra::SrcDistObject &, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool) --> void", pybind11::arg("source"), pybind11::arg("exporter"), pybind11::arg("CM"), pybind11::arg("restrictedMode"));
		cl.def("endImport", [](Tpetra::DistObject<long long,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> &o, const class Tpetra::SrcDistObject & a0, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & a1, const enum Tpetra::CombineMode & a2) -> void { return o.endImport(a0, a1, a2); }, "", pybind11::arg("source"), pybind11::arg("exporter"), pybind11::arg("CM"));
		cl.def("endImport", (void (Tpetra::DistObject<long long,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::SrcDistObject &, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool)) &Tpetra::DistObject<long long, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::endImport, "C++: Tpetra::DistObject<long long, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::endImport(const class Tpetra::SrcDistObject &, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool) --> void", pybind11::arg("source"), pybind11::arg("exporter"), pybind11::arg("CM"), pybind11::arg("restrictedMode"));
		cl.def("endExport", [](Tpetra::DistObject<long long,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> &o, const class Tpetra::SrcDistObject & a0, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & a1, const enum Tpetra::CombineMode & a2) -> void { return o.endExport(a0, a1, a2); }, "", pybind11::arg("source"), pybind11::arg("importer"), pybind11::arg("CM"));
		cl.def("endExport", (void (Tpetra::DistObject<long long,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::SrcDistObject &, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool)) &Tpetra::DistObject<long long, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::endExport, "C++: Tpetra::DistObject<long long, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::endExport(const class Tpetra::SrcDistObject &, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool) --> void", pybind11::arg("source"), pybind11::arg("importer"), pybind11::arg("CM"), pybind11::arg("restrictedMode"));
		cl.def("transferArrived", (bool (Tpetra::DistObject<long long,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)() const) &Tpetra::DistObject<long long, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::transferArrived, "C++: Tpetra::DistObject<long long, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::transferArrived() const --> bool");
		cl.def("isDistributed", (bool (Tpetra::DistObject<long long,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)() const) &Tpetra::DistObject<long long, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::isDistributed, "C++: Tpetra::DistObject<long long, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::isDistributed() const --> bool");
		cl.def("getMap", (class Teuchos::RCP<const class Tpetra::Map<int, long long> > (Tpetra::DistObject<long long,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)() const) &Tpetra::DistObject<long long, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::getMap, "C++: Tpetra::DistObject<long long, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::getMap() const --> class Teuchos::RCP<const class Tpetra::Map<int, long long> >");
		cl.def("description", (std::string (Tpetra::DistObject<long long,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)() const) &Tpetra::DistObject<long long, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::description, "C++: Tpetra::DistObject<long long, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::description() const --> std::string");
		cl.def("describe", [](Tpetra::DistObject<long long,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> const &o, class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > & a0) -> void { return o.describe(a0); }, "", pybind11::arg("out"));
		cl.def("describe", (void (Tpetra::DistObject<long long,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > &, const enum Teuchos::EVerbosityLevel) const) &Tpetra::DistObject<long long, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::describe, "C++: Tpetra::DistObject<long long, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::describe(class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > &, const enum Teuchos::EVerbosityLevel) const --> void", pybind11::arg("out"), pybind11::arg("verbLevel"));
		cl.def("removeEmptyProcessesInPlace", (void (Tpetra::DistObject<long long,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Teuchos::RCP<const class Tpetra::Map<int, long long> > &)) &Tpetra::DistObject<long long, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::removeEmptyProcessesInPlace, "C++: Tpetra::DistObject<long long, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::removeEmptyProcessesInPlace(const class Teuchos::RCP<const class Tpetra::Map<int, long long> > &) --> void", pybind11::arg("newMap"));
	}
	{ // Tpetra::DistObject file:Tpetra_DistObject_decl.hpp line:321
		pybind11::class_<Tpetra::DistObject<int,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>, Teuchos::RCP<Tpetra::DistObject<int,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>>, PyCallBack_Tpetra_DistObject_int_int_long_long_Kokkos_Compat_KokkosDeviceWrapperNode_Kokkos_Serial_Kokkos_HostSpace_t, Tpetra::SrcDistObject> cl(M("Tpetra"), "DistObject_int_int_long_long_Kokkos_Compat_KokkosDeviceWrapperNode_Kokkos_Serial_Kokkos_HostSpace_t", "");
		cl.def( pybind11::init<const class Teuchos::RCP<const class Tpetra::Map<int, long long> > &>(), pybind11::arg("map") );

		cl.def(pybind11::init<PyCallBack_Tpetra_DistObject_int_int_long_long_Kokkos_Compat_KokkosDeviceWrapperNode_Kokkos_Serial_Kokkos_HostSpace_t const &>());
		cl.def("assign", (class Tpetra::DistObject<int, int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & (Tpetra::DistObject<int,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::DistObject<int, int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &)) &Tpetra::DistObject<int, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::operator=, "C++: Tpetra::DistObject<int, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::operator=(const class Tpetra::DistObject<int, int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &) --> class Tpetra::DistObject<int, int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &", pybind11::return_value_policy::automatic, pybind11::arg(""));
		cl.def("doImport", [](Tpetra::DistObject<int,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> &o, const class Tpetra::SrcDistObject & a0, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & a1, const enum Tpetra::CombineMode & a2) -> void { return o.doImport(a0, a1, a2); }, "", pybind11::arg("source"), pybind11::arg("importer"), pybind11::arg("CM"));
		cl.def("doImport", (void (Tpetra::DistObject<int,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::SrcDistObject &, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool)) &Tpetra::DistObject<int, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::doImport, "C++: Tpetra::DistObject<int, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::doImport(const class Tpetra::SrcDistObject &, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool) --> void", pybind11::arg("source"), pybind11::arg("importer"), pybind11::arg("CM"), pybind11::arg("restrictedMode"));
		cl.def("doExport", [](Tpetra::DistObject<int,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> &o, const class Tpetra::SrcDistObject & a0, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & a1, const enum Tpetra::CombineMode & a2) -> void { return o.doExport(a0, a1, a2); }, "", pybind11::arg("source"), pybind11::arg("exporter"), pybind11::arg("CM"));
		cl.def("doExport", (void (Tpetra::DistObject<int,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::SrcDistObject &, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool)) &Tpetra::DistObject<int, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::doExport, "C++: Tpetra::DistObject<int, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::doExport(const class Tpetra::SrcDistObject &, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool) --> void", pybind11::arg("source"), pybind11::arg("exporter"), pybind11::arg("CM"), pybind11::arg("restrictedMode"));
		cl.def("doImport", [](Tpetra::DistObject<int,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> &o, const class Tpetra::SrcDistObject & a0, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & a1, const enum Tpetra::CombineMode & a2) -> void { return o.doImport(a0, a1, a2); }, "", pybind11::arg("source"), pybind11::arg("exporter"), pybind11::arg("CM"));
		cl.def("doImport", (void (Tpetra::DistObject<int,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::SrcDistObject &, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool)) &Tpetra::DistObject<int, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::doImport, "C++: Tpetra::DistObject<int, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::doImport(const class Tpetra::SrcDistObject &, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool) --> void", pybind11::arg("source"), pybind11::arg("exporter"), pybind11::arg("CM"), pybind11::arg("restrictedMode"));
		cl.def("doExport", [](Tpetra::DistObject<int,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> &o, const class Tpetra::SrcDistObject & a0, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & a1, const enum Tpetra::CombineMode & a2) -> void { return o.doExport(a0, a1, a2); }, "", pybind11::arg("source"), pybind11::arg("importer"), pybind11::arg("CM"));
		cl.def("doExport", (void (Tpetra::DistObject<int,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::SrcDistObject &, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool)) &Tpetra::DistObject<int, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::doExport, "C++: Tpetra::DistObject<int, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::doExport(const class Tpetra::SrcDistObject &, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool) --> void", pybind11::arg("source"), pybind11::arg("importer"), pybind11::arg("CM"), pybind11::arg("restrictedMode"));
		cl.def("beginImport", [](Tpetra::DistObject<int,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> &o, const class Tpetra::SrcDistObject & a0, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & a1, const enum Tpetra::CombineMode & a2) -> void { return o.beginImport(a0, a1, a2); }, "", pybind11::arg("source"), pybind11::arg("importer"), pybind11::arg("CM"));
		cl.def("beginImport", (void (Tpetra::DistObject<int,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::SrcDistObject &, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool)) &Tpetra::DistObject<int, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::beginImport, "C++: Tpetra::DistObject<int, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::beginImport(const class Tpetra::SrcDistObject &, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool) --> void", pybind11::arg("source"), pybind11::arg("importer"), pybind11::arg("CM"), pybind11::arg("restrictedMode"));
		cl.def("beginExport", [](Tpetra::DistObject<int,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> &o, const class Tpetra::SrcDistObject & a0, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & a1, const enum Tpetra::CombineMode & a2) -> void { return o.beginExport(a0, a1, a2); }, "", pybind11::arg("source"), pybind11::arg("exporter"), pybind11::arg("CM"));
		cl.def("beginExport", (void (Tpetra::DistObject<int,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::SrcDistObject &, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool)) &Tpetra::DistObject<int, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::beginExport, "C++: Tpetra::DistObject<int, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::beginExport(const class Tpetra::SrcDistObject &, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool) --> void", pybind11::arg("source"), pybind11::arg("exporter"), pybind11::arg("CM"), pybind11::arg("restrictedMode"));
		cl.def("beginImport", [](Tpetra::DistObject<int,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> &o, const class Tpetra::SrcDistObject & a0, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & a1, const enum Tpetra::CombineMode & a2) -> void { return o.beginImport(a0, a1, a2); }, "", pybind11::arg("source"), pybind11::arg("exporter"), pybind11::arg("CM"));
		cl.def("beginImport", (void (Tpetra::DistObject<int,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::SrcDistObject &, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool)) &Tpetra::DistObject<int, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::beginImport, "C++: Tpetra::DistObject<int, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::beginImport(const class Tpetra::SrcDistObject &, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool) --> void", pybind11::arg("source"), pybind11::arg("exporter"), pybind11::arg("CM"), pybind11::arg("restrictedMode"));
		cl.def("beginExport", [](Tpetra::DistObject<int,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> &o, const class Tpetra::SrcDistObject & a0, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & a1, const enum Tpetra::CombineMode & a2) -> void { return o.beginExport(a0, a1, a2); }, "", pybind11::arg("source"), pybind11::arg("importer"), pybind11::arg("CM"));
		cl.def("beginExport", (void (Tpetra::DistObject<int,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::SrcDistObject &, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool)) &Tpetra::DistObject<int, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::beginExport, "C++: Tpetra::DistObject<int, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::beginExport(const class Tpetra::SrcDistObject &, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool) --> void", pybind11::arg("source"), pybind11::arg("importer"), pybind11::arg("CM"), pybind11::arg("restrictedMode"));
		cl.def("endImport", [](Tpetra::DistObject<int,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> &o, const class Tpetra::SrcDistObject & a0, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & a1, const enum Tpetra::CombineMode & a2) -> void { return o.endImport(a0, a1, a2); }, "", pybind11::arg("source"), pybind11::arg("importer"), pybind11::arg("CM"));
		cl.def("endImport", (void (Tpetra::DistObject<int,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::SrcDistObject &, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool)) &Tpetra::DistObject<int, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::endImport, "C++: Tpetra::DistObject<int, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::endImport(const class Tpetra::SrcDistObject &, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool) --> void", pybind11::arg("source"), pybind11::arg("importer"), pybind11::arg("CM"), pybind11::arg("restrictedMode"));
		cl.def("endExport", [](Tpetra::DistObject<int,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> &o, const class Tpetra::SrcDistObject & a0, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & a1, const enum Tpetra::CombineMode & a2) -> void { return o.endExport(a0, a1, a2); }, "", pybind11::arg("source"), pybind11::arg("exporter"), pybind11::arg("CM"));
		cl.def("endExport", (void (Tpetra::DistObject<int,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::SrcDistObject &, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool)) &Tpetra::DistObject<int, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::endExport, "C++: Tpetra::DistObject<int, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::endExport(const class Tpetra::SrcDistObject &, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool) --> void", pybind11::arg("source"), pybind11::arg("exporter"), pybind11::arg("CM"), pybind11::arg("restrictedMode"));
		cl.def("endImport", [](Tpetra::DistObject<int,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> &o, const class Tpetra::SrcDistObject & a0, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & a1, const enum Tpetra::CombineMode & a2) -> void { return o.endImport(a0, a1, a2); }, "", pybind11::arg("source"), pybind11::arg("exporter"), pybind11::arg("CM"));
		cl.def("endImport", (void (Tpetra::DistObject<int,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::SrcDistObject &, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool)) &Tpetra::DistObject<int, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::endImport, "C++: Tpetra::DistObject<int, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::endImport(const class Tpetra::SrcDistObject &, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool) --> void", pybind11::arg("source"), pybind11::arg("exporter"), pybind11::arg("CM"), pybind11::arg("restrictedMode"));
		cl.def("endExport", [](Tpetra::DistObject<int,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> &o, const class Tpetra::SrcDistObject & a0, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & a1, const enum Tpetra::CombineMode & a2) -> void { return o.endExport(a0, a1, a2); }, "", pybind11::arg("source"), pybind11::arg("importer"), pybind11::arg("CM"));
		cl.def("endExport", (void (Tpetra::DistObject<int,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::SrcDistObject &, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool)) &Tpetra::DistObject<int, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::endExport, "C++: Tpetra::DistObject<int, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::endExport(const class Tpetra::SrcDistObject &, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool) --> void", pybind11::arg("source"), pybind11::arg("importer"), pybind11::arg("CM"), pybind11::arg("restrictedMode"));
		cl.def("transferArrived", (bool (Tpetra::DistObject<int,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)() const) &Tpetra::DistObject<int, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::transferArrived, "C++: Tpetra::DistObject<int, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::transferArrived() const --> bool");
		cl.def("isDistributed", (bool (Tpetra::DistObject<int,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)() const) &Tpetra::DistObject<int, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::isDistributed, "C++: Tpetra::DistObject<int, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::isDistributed() const --> bool");
		cl.def("getMap", (class Teuchos::RCP<const class Tpetra::Map<int, long long> > (Tpetra::DistObject<int,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)() const) &Tpetra::DistObject<int, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::getMap, "C++: Tpetra::DistObject<int, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::getMap() const --> class Teuchos::RCP<const class Tpetra::Map<int, long long> >");
		cl.def("description", (std::string (Tpetra::DistObject<int,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)() const) &Tpetra::DistObject<int, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::description, "C++: Tpetra::DistObject<int, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::description() const --> std::string");
		cl.def("describe", [](Tpetra::DistObject<int,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> const &o, class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > & a0) -> void { return o.describe(a0); }, "", pybind11::arg("out"));
		cl.def("describe", (void (Tpetra::DistObject<int,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > &, const enum Teuchos::EVerbosityLevel) const) &Tpetra::DistObject<int, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::describe, "C++: Tpetra::DistObject<int, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::describe(class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > &, const enum Teuchos::EVerbosityLevel) const --> void", pybind11::arg("out"), pybind11::arg("verbLevel"));
		cl.def("removeEmptyProcessesInPlace", (void (Tpetra::DistObject<int,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Teuchos::RCP<const class Tpetra::Map<int, long long> > &)) &Tpetra::DistObject<int, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::removeEmptyProcessesInPlace, "C++: Tpetra::DistObject<int, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::removeEmptyProcessesInPlace(const class Teuchos::RCP<const class Tpetra::Map<int, long long> > &) --> void", pybind11::arg("newMap"));
	}
	{ // Tpetra::DistObject file:Tpetra_DistObject_decl.hpp line:321
		pybind11::class_<Tpetra::DistObject<char,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>, Teuchos::RCP<Tpetra::DistObject<char,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>>, PyCallBack_Tpetra_DistObject_char_int_long_long_Kokkos_Compat_KokkosDeviceWrapperNode_Kokkos_Serial_Kokkos_HostSpace_t, Tpetra::SrcDistObject> cl(M("Tpetra"), "DistObject_char_int_long_long_Kokkos_Compat_KokkosDeviceWrapperNode_Kokkos_Serial_Kokkos_HostSpace_t", "");
		cl.def( pybind11::init<const class Teuchos::RCP<const class Tpetra::Map<int, long long> > &>(), pybind11::arg("map") );

		cl.def(pybind11::init<PyCallBack_Tpetra_DistObject_char_int_long_long_Kokkos_Compat_KokkosDeviceWrapperNode_Kokkos_Serial_Kokkos_HostSpace_t const &>());
		cl.def("assign", (class Tpetra::DistObject<char, int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & (Tpetra::DistObject<char,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::DistObject<char, int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &)) &Tpetra::DistObject<char, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::operator=, "C++: Tpetra::DistObject<char, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::operator=(const class Tpetra::DistObject<char, int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &) --> class Tpetra::DistObject<char, int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &", pybind11::return_value_policy::automatic, pybind11::arg(""));
		cl.def("doImport", [](Tpetra::DistObject<char,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> &o, const class Tpetra::SrcDistObject & a0, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & a1, const enum Tpetra::CombineMode & a2) -> void { return o.doImport(a0, a1, a2); }, "", pybind11::arg("source"), pybind11::arg("importer"), pybind11::arg("CM"));
		cl.def("doImport", (void (Tpetra::DistObject<char,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::SrcDistObject &, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool)) &Tpetra::DistObject<char, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::doImport, "C++: Tpetra::DistObject<char, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::doImport(const class Tpetra::SrcDistObject &, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool) --> void", pybind11::arg("source"), pybind11::arg("importer"), pybind11::arg("CM"), pybind11::arg("restrictedMode"));
		cl.def("doExport", [](Tpetra::DistObject<char,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> &o, const class Tpetra::SrcDistObject & a0, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & a1, const enum Tpetra::CombineMode & a2) -> void { return o.doExport(a0, a1, a2); }, "", pybind11::arg("source"), pybind11::arg("exporter"), pybind11::arg("CM"));
		cl.def("doExport", (void (Tpetra::DistObject<char,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::SrcDistObject &, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool)) &Tpetra::DistObject<char, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::doExport, "C++: Tpetra::DistObject<char, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::doExport(const class Tpetra::SrcDistObject &, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool) --> void", pybind11::arg("source"), pybind11::arg("exporter"), pybind11::arg("CM"), pybind11::arg("restrictedMode"));
		cl.def("doImport", [](Tpetra::DistObject<char,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> &o, const class Tpetra::SrcDistObject & a0, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & a1, const enum Tpetra::CombineMode & a2) -> void { return o.doImport(a0, a1, a2); }, "", pybind11::arg("source"), pybind11::arg("exporter"), pybind11::arg("CM"));
		cl.def("doImport", (void (Tpetra::DistObject<char,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::SrcDistObject &, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool)) &Tpetra::DistObject<char, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::doImport, "C++: Tpetra::DistObject<char, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::doImport(const class Tpetra::SrcDistObject &, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool) --> void", pybind11::arg("source"), pybind11::arg("exporter"), pybind11::arg("CM"), pybind11::arg("restrictedMode"));
		cl.def("doExport", [](Tpetra::DistObject<char,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> &o, const class Tpetra::SrcDistObject & a0, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & a1, const enum Tpetra::CombineMode & a2) -> void { return o.doExport(a0, a1, a2); }, "", pybind11::arg("source"), pybind11::arg("importer"), pybind11::arg("CM"));
		cl.def("doExport", (void (Tpetra::DistObject<char,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::SrcDistObject &, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool)) &Tpetra::DistObject<char, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::doExport, "C++: Tpetra::DistObject<char, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::doExport(const class Tpetra::SrcDistObject &, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool) --> void", pybind11::arg("source"), pybind11::arg("importer"), pybind11::arg("CM"), pybind11::arg("restrictedMode"));
		cl.def("beginImport", [](Tpetra::DistObject<char,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> &o, const class Tpetra::SrcDistObject & a0, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & a1, const enum Tpetra::CombineMode & a2) -> void { return o.beginImport(a0, a1, a2); }, "", pybind11::arg("source"), pybind11::arg("importer"), pybind11::arg("CM"));
		cl.def("beginImport", (void (Tpetra::DistObject<char,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::SrcDistObject &, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool)) &Tpetra::DistObject<char, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::beginImport, "C++: Tpetra::DistObject<char, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::beginImport(const class Tpetra::SrcDistObject &, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool) --> void", pybind11::arg("source"), pybind11::arg("importer"), pybind11::arg("CM"), pybind11::arg("restrictedMode"));
		cl.def("beginExport", [](Tpetra::DistObject<char,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> &o, const class Tpetra::SrcDistObject & a0, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & a1, const enum Tpetra::CombineMode & a2) -> void { return o.beginExport(a0, a1, a2); }, "", pybind11::arg("source"), pybind11::arg("exporter"), pybind11::arg("CM"));
		cl.def("beginExport", (void (Tpetra::DistObject<char,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::SrcDistObject &, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool)) &Tpetra::DistObject<char, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::beginExport, "C++: Tpetra::DistObject<char, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::beginExport(const class Tpetra::SrcDistObject &, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool) --> void", pybind11::arg("source"), pybind11::arg("exporter"), pybind11::arg("CM"), pybind11::arg("restrictedMode"));
		cl.def("beginImport", [](Tpetra::DistObject<char,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> &o, const class Tpetra::SrcDistObject & a0, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & a1, const enum Tpetra::CombineMode & a2) -> void { return o.beginImport(a0, a1, a2); }, "", pybind11::arg("source"), pybind11::arg("exporter"), pybind11::arg("CM"));
		cl.def("beginImport", (void (Tpetra::DistObject<char,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::SrcDistObject &, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool)) &Tpetra::DistObject<char, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::beginImport, "C++: Tpetra::DistObject<char, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::beginImport(const class Tpetra::SrcDistObject &, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool) --> void", pybind11::arg("source"), pybind11::arg("exporter"), pybind11::arg("CM"), pybind11::arg("restrictedMode"));
		cl.def("beginExport", [](Tpetra::DistObject<char,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> &o, const class Tpetra::SrcDistObject & a0, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & a1, const enum Tpetra::CombineMode & a2) -> void { return o.beginExport(a0, a1, a2); }, "", pybind11::arg("source"), pybind11::arg("importer"), pybind11::arg("CM"));
		cl.def("beginExport", (void (Tpetra::DistObject<char,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::SrcDistObject &, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool)) &Tpetra::DistObject<char, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::beginExport, "C++: Tpetra::DistObject<char, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::beginExport(const class Tpetra::SrcDistObject &, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool) --> void", pybind11::arg("source"), pybind11::arg("importer"), pybind11::arg("CM"), pybind11::arg("restrictedMode"));
		cl.def("endImport", [](Tpetra::DistObject<char,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> &o, const class Tpetra::SrcDistObject & a0, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & a1, const enum Tpetra::CombineMode & a2) -> void { return o.endImport(a0, a1, a2); }, "", pybind11::arg("source"), pybind11::arg("importer"), pybind11::arg("CM"));
		cl.def("endImport", (void (Tpetra::DistObject<char,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::SrcDistObject &, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool)) &Tpetra::DistObject<char, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::endImport, "C++: Tpetra::DistObject<char, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::endImport(const class Tpetra::SrcDistObject &, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool) --> void", pybind11::arg("source"), pybind11::arg("importer"), pybind11::arg("CM"), pybind11::arg("restrictedMode"));
		cl.def("endExport", [](Tpetra::DistObject<char,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> &o, const class Tpetra::SrcDistObject & a0, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & a1, const enum Tpetra::CombineMode & a2) -> void { return o.endExport(a0, a1, a2); }, "", pybind11::arg("source"), pybind11::arg("exporter"), pybind11::arg("CM"));
		cl.def("endExport", (void (Tpetra::DistObject<char,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::SrcDistObject &, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool)) &Tpetra::DistObject<char, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::endExport, "C++: Tpetra::DistObject<char, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::endExport(const class Tpetra::SrcDistObject &, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool) --> void", pybind11::arg("source"), pybind11::arg("exporter"), pybind11::arg("CM"), pybind11::arg("restrictedMode"));
		cl.def("endImport", [](Tpetra::DistObject<char,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> &o, const class Tpetra::SrcDistObject & a0, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & a1, const enum Tpetra::CombineMode & a2) -> void { return o.endImport(a0, a1, a2); }, "", pybind11::arg("source"), pybind11::arg("exporter"), pybind11::arg("CM"));
		cl.def("endImport", (void (Tpetra::DistObject<char,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::SrcDistObject &, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool)) &Tpetra::DistObject<char, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::endImport, "C++: Tpetra::DistObject<char, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::endImport(const class Tpetra::SrcDistObject &, const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool) --> void", pybind11::arg("source"), pybind11::arg("exporter"), pybind11::arg("CM"), pybind11::arg("restrictedMode"));
		cl.def("endExport", [](Tpetra::DistObject<char,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> &o, const class Tpetra::SrcDistObject & a0, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > & a1, const enum Tpetra::CombineMode & a2) -> void { return o.endExport(a0, a1, a2); }, "", pybind11::arg("source"), pybind11::arg("importer"), pybind11::arg("CM"));
		cl.def("endExport", (void (Tpetra::DistObject<char,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::SrcDistObject &, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool)) &Tpetra::DistObject<char, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::endExport, "C++: Tpetra::DistObject<char, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::endExport(const class Tpetra::SrcDistObject &, const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &, const enum Tpetra::CombineMode, const bool) --> void", pybind11::arg("source"), pybind11::arg("importer"), pybind11::arg("CM"), pybind11::arg("restrictedMode"));
		cl.def("transferArrived", (bool (Tpetra::DistObject<char,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)() const) &Tpetra::DistObject<char, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::transferArrived, "C++: Tpetra::DistObject<char, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::transferArrived() const --> bool");
		cl.def("isDistributed", (bool (Tpetra::DistObject<char,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)() const) &Tpetra::DistObject<char, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::isDistributed, "C++: Tpetra::DistObject<char, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::isDistributed() const --> bool");
		cl.def("getMap", (class Teuchos::RCP<const class Tpetra::Map<int, long long> > (Tpetra::DistObject<char,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)() const) &Tpetra::DistObject<char, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::getMap, "C++: Tpetra::DistObject<char, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::getMap() const --> class Teuchos::RCP<const class Tpetra::Map<int, long long> >");
		cl.def("description", (std::string (Tpetra::DistObject<char,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)() const) &Tpetra::DistObject<char, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::description, "C++: Tpetra::DistObject<char, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::description() const --> std::string");
		cl.def("describe", [](Tpetra::DistObject<char,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> const &o, class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > & a0) -> void { return o.describe(a0); }, "", pybind11::arg("out"));
		cl.def("describe", (void (Tpetra::DistObject<char,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > &, const enum Teuchos::EVerbosityLevel) const) &Tpetra::DistObject<char, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::describe, "C++: Tpetra::DistObject<char, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::describe(class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > &, const enum Teuchos::EVerbosityLevel) const --> void", pybind11::arg("out"), pybind11::arg("verbLevel"));
		cl.def("removeEmptyProcessesInPlace", (void (Tpetra::DistObject<char,int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Teuchos::RCP<const class Tpetra::Map<int, long long> > &)) &Tpetra::DistObject<char, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::removeEmptyProcessesInPlace, "C++: Tpetra::DistObject<char, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::removeEmptyProcessesInPlace(const class Teuchos::RCP<const class Tpetra::Map<int, long long> > &) --> void", pybind11::arg("newMap"));
	}
}
