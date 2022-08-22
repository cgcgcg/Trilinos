#include <KokkosCompat_ClassicNodeAPI_Wrapper.hpp>
#include <Kokkos_Concepts.hpp>
#include <Kokkos_HostSpace.hpp>
#include <Kokkos_Serial.hpp>
#include <Kokkos_View.hpp>
#include <Teuchos_ArrayViewDecl.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_Describable.hpp>
#include <Teuchos_ENull.hpp>
#include <Teuchos_FancyOStream.hpp>
#include <Teuchos_FilteredIterator.hpp>
#include <Teuchos_LabeledObject.hpp>
#include <Teuchos_ParameterEntry.hpp>
#include <Teuchos_ParameterEntryValidator.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_ParameterListModifier.hpp>
#include <Teuchos_PtrDecl.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <Teuchos_RCPNode.hpp>
#include <Teuchos_StringIndexedOrderedValueObjectContainer.hpp>
#include <Teuchos_VerbosityLevel.hpp>
#include <Teuchos_any.hpp>
#include <Tpetra_CombineMode.hpp>
#include <Tpetra_ConfigDefs.hpp>
#include <Tpetra_Core.hpp>
#include <Tpetra_Details_LocalMap.hpp>
#include <Tpetra_Directory_decl.hpp>
#include <Tpetra_Map_decl.hpp>
#include <cwchar>
#include <deque>
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

// Tpetra::Directory file:Tpetra_Directory_decl.hpp line:126
struct PyCallBack_Tpetra_Directory_int_long_long_Kokkos_Compat_KokkosDeviceWrapperNode_Kokkos_Serial_Kokkos_HostSpace_t : public Tpetra::Directory<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> {
	using Tpetra::Directory<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::Directory;

	std::string description() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::Directory<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> *>(this), "description");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<std::string>::value) {
				static pybind11::detail::override_caster_t<std::string> caster;
				return pybind11::detail::cast_ref<std::string>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<std::string>(std::move(o));
		}
		return Directory::description();
	}
	void describe(class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > & a0, const enum Teuchos::EVerbosityLevel a1) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::Directory<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> *>(this), "describe");
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
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::Directory<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> *>(this), "setObjectLabel");
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
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::Directory<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>> *>(this), "getObjectLabel");
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

void bind_Tpetra_ConfigDefs(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	// Tpetra::LocalGlobal file:Tpetra_ConfigDefs.hpp line:116
	pybind11::enum_<Tpetra::LocalGlobal>(M("Tpetra"), "LocalGlobal", pybind11::arithmetic(), "Enum for local versus global allocation of Map entries.\n\n  means that the Map's entries are locally\n replicated across all processes.\n\n  means that the Map's entries are globally\n distributed across all processes.")
		.value("LocallyReplicated", Tpetra::LocallyReplicated)
		.value("GloballyDistributed", Tpetra::GloballyDistributed)
		.export_values();

;

	// Tpetra::LookupStatus file:Tpetra_ConfigDefs.hpp line:122
	pybind11::enum_<Tpetra::LookupStatus>(M("Tpetra"), "LookupStatus", pybind11::arithmetic(), "Return status of Map remote index lookup (getRemoteIndexList()).")
		.value("AllIDsPresent", Tpetra::AllIDsPresent)
		.value("IDNotPresent", Tpetra::IDNotPresent)
		.export_values();

;

	// Tpetra::OptimizeOption file:Tpetra_ConfigDefs.hpp line:129
	pybind11::enum_<Tpetra::OptimizeOption>(M("Tpetra"), "OptimizeOption", pybind11::arithmetic(), "Optimize storage option ")
		.value("DoOptimizeStorage", Tpetra::DoOptimizeStorage)
		.value("DoNotOptimizeStorage", Tpetra::DoNotOptimizeStorage)
		.export_values();

;

	// Tpetra::EPrivateComputeViewConstructor file:Tpetra_ConfigDefs.hpp line:134
	pybind11::enum_<Tpetra::EPrivateComputeViewConstructor>(M("Tpetra"), "EPrivateComputeViewConstructor", pybind11::arithmetic(), "")
		.value("COMPUTE_VIEW_CONSTRUCTOR", Tpetra::COMPUTE_VIEW_CONSTRUCTOR)
		.export_values();

;

	// Tpetra::EPrivateHostViewConstructor file:Tpetra_ConfigDefs.hpp line:138
	pybind11::enum_<Tpetra::EPrivateHostViewConstructor>(M("Tpetra"), "EPrivateHostViewConstructor", pybind11::arithmetic(), "")
		.value("HOST_VIEW_CONSTRUCTOR", Tpetra::HOST_VIEW_CONSTRUCTOR)
		.export_values();

;

	// Tpetra::CombineMode file:Tpetra_CombineMode.hpp line:97
	pybind11::enum_<Tpetra::CombineMode>(M("Tpetra"), "CombineMode", pybind11::arithmetic(), "Rule for combining data in an Import or Export\n\n Import or Export (data redistribution) operations might need to\n combine data received from other processes with existing data on\n the calling process.  This enum tells Tpetra how to do that for\n a specific Import or Export operation.  Each Tpetra object may\n interpret the CombineMode in a different way, so you should\n check the Tpetra object's documentation for details.\n\n Here is the list of supported combine modes:\n   - ADD: Sum new values \n   - INSERT: Insert new values that don't currently exist\n   - REPLACE: Replace existing values with new values\n   - ABSMAX: If \n\n is the old value and \n\n     the incoming new value, replace \n with\n     \n\n.\n   - ZERO: Replace old values with zero\n   - ADD_ASSIGN: Do addition assignment (+=) of values into existing values\n     \n\n  \n     May not be supported in all classes\n\n ADD, REPLACE and ADD_ASSIGN are intended for modifying values that already\n exist.  Tpetra objects will generally work correctly if those\n values don't already exist.  (For example, ADD will behave like\n INSERT if the entry does not yet exist on the calling process.)\n However, performance may suffer.\n\n The ZERO combine mode is a special case that bypasses\n communication.  It may seem odd to include a \"combine mode\" that\n doesn't actually combine.  However, this is useful for\n computations like domain decomposition with overlap.  A ZERO\n combine mode with overlap is different than an ADD combine mode\n without overlap.  (See Ifpack2::AdditiveSchwarz, which inspired\n inclusion of this combine mode.)  Furthermore, Import and Export\n also encapsulate a local permutation; if you want only to\n execute the local permutation without communication, you may use\n the ZERO combine mode.")
		.value("ADD", Tpetra::ADD)
		.value("INSERT", Tpetra::INSERT)
		.value("REPLACE", Tpetra::REPLACE)
		.value("ABSMAX", Tpetra::ABSMAX)
		.value("ZERO", Tpetra::ZERO)
		.value("ADD_ASSIGN", Tpetra::ADD_ASSIGN)
		.export_values();

;

	// Tpetra::setCombineModeParameter(class Teuchos::ParameterList &, const std::string &) file:Tpetra_CombineMode.hpp line:130
	M("Tpetra").def("setCombineModeParameter", (void (*)(class Teuchos::ParameterList &, const std::string &)) &Tpetra::setCombineModeParameter, "Set CombineMode parameter in a Teuchos::ParameterList.\n\n If you are constructing a Teuchos::ParameterList with a\n CombineMode parameter, set the parameter by using this function.\n This will use a special feature of Teuchos -- custom parameter\n list validation -- so that users can specify CombineMode values\n by string, rather than enum value.  The strings are the same as\n the enum names: \"ADD\", \"INSERT\", \"REPLACE\", \"ABSMAX\", and\n \"ZERO\".  They are not case sensitive.\n\n Using this function to set a CombineMode parameter will ensure\n that the XML serialization of the resulting\n Teuchos::ParameterList will refer to the CombineMode enum values\n using human-readable string names, rather than raw integers.\n\n \n [out] Teuchos::ParameterList to which you want to\n   add the Tpetra::CombineMode parameter.\n\n \n [in] String name to use for the parameter.  For\n   example, you might wish to call the parameter \"Combine Mode\",\n   \"Tpetra::CombineMode\", or \"combine mode\".  The parameter's\n   name is case sensitive, even though the string values\n   are not.\n\nC++: Tpetra::setCombineModeParameter(class Teuchos::ParameterList &, const std::string &) --> void", pybind11::arg("plist"), pybind11::arg("paramName"));

	// Tpetra::combineModeToString(const enum Tpetra::CombineMode) file:Tpetra_CombineMode.hpp line:134
	M("Tpetra").def("combineModeToString", (std::string (*)(const enum Tpetra::CombineMode)) &Tpetra::combineModeToString, "Human-readable string representation of the given CombineMode.\n\nC++: Tpetra::combineModeToString(const enum Tpetra::CombineMode) --> std::string", pybind11::arg("combineMode"));

	// Tpetra::ESweepDirection file:Tpetra_ConfigDefs.hpp line:232
	pybind11::enum_<Tpetra::ESweepDirection>(M("Tpetra"), "ESweepDirection", pybind11::arithmetic(), "Sweep direction for Gauss-Seidel or Successive Over-Relaxation (SOR).")
		.value("Forward", Tpetra::Forward)
		.value("Backward", Tpetra::Backward)
		.value("Symmetric", Tpetra::Symmetric)
		.export_values();

;

	// Tpetra::getDefaultComm() file:Tpetra_Core.hpp line:69
	M("Tpetra").def("getDefaultComm", (class Teuchos::RCP<const class Teuchos::Comm<int> > (*)()) &Tpetra::getDefaultComm, "Get Tpetra's default communicator.\n\n \n One of the Tpetra::initialize() functions has been called.\n\n \n If one of the versions of initialize() was called that\n   takes a default communicator, this function returns that\n   communicator.  Otherwise, this function returns MPI_COMM_WORLD\n   (wrapped in a Teuchos wrapper) if Trilinos was built with MPI\n   enabled, or a Teuchos::SerialComm instance otherwise.\n\nC++: Tpetra::getDefaultComm() --> class Teuchos::RCP<const class Teuchos::Comm<int> >");

	// Tpetra::isInitialized() file:Tpetra_Core.hpp line:77
	M("Tpetra").def("isInitialized", (bool (*)()) &Tpetra::isInitialized, "Whether Tpetra is in an initialized state.\n\n Initialize Tpetra by calling one of the versions of\n initialize().  After initialize() returns, Tpetra is\n initialized.  Once finalize() returns, Tpetra is no longer\n initialized.\n\nC++: Tpetra::isInitialized() --> bool");

	// Tpetra::finalize() file:Tpetra_Core.hpp line:178
	M("Tpetra").def("finalize", (void (*)()) &Tpetra::finalize, "Finalize Tpetra.\n\n If Tpetra::initialize initialized Kokkos, finalize Kokkos.  If\n Tpetra::initialize initialized MPI, finalize MPI.  Don't call\n this unless you first call Tpetra::initialize.\n\n If you (the user) initialized Kokkos resp. MPI before\n Tpetra::initialize was called, then this function does NOT\n finalize Kokkos resp. MPI.  In that case, you (the user) are\n responsible for finalizing Kokkos resp. MPI.\n\nC++: Tpetra::finalize() --> void");

	{ // Tpetra::ScopeGuard file:Tpetra_Core.hpp line:219
		pybind11::class_<Tpetra::ScopeGuard, Teuchos::RCP<Tpetra::ScopeGuard>> cl(M("Tpetra"), "ScopeGuard", "Scope guard whose destructor automatically calls\n   Tpetra::finalize for you.\n\n This class' constructor does the same thing as\n Tpetra::initialize (see above).  Its destructor automatically\n calls Tpetra::finalize.  This ensures correct Tpetra\n finalization even if intervening code throws an exception.\n\n Compare to Kokkos::ScopeGuard and Teuchos::GlobalMPISession.\n\n Always give the ScopeGuard instance a name.  Otherwise, you'll\n create a temporary object whose destructor will be called right\n away.  That's not what you want.\n\n Here is an example of how to use this class:\n \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
	}
	{ // Tpetra::Directory file:Tpetra_Directory_decl.hpp line:126
		pybind11::class_<Tpetra::Directory<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>, Teuchos::RCP<Tpetra::Directory<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>>, PyCallBack_Tpetra_Directory_int_long_long_Kokkos_Compat_KokkosDeviceWrapperNode_Kokkos_Serial_Kokkos_HostSpace_t> cl(M("Tpetra"), "Directory_int_long_long_Kokkos_Compat_KokkosDeviceWrapperNode_Kokkos_Serial_Kokkos_HostSpace_t", "");
		cl.def( pybind11::init( [](){ return new Tpetra::Directory<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>(); }, [](){ return new PyCallBack_Tpetra_Directory_int_long_long_Kokkos_Compat_KokkosDeviceWrapperNode_Kokkos_Serial_Kokkos_HostSpace_t(); } ) );
		cl.def("initialize", (void (Tpetra::Directory<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::Map<int, long long> &)) &Tpetra::Directory<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::initialize, "C++: Tpetra::Directory<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::initialize(const class Tpetra::Map<int, long long> &) --> void", pybind11::arg("map"));
		cl.def("initialized", (bool (Tpetra::Directory<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)() const) &Tpetra::Directory<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::initialized, "C++: Tpetra::Directory<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::initialized() const --> bool");
		cl.def("description", (std::string (Tpetra::Directory<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)() const) &Tpetra::Directory<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::description, "C++: Tpetra::Directory<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::description() const --> std::string");
		cl.def("getDirectoryEntries", (enum Tpetra::LookupStatus (Tpetra::Directory<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::Map<int, long long> &, const class Teuchos::ArrayView<const long long> &, const class Teuchos::ArrayView<int> &) const) &Tpetra::Directory<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::getDirectoryEntries, "C++: Tpetra::Directory<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::getDirectoryEntries(const class Tpetra::Map<int, long long> &, const class Teuchos::ArrayView<const long long> &, const class Teuchos::ArrayView<int> &) const --> enum Tpetra::LookupStatus", pybind11::arg("map"), pybind11::arg("globalIDs"), pybind11::arg("nodeIDs"));
		cl.def("getDirectoryEntries", (enum Tpetra::LookupStatus (Tpetra::Directory<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::Map<int, long long> &, const class Teuchos::ArrayView<const long long> &, const class Teuchos::ArrayView<int> &, const class Teuchos::ArrayView<int> &) const) &Tpetra::Directory<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::getDirectoryEntries, "C++: Tpetra::Directory<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::getDirectoryEntries(const class Tpetra::Map<int, long long> &, const class Teuchos::ArrayView<const long long> &, const class Teuchos::ArrayView<int> &, const class Teuchos::ArrayView<int> &) const --> enum Tpetra::LookupStatus", pybind11::arg("map"), pybind11::arg("globalIDs"), pybind11::arg("nodeIDs"), pybind11::arg("localIDs"));
		cl.def("isOneToOne", (bool (Tpetra::Directory<int,long long,Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>>::*)(const class Tpetra::Map<int, long long> &) const) &Tpetra::Directory<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::isOneToOne, "C++: Tpetra::Directory<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >::isOneToOne(const class Tpetra::Map<int, long long> &) const --> bool", pybind11::arg("map"));
	}
}
