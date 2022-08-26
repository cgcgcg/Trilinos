#include <KokkosCompat_ClassicNodeAPI_Wrapper.hpp> // Kokkos::Compat::KokkosDeviceWrapperNode
#include <Kokkos_HostSpace.hpp> // Kokkos::HostSpace
#include <Kokkos_Serial.hpp> // Kokkos::Serial
#include <Teuchos_Array.hpp> // Teuchos::Array
#include <Teuchos_ArrayViewDecl.hpp> // Teuchos::ArrayView
#include <Teuchos_ENull.hpp> // Teuchos::ENull
#include <Teuchos_FilteredIterator.hpp> // Teuchos::FilteredIterator
#include <Teuchos_ParameterEntry.hpp> // Teuchos::ParameterEntry
#include <Teuchos_ParameterEntryValidator.hpp> // Teuchos::ParameterEntryValidator
#include <Teuchos_ParameterList.hpp> // Teuchos::EValidateDefaults
#include <Teuchos_ParameterList.hpp> // Teuchos::EValidateUsed
#include <Teuchos_ParameterList.hpp> // Teuchos::ParameterList
#include <Teuchos_ParameterListModifier.hpp> // Teuchos::ParameterListModifier
#include <Teuchos_PtrDecl.hpp> // Teuchos::Ptr
#include <Teuchos_RCPDecl.hpp> // Teuchos::ERCPUndefinedWeakNoDealloc
#include <Teuchos_RCPDecl.hpp> // Teuchos::ERCPWeakNoDealloc
#include <Teuchos_RCPDecl.hpp> // Teuchos::RCP
#include <Teuchos_RCPNode.hpp> // Teuchos::ERCPNodeLookup
#include <Teuchos_RCPNode.hpp> // Teuchos::ERCPStrength
#include <Teuchos_RCPNode.hpp> // Teuchos::RCPNodeHandle
#include <Teuchos_StringIndexedOrderedValueObjectContainer.hpp> // Teuchos::StringIndexedOrderedValueObjectContainerBase
#include <Teuchos_TwoDArray.hpp> // Teuchos::TwoDArray
#include <Teuchos_any.hpp> // Teuchos::any
#include <Tpetra_CombineMode.hpp> // Tpetra::CombineMode
#include <Tpetra_CombineMode.hpp> // Tpetra::combineModeToString
#include <Tpetra_CombineMode.hpp> // Tpetra::setCombineModeParameter
#include <Tpetra_ConfigDefs.hpp> // Tpetra::EPrivateComputeViewConstructor
#include <Tpetra_ConfigDefs.hpp> // Tpetra::EPrivateHostViewConstructor
#include <Tpetra_ConfigDefs.hpp> // Tpetra::ESweepDirection
#include <Tpetra_ConfigDefs.hpp> // Tpetra::LocalGlobal
#include <Tpetra_ConfigDefs.hpp> // Tpetra::LookupStatus
#include <Tpetra_ConfigDefs.hpp> // Tpetra::OptimizeOption
#include <Tpetra_Packable.hpp> // Tpetra::Packable
#include <Tpetra_SrcDistObject.hpp> // Tpetra::SrcDistObject
#include <cwchar> // (anonymous)
#include <deque> // std::_Deque_iterator
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

// Tpetra::Packable file:Tpetra_Packable.hpp line:96
struct PyCallBack_Tpetra_Packable_char_int_t : public Tpetra::Packable<char,int> {
	using Tpetra::Packable<char,int>::Packable;

	void pack(const class Teuchos::ArrayView<const int> & a0, class Teuchos::Array<char> & a1, const class Teuchos::ArrayView<unsigned long> & a2, unsigned long & a3) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::Packable<char,int> *>(this), "pack");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2, a3);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		pybind11::pybind11_fail("Tried to call pure virtual function \"Packable::pack\"");
	}
};

// Tpetra::Packable file:Tpetra_Packable.hpp line:96
struct PyCallBack_Tpetra_Packable_long_long_int_t : public Tpetra::Packable<long long,int> {
	using Tpetra::Packable<long long,int>::Packable;

	void pack(const class Teuchos::ArrayView<const int> & a0, class Teuchos::Array<long long> & a1, const class Teuchos::ArrayView<unsigned long> & a2, unsigned long & a3) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Tpetra::Packable<long long,int> *>(this), "pack");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2, a3);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		pybind11::pybind11_fail("Tried to call pure virtual function \"Packable::pack\"");
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

	{ // Tpetra::Packable file:Tpetra_Packable.hpp line:96
		pybind11::class_<Tpetra::Packable<char,int>, Teuchos::RCP<Tpetra::Packable<char,int>>, PyCallBack_Tpetra_Packable_char_int_t> cl(M("Tpetra"), "Packable_char_int_t", "");
		cl.def(pybind11::init<PyCallBack_Tpetra_Packable_char_int_t const &>());
		cl.def( pybind11::init( [](){ return new PyCallBack_Tpetra_Packable_char_int_t(); } ) );
		cl.def("pack", (void (Tpetra::Packable<char,int>::*)(const class Teuchos::ArrayView<const int> &, class Teuchos::Array<char> &, const class Teuchos::ArrayView<unsigned long> &, unsigned long &) const) &Tpetra::Packable<char, int>::pack, "C++: Tpetra::Packable<char, int>::pack(const class Teuchos::ArrayView<const int> &, class Teuchos::Array<char> &, const class Teuchos::ArrayView<unsigned long> &, unsigned long &) const --> void", pybind11::arg("exportLIDs"), pybind11::arg("exports"), pybind11::arg("numPacketsPerLID"), pybind11::arg("constantNumPackets"));
		cl.def("assign", (class Tpetra::Packable<char, int> & (Tpetra::Packable<char,int>::*)(const class Tpetra::Packable<char, int> &)) &Tpetra::Packable<char, int>::operator=, "C++: Tpetra::Packable<char, int>::operator=(const class Tpetra::Packable<char, int> &) --> class Tpetra::Packable<char, int> &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // Tpetra::Packable file:Tpetra_Packable.hpp line:96
		pybind11::class_<Tpetra::Packable<long long,int>, Teuchos::RCP<Tpetra::Packable<long long,int>>, PyCallBack_Tpetra_Packable_long_long_int_t> cl(M("Tpetra"), "Packable_long_long_int_t", "");
		cl.def(pybind11::init<PyCallBack_Tpetra_Packable_long_long_int_t const &>());
		cl.def( pybind11::init( [](){ return new PyCallBack_Tpetra_Packable_long_long_int_t(); } ) );
		cl.def("pack", (void (Tpetra::Packable<long long,int>::*)(const class Teuchos::ArrayView<const int> &, class Teuchos::Array<long long> &, const class Teuchos::ArrayView<unsigned long> &, unsigned long &) const) &Tpetra::Packable<long long, int>::pack, "C++: Tpetra::Packable<long long, int>::pack(const class Teuchos::ArrayView<const int> &, class Teuchos::Array<long long> &, const class Teuchos::ArrayView<unsigned long> &, unsigned long &) const --> void", pybind11::arg("exportLIDs"), pybind11::arg("exports"), pybind11::arg("numPacketsPerLID"), pybind11::arg("constantNumPackets"));
		cl.def("assign", (class Tpetra::Packable<long long, int> & (Tpetra::Packable<long long,int>::*)(const class Tpetra::Packable<long long, int> &)) &Tpetra::Packable<long long, int>::operator=, "C++: Tpetra::Packable<long long, int>::operator=(const class Tpetra::Packable<long long, int> &) --> class Tpetra::Packable<long long, int> &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // Tpetra::SrcDistObject file:Tpetra_SrcDistObject.hpp line:89
		pybind11::class_<Tpetra::SrcDistObject, Teuchos::RCP<Tpetra::SrcDistObject>> cl(M("Tpetra"), "SrcDistObject", "Abstract base class for objects that can be the source of\n   an Import or Export operation.\n\n Any object that may be the source of an Import or Export data\n redistribution operation must inherit from this class.  This\n class implements no methods, other than a trivial virtual\n destructor.  If a subclass X inherits from this class, that\n indicates that the subclass can be the source of an Import or\n Export, for some set of subclasses of DistObject.  A\n subclass Y of DistObject which is the target of the Import or\n Export operation will attempt to cast the input source\n SrcDistObject to a subclass which it knows how to treat as a\n source object.  The target subclass Y is responsible for knowing\n what source classes to expect, and how to interpret the\n resulting source object.\n\n DistObject inherits from this class, since a DistObject subclass\n may be either the source or the target of an Import or Export.\n A SrcDistObject subclass which does not inherit from DistObject\n need only be a valid source of an Import or Export; it need not\n be a valid target.\n\n This object compares to the Epetra class Epetra_SrcDistObject.\n Unlike in Epetra, this class in Tpetra does not include\n a getMap() method.  This is for two reasons.  First, consider\n the following inheritance hierarchy: DistObject and RowGraph\n inherit from SrcDistObject, and CrsGraph inherits from\n DistObject and RowGraph.  If SrcDistObject had a virtual getMap\n method, that would make resolution of the method ambiguous.\n Second, it is not necessary for SrcDistObject to have a getMap\n method, because a SrcDistObject alone does not suffice as the\n source of an Import or Export.  Any DistObject subclass must\n cast the SrcDistObject to a subclass which it knows how to treat\n as the source of an Import or Export.  Thus, it's not necessary\n for SrcDistObject to have a getMap method, since it needs to be\n cast anyway before use.  In general, I prefer to keep interfaces\n as simple as possible.");
		cl.def( pybind11::init( [](Tpetra::SrcDistObject const &o){ return new Tpetra::SrcDistObject(o); } ) );
		cl.def( pybind11::init( [](){ return new Tpetra::SrcDistObject(); } ) );
	}
}
