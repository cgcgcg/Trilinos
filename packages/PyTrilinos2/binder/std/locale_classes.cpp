#include <KokkosCompat_ClassicNodeAPI_Wrapper.hpp> // Kokkos::Compat::KokkosDeviceWrapperNode
#include <KokkosSparse_CrsMatrix.hpp> // KokkosSparse::CrsMatrix
#include <Kokkos_Concepts.hpp> // Kokkos::Device
#include <Kokkos_DualView.hpp> // Kokkos::DualView
#include <Kokkos_HostSpace.hpp> // Kokkos::HostSpace
#include <Kokkos_Layout.hpp> // Kokkos::LayoutLeft
#include <Kokkos_MemoryTraits.hpp> // Kokkos::MemoryTraits
#include <Kokkos_Serial.hpp> // Kokkos::Serial
#include <Kokkos_StaticCrsGraph.hpp> // Kokkos::StaticCrsGraph
#include <Kokkos_Tuners.hpp> // Kokkos::Tools::Experimental::Impl::ValueHierarchyNode
#include <Kokkos_View.hpp> // Kokkos::View
#include <MueLu_Aggregates_decl.hpp> // MueLu::Aggregates
#include <MueLu_FacadeClassFactory_decl.hpp> // MueLu::FacadeClassFactory
#include <MueLu_FactoryBase.hpp> // MueLu::FactoryBase
#include <MueLu_FactoryFactory_fwd.hpp> // MueLu::FactoryFactory
#include <MueLu_FactoryManagerBase.hpp> // MueLu::FactoryManagerBase
#include <MueLu_FactoryManager_decl.hpp> // MueLu::FactoryManager
#include <MueLu_Hierarchy_decl.hpp> // MueLu::Hierarchy
#include <MueLu_Level.hpp> // 
#include <MueLu_Level.hpp> // MueLu::Level
#include <MueLu_MLParameterListInterpreter_decl.hpp> // MueLu::MLParameterListInterpreter
#include <MueLu_NoFactory.hpp> // MueLu::NoFactory
#include <MueLu_ParameterListInterpreter_decl.hpp> // MueLu::ParameterListInterpreter
#include <MueLu_TpetraOperator_decl.hpp> // MueLu::TpetraOperator
#include <MueLu_Types.hpp> // MueLu::CycleType
#include <MueLu_VariableContainer.hpp> // MueLu::VariableContainer
#include <Teuchos_Array.hpp> // Teuchos::Array
#include <Teuchos_ArrayRCPDecl.hpp> // Teuchos::ArrayRCP
#include <Teuchos_ArrayViewDecl.hpp> // Teuchos::ArrayView
#include <Teuchos_BLAS_types.hpp> // Teuchos::ETransp
#include <Teuchos_Comm.hpp> // Teuchos::Comm
#include <Teuchos_Comm.hpp> // Teuchos::CommRequest
#include <Teuchos_Comm.hpp> // Teuchos::CommStatus
#include <Teuchos_DataAccess.hpp> // Teuchos::DataAccess
#include <Teuchos_DefaultSerialComm.hpp> // Teuchos::SerialComm
#include <Teuchos_DefaultSerialComm.hpp> // Teuchos::SerialCommStatus
#include <Teuchos_Dependency.hpp> // Teuchos::Dependency
#include <Teuchos_ENull.hpp> // Teuchos::ENull
#include <Teuchos_EReductionType.hpp> // Teuchos::EReductionType
#include <Teuchos_FancyOStream.hpp> // Teuchos::basic_FancyOStream
#include <Teuchos_FancyOStream.hpp> // Teuchos::basic_OSTab
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
#include <Teuchos_RCPDecl.hpp> // Teuchos::RCPComp
#include <Teuchos_RCPNode.hpp> // Teuchos::EPrePostDestruction
#include <Teuchos_RCPNode.hpp> // Teuchos::ERCPNodeLookup
#include <Teuchos_RCPNode.hpp> // Teuchos::ERCPStrength
#include <Teuchos_RCPNode.hpp> // Teuchos::RCPNode
#include <Teuchos_RCPNode.hpp> // Teuchos::RCPNodeHandle
#include <Teuchos_Range1D.hpp> // Teuchos::Range1D
#include <Teuchos_ReductionOp.hpp> // Teuchos::ValueTypeReductionOp
#include <Teuchos_SerializationTraits.hpp> // Teuchos::SerializationTraits
#include <Teuchos_StackedTimer.hpp> // Teuchos::StackedTimer
#include <Teuchos_StandardParameterEntryValidators.hpp> // 
#include <Teuchos_StringIndexedOrderedValueObjectContainer.hpp> // Teuchos::StringIndexedOrderedValueObjectContainerBase
#include <Teuchos_Time.hpp> // Teuchos::Time
#include <Teuchos_TimeMonitor.hpp> // Teuchos::TimeMonitorSurrogateImpl
#include <Teuchos_TwoDArray.hpp> // Teuchos::TwoDArray
#include <Teuchos_VerbosityLevel.hpp> // Teuchos::EVerbosityLevel
#include <Teuchos_any.hpp> // Teuchos::any
#include <Tpetra_Access.hpp> // Tpetra::Access::OverwriteAllStruct
#include <Tpetra_Access.hpp> // Tpetra::Access::ReadOnlyStruct
#include <Tpetra_Access.hpp> // Tpetra::Access::ReadWriteStruct
#include <Tpetra_BlockCrsMatrix_decl.hpp> // Tpetra::BlockCrsMatrix
#include <Tpetra_CombineMode.hpp> // Tpetra::CombineMode
#include <Tpetra_ConfigDefs.hpp> // Tpetra::LocalGlobal
#include <Tpetra_ConfigDefs.hpp> // Tpetra::LookupStatus
#include <Tpetra_CrsGraph_decl.hpp> // Tpetra::CrsGraph
#include <Tpetra_CrsMatrix_decl.hpp> // Tpetra::CrsMatrix
#include <Tpetra_Details_CrsPadding.hpp> // Tpetra::Details::CrsPadding
#include <Tpetra_Details_LocalMap.hpp> // Tpetra::Details::LocalMap
#include <Tpetra_Details_WrappedDualView.hpp> // Tpetra::Details::WrappedDualView
#include <Tpetra_Directory_decl.hpp> // Tpetra::Directory
#include <Tpetra_Export_decl.hpp> // Tpetra::Export
#include <Tpetra_Import_decl.hpp> // Tpetra::Import
#include <Tpetra_LocalCrsMatrixOperator_decl.hpp> // Tpetra::LocalCrsMatrixOperator
#include <Tpetra_Map_decl.hpp> // Tpetra::Map
#include <Tpetra_MultiVector_decl.hpp> // Tpetra::MultiVector
#include <Tpetra_RowGraph_decl.hpp> // Tpetra::RowGraph
#include <Tpetra_RowMatrix_decl.hpp> // Tpetra::RowMatrix
#include <Tpetra_SrcDistObject.hpp> // Tpetra::SrcDistObject
#include <Tpetra_Vector_decl.hpp> // Tpetra::Vector
#include <Xpetra_Access.hpp> // Xpetra::Access::OverwriteAllStruct
#include <Xpetra_Access.hpp> // Xpetra::Access::ReadOnlyStruct
#include <Xpetra_Access.hpp> // Xpetra::Access::ReadWriteStruct
#include <Xpetra_ConfigDefs.hpp> // Xpetra::CombineMode
#include <Xpetra_ConfigDefs.hpp> // Xpetra::LookupStatus
#include <Xpetra_CrsGraph.hpp> // Xpetra::CrsGraph
#include <Xpetra_CrsMatrix.hpp> // Xpetra::CrsMatrix
#include <Xpetra_CrsMatrixWrap_decl.hpp> // Xpetra::CrsMatrixWrap
#include <Xpetra_DistObject.hpp> // Xpetra::DistObject
#include <Xpetra_Export.hpp> // Xpetra::Export
#include <Xpetra_Import.hpp> // Xpetra::Import
#include <Xpetra_Map_decl.hpp> // Xpetra::Map
#include <Xpetra_Map_decl.hpp> // Xpetra::UnderlyingLib
#include <Xpetra_Matrix.hpp> // Xpetra::Matrix
#include <Xpetra_MatrixView.hpp> // Xpetra::MatrixView
#include <Xpetra_MultiVector_decl.hpp> // Xpetra::MultiVector
#include <Xpetra_MultiVector_fwd.hpp> // Xpetra::MultiVector
#include <Xpetra_Operator.hpp> // Xpetra::Operator
#include <Xpetra_TpetraBlockCrsMatrix_decl.hpp> // Xpetra::TpetraBlockCrsMatrix
#include <Xpetra_TpetraCrsMatrix_decl.hpp> // Xpetra::TpetraCrsMatrix
#include <Xpetra_TpetraMultiVector_decl.hpp> // Xpetra::TpetraMultiVector
#include <chrono> // std::chrono::duration
#include <cwchar> // (anonymous)
#include <deque> // std::_Deque_iterator
#include <fstream> // std::basic_filebuf
#include <fstream> // std::basic_ofstream
#include <functional> // std::less
#include <iomanip> // std::_Setbase
#include <iomanip> // std::_Setprecision
#include <iomanip> // std::_Setw
#include <ios> // std::_Ios_Fmtflags
#include <ios> // std::_Ios_Iostate
#include <ios> // std::_Ios_Openmode
#include <ios> // std::_Ios_Seekdir
#include <ios> // std::boolalpha
#include <ios> // std::dec
#include <ios> // std::defaultfloat
#include <ios> // std::fixed
#include <ios> // std::fpos
#include <ios> // std::hex
#include <ios> // std::hexfloat
#include <ios> // std::internal
#include <ios> // std::io_errc
#include <ios> // std::ios_base
#include <ios> // std::ios_base::Init
#include <ios> // std::ios_base::failure
#include <ios> // std::is_error_code_enum
#include <ios> // std::left
#include <ios> // std::make_error_code
#include <ios> // std::make_error_condition
#include <ios> // std::noboolalpha
#include <ios> // std::noshowbase
#include <ios> // std::noshowpoint
#include <ios> // std::noshowpos
#include <ios> // std::noskipws
#include <ios> // std::nounitbuf
#include <ios> // std::nouppercase
#include <ios> // std::oct
#include <ios> // std::right
#include <ios> // std::scientific
#include <ios> // std::showbase
#include <ios> // std::showpoint
#include <ios> // std::showpos
#include <ios> // std::skipws
#include <ios> // std::unitbuf
#include <ios> // std::uppercase
#include <iterator> // __gnu_cxx::__normal_iterator
#include <iterator> // std::move_iterator
#include <locale> // std::collate
#include <locale> // std::collate_byname
#include <locale> // std::locale
#include <map> // std::_Rb_tree_const_iterator
#include <map> // std::_Rb_tree_iterator
#include <map> // std::map
#include <memory> // std::allocator
#include <memory> // std::default_delete
#include <memory> // std::shared_ptr
#include <memory> // std::unique_ptr
#include <ostream> // std::basic_ostream
#include <random> // std::bernoulli_distribution
#include <ratio> // std::ratio
#include <set> // std::set
#include <sstream> // __str__
#include <sstream> // std::basic_ostringstream
#include <sstream> // std::basic_stringbuf
#include <stdexcept> // std::domain_error
#include <stdexcept> // std::invalid_argument
#include <stdexcept> // std::length_error
#include <stdexcept> // std::logic_error
#include <stdexcept> // std::out_of_range
#include <stdexcept> // std::overflow_error
#include <stdexcept> // std::range_error
#include <stdexcept> // std::runtime_error
#include <stdexcept> // std::underflow_error
#include <streambuf> // std::basic_streambuf
#include <string> // std::basic_string
#include <string> // std::char_traits
#include <system_error> // std::_V2::error_category
#include <system_error> // std::errc
#include <system_error> // std::error_code
#include <system_error> // std::error_condition
#include <system_error> // std::hash
#include <system_error> // std::is_error_code_enum
#include <system_error> // std::is_error_condition_enum
#include <system_error> // std::make_error_code
#include <system_error> // std::make_error_condition
#include <system_error> // std::system_error
#include <thread> // std::thread
#include <typeinfo> // std::type_info
#include <unordered_map> // std::__detail::_Node_const_iterator
#include <unordered_map> // std::__detail::_Node_iterator
#include <utility> // std::pair
#include <vector> // std::_Bit_iterator
#include <vector> // std::_Bit_iterator_base
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

// std::logic_error file:stdexcept line:113
struct PyCallBack_std_logic_error : public std::logic_error {
	using std::logic_error::logic_error;

	const char * what() const noexcept override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const std::logic_error *>(this), "what");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<const char *>::value) {
				static pybind11::detail::override_caster_t<const char *> caster;
				return pybind11::detail::cast_ref<const char *>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<const char *>(std::move(o));
		}
		return logic_error::what();
	}
};

// std::invalid_argument file:stdexcept line:158
struct PyCallBack_std_invalid_argument : public std::invalid_argument {
	using std::invalid_argument::invalid_argument;

	const char * what() const noexcept override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const std::invalid_argument *>(this), "what");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<const char *>::value) {
				static pybind11::detail::override_caster_t<const char *> caster;
				return pybind11::detail::cast_ref<const char *>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<const char *>(std::move(o));
		}
		return logic_error::what();
	}
};

// std::runtime_error file:stdexcept line:197
struct PyCallBack_std_runtime_error : public std::runtime_error {
	using std::runtime_error::runtime_error;

	const char * what() const noexcept override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const std::runtime_error *>(this), "what");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<const char *>::value) {
				static pybind11::detail::override_caster_t<const char *> caster;
				return pybind11::detail::cast_ref<const char *>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<const char *>(std::move(o));
		}
		return runtime_error::what();
	}
};

// std::basic_streambuf file:bits/streambuf.tcc line:149
struct PyCallBack_std_streambuf : public std::streambuf {
	using std::streambuf::basic_streambuf;

	void imbue(const class std::locale & a0) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const std::streambuf *>(this), "imbue");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return basic_streambuf::imbue(a0);
	}
	class std::basic_streambuf<char> * setbuf(char * a0, long a1) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const std::streambuf *>(this), "setbuf");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<class std::basic_streambuf<char> *>::value) {
				static pybind11::detail::override_caster_t<class std::basic_streambuf<char> *> caster;
				return pybind11::detail::cast_ref<class std::basic_streambuf<char> *>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<class std::basic_streambuf<char> *>(std::move(o));
		}
		return basic_streambuf::setbuf(a0, a1);
	}
	class std::fpos<__mbstate_t> seekoff(long a0, enum std::_Ios_Seekdir a1, enum std::_Ios_Openmode a2) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const std::streambuf *>(this), "seekoff");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<class std::fpos<__mbstate_t>>::value) {
				static pybind11::detail::override_caster_t<class std::fpos<__mbstate_t>> caster;
				return pybind11::detail::cast_ref<class std::fpos<__mbstate_t>>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<class std::fpos<__mbstate_t>>(std::move(o));
		}
		return basic_streambuf::seekoff(a0, a1, a2);
	}
	class std::fpos<__mbstate_t> seekpos(class std::fpos<__mbstate_t> a0, enum std::_Ios_Openmode a1) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const std::streambuf *>(this), "seekpos");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<class std::fpos<__mbstate_t>>::value) {
				static pybind11::detail::override_caster_t<class std::fpos<__mbstate_t>> caster;
				return pybind11::detail::cast_ref<class std::fpos<__mbstate_t>>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<class std::fpos<__mbstate_t>>(std::move(o));
		}
		return basic_streambuf::seekpos(a0, a1);
	}
	int sync() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const std::streambuf *>(this), "sync");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<int>::value) {
				static pybind11::detail::override_caster_t<int> caster;
				return pybind11::detail::cast_ref<int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<int>(std::move(o));
		}
		return basic_streambuf::sync();
	}
	long showmanyc() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const std::streambuf *>(this), "showmanyc");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<long>::value) {
				static pybind11::detail::override_caster_t<long> caster;
				return pybind11::detail::cast_ref<long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long>(std::move(o));
		}
		return basic_streambuf::showmanyc();
	}
	long xsgetn(char * a0, long a1) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const std::streambuf *>(this), "xsgetn");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<long>::value) {
				static pybind11::detail::override_caster_t<long> caster;
				return pybind11::detail::cast_ref<long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long>(std::move(o));
		}
		return basic_streambuf::xsgetn(a0, a1);
	}
	int underflow() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const std::streambuf *>(this), "underflow");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<int>::value) {
				static pybind11::detail::override_caster_t<int> caster;
				return pybind11::detail::cast_ref<int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<int>(std::move(o));
		}
		return basic_streambuf::underflow();
	}
	int uflow() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const std::streambuf *>(this), "uflow");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<int>::value) {
				static pybind11::detail::override_caster_t<int> caster;
				return pybind11::detail::cast_ref<int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<int>(std::move(o));
		}
		return basic_streambuf::uflow();
	}
	int pbackfail(int a0) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const std::streambuf *>(this), "pbackfail");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<int>::value) {
				static pybind11::detail::override_caster_t<int> caster;
				return pybind11::detail::cast_ref<int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<int>(std::move(o));
		}
		return basic_streambuf::pbackfail(a0);
	}
	long xsputn(const char * a0, long a1) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const std::streambuf *>(this), "xsputn");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<long>::value) {
				static pybind11::detail::override_caster_t<long> caster;
				return pybind11::detail::cast_ref<long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long>(std::move(o));
		}
		return basic_streambuf::xsputn(a0, a1);
	}
	int overflow(int a0) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const std::streambuf *>(this), "overflow");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<int>::value) {
				static pybind11::detail::override_caster_t<int> caster;
				return pybind11::detail::cast_ref<int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<int>(std::move(o));
		}
		return basic_streambuf::overflow(a0);
	}
};

void bind_std_locale_classes(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // std::locale file:bits/locale_classes.h line:62
		pybind11::class_<std::locale, std::shared_ptr<std::locale>> cl(M("std"), "locale", "");
		cl.def( pybind11::init( [](){ return new std::locale(); } ) );
		cl.def( pybind11::init( [](std::locale const &o){ return new std::locale(o); } ) );
		cl.def( pybind11::init<const char *>(), pybind11::arg("__s") );

		cl.def( pybind11::init<const class std::locale &, const char *, int>(), pybind11::arg("__base"), pybind11::arg("__s"), pybind11::arg("__cat") );

		cl.def( pybind11::init<const std::string &>(), pybind11::arg("__s") );

		cl.def( pybind11::init<const class std::locale &, const std::string &, int>(), pybind11::arg("__base"), pybind11::arg("__s"), pybind11::arg("__cat") );

		cl.def( pybind11::init<const class std::locale &, const class std::locale &, int>(), pybind11::arg("__base"), pybind11::arg("__add"), pybind11::arg("__cat") );

		cl.def("assign", (const class std::locale & (std::locale::*)(const class std::locale &)) &std::locale::operator=, "C++: std::locale::operator=(const class std::locale &) --> const class std::locale &", pybind11::return_value_policy::automatic, pybind11::arg("__other"));
		cl.def("name", (std::string (std::locale::*)() const) &std::locale::name, "C++: std::locale::name() const --> std::string");
		cl.def("__eq__", (bool (std::locale::*)(const class std::locale &) const) &std::locale::operator==, "C++: std::locale::operator==(const class std::locale &) const --> bool", pybind11::arg("__other"));
		cl.def("__ne__", (bool (std::locale::*)(const class std::locale &) const) &std::locale::operator!=, "C++: std::locale::operator!=(const class std::locale &) const --> bool", pybind11::arg("__other"));
		cl.def_static("global", (class std::locale (*)(const class std::locale &)) &std::locale::global, "C++: std::locale::global(const class std::locale &) --> class std::locale", pybind11::arg("__loc"));
		cl.def_static("classic", (const class std::locale & (*)()) &std::locale::classic, "C++: std::locale::classic() --> const class std::locale &", pybind11::return_value_policy::automatic);

		{ // std::locale::id file:bits/locale_classes.h line:483
			auto & enclosing_class = cl;
			pybind11::class_<std::locale::id, std::shared_ptr<std::locale::id>> cl(enclosing_class, "id", "");
			cl.def( pybind11::init( [](){ return new std::locale::id(); } ) );
			cl.def("_M_id", (unsigned long (std::locale::id::*)() const) &std::locale::id::_M_id, "C++: std::locale::id::_M_id() const --> unsigned long");
		}

		{ // std::locale::_Impl file:bits/locale_classes.h line:522
			auto & enclosing_class = cl;
			pybind11::class_<std::locale::_Impl, std::locale::_Impl*> cl(enclosing_class, "_Impl", "");
		}

	}
	{ // std::logic_error file:stdexcept line:113
		pybind11::class_<std::logic_error, std::shared_ptr<std::logic_error>, PyCallBack_std_logic_error, std::exception> cl(M("std"), "logic_error", "");
		cl.def( pybind11::init<const std::string &>(), pybind11::arg("__arg") );

		cl.def( pybind11::init<const char *>(), pybind11::arg("") );

		cl.def( pybind11::init( [](PyCallBack_std_logic_error const &o){ return new PyCallBack_std_logic_error(o); } ) );
		cl.def( pybind11::init( [](std::logic_error const &o){ return new std::logic_error(o); } ) );
		cl.def("assign", (class std::logic_error & (std::logic_error::*)(const class std::logic_error &)) &std::logic_error::operator=, "C++: std::logic_error::operator=(const class std::logic_error &) --> class std::logic_error &", pybind11::return_value_policy::automatic, pybind11::arg(""));
		cl.def("what", (const char * (std::logic_error::*)() const) &std::logic_error::what, "C++: std::logic_error::what() const --> const char *", pybind11::return_value_policy::automatic);
	}
	{ // std::invalid_argument file:stdexcept line:158
		pybind11::class_<std::invalid_argument, std::shared_ptr<std::invalid_argument>, PyCallBack_std_invalid_argument, std::logic_error> cl(M("std"), "invalid_argument", "");
		cl.def( pybind11::init<const std::string &>(), pybind11::arg("__arg") );

		cl.def( pybind11::init<const char *>(), pybind11::arg("") );

		cl.def( pybind11::init( [](PyCallBack_std_invalid_argument const &o){ return new PyCallBack_std_invalid_argument(o); } ) );
		cl.def( pybind11::init( [](std::invalid_argument const &o){ return new std::invalid_argument(o); } ) );
		cl.def("assign", (class std::invalid_argument & (std::invalid_argument::*)(const class std::invalid_argument &)) &std::invalid_argument::operator=, "C++: std::invalid_argument::operator=(const class std::invalid_argument &) --> class std::invalid_argument &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // std::runtime_error file:stdexcept line:197
		pybind11::class_<std::runtime_error, std::shared_ptr<std::runtime_error>, PyCallBack_std_runtime_error, std::exception> cl(M("std"), "runtime_error", "");
		cl.def( pybind11::init<const std::string &>(), pybind11::arg("__arg") );

		cl.def( pybind11::init<const char *>(), pybind11::arg("") );

		cl.def( pybind11::init( [](PyCallBack_std_runtime_error const &o){ return new PyCallBack_std_runtime_error(o); } ) );
		cl.def( pybind11::init( [](std::runtime_error const &o){ return new std::runtime_error(o); } ) );
		cl.def("assign", (class std::runtime_error & (std::runtime_error::*)(const class std::runtime_error &)) &std::runtime_error::operator=, "C++: std::runtime_error::operator=(const class std::runtime_error &) --> class std::runtime_error &", pybind11::return_value_policy::automatic, pybind11::arg(""));
		cl.def("what", (const char * (std::runtime_error::*)() const) &std::runtime_error::what, "C++: std::runtime_error::what() const --> const char *", pybind11::return_value_policy::automatic);
	}
	// std::_Ios_Fmtflags file:bits/ios_base.h line:57
	pybind11::enum_<std::_Ios_Fmtflags>(M("std"), "_Ios_Fmtflags", pybind11::arithmetic(), "")
		.value("_S_boolalpha", std::_S_boolalpha)
		.value("_S_dec", std::_S_dec)
		.value("_S_fixed", std::_S_fixed)
		.value("_S_hex", std::_S_hex)
		.value("_S_internal", std::_S_internal)
		.value("_S_left", std::_S_left)
		.value("_S_oct", std::_S_oct)
		.value("_S_right", std::_S_right)
		.value("_S_scientific", std::_S_scientific)
		.value("_S_showbase", std::_S_showbase)
		.value("_S_showpoint", std::_S_showpoint)
		.value("_S_showpos", std::_S_showpos)
		.value("_S_skipws", std::_S_skipws)
		.value("_S_unitbuf", std::_S_unitbuf)
		.value("_S_uppercase", std::_S_uppercase)
		.value("_S_adjustfield", std::_S_adjustfield)
		.value("_S_basefield", std::_S_basefield)
		.value("_S_floatfield", std::_S_floatfield)
		.value("_S_ios_fmtflags_end", std::_S_ios_fmtflags_end)
		.value("_S_ios_fmtflags_max", std::_S_ios_fmtflags_max)
		.value("_S_ios_fmtflags_min", std::_S_ios_fmtflags_min)
		.export_values();

;

	// std::_Ios_Openmode file:bits/ios_base.h line:111
	pybind11::enum_<std::_Ios_Openmode>(M("std"), "_Ios_Openmode", pybind11::arithmetic(), "")
		.value("_S_app", std::_S_app)
		.value("_S_ate", std::_S_ate)
		.value("_S_bin", std::_S_bin)
		.value("_S_in", std::_S_in)
		.value("_S_out", std::_S_out)
		.value("_S_trunc", std::_S_trunc)
		.value("_S_ios_openmode_end", std::_S_ios_openmode_end)
		.value("_S_ios_openmode_max", std::_S_ios_openmode_max)
		.value("_S_ios_openmode_min", std::_S_ios_openmode_min)
		.export_values();

;

	// std::_Ios_Seekdir file:bits/ios_base.h line:193
	pybind11::enum_<std::_Ios_Seekdir>(M("std"), "_Ios_Seekdir", pybind11::arithmetic(), "")
		.value("_S_beg", std::_S_beg)
		.value("_S_cur", std::_S_cur)
		.value("_S_end", std::_S_end)
		.value("_S_ios_seekdir_end", std::_S_ios_seekdir_end)
		.export_values();

;

	{ // std::basic_streambuf file:bits/streambuf.tcc line:149
		pybind11::class_<std::streambuf, std::shared_ptr<std::streambuf>, PyCallBack_std_streambuf> cl(M("std"), "streambuf", "");
		cl.def("pubimbue", (class std::locale (std::streambuf::*)(const class std::locale &)) &std::basic_streambuf<char, std::char_traits<char> >::pubimbue, "C++: std::basic_streambuf<char, std::char_traits<char> >::pubimbue(const class std::locale &) --> class std::locale", pybind11::arg("__loc"));
		cl.def("getloc", (class std::locale (std::streambuf::*)() const) &std::basic_streambuf<char, std::char_traits<char> >::getloc, "C++: std::basic_streambuf<char, std::char_traits<char> >::getloc() const --> class std::locale");
		cl.def("pubsetbuf", (class std::basic_streambuf<char> * (std::streambuf::*)(char *, long)) &std::basic_streambuf<char, std::char_traits<char> >::pubsetbuf, "C++: std::basic_streambuf<char, std::char_traits<char> >::pubsetbuf(char *, long) --> class std::basic_streambuf<char> *", pybind11::return_value_policy::automatic, pybind11::arg("__s"), pybind11::arg("__n"));
		cl.def("pubseekoff", [](std::streambuf &o, long const & a0, enum std::_Ios_Seekdir const & a1) -> std::fpos<__mbstate_t> { return o.pubseekoff(a0, a1); }, "", pybind11::arg("__off"), pybind11::arg("__way"));
		cl.def("pubseekoff", (class std::fpos<__mbstate_t> (std::streambuf::*)(long, enum std::_Ios_Seekdir, enum std::_Ios_Openmode)) &std::basic_streambuf<char, std::char_traits<char> >::pubseekoff, "C++: std::basic_streambuf<char, std::char_traits<char> >::pubseekoff(long, enum std::_Ios_Seekdir, enum std::_Ios_Openmode) --> class std::fpos<__mbstate_t>", pybind11::arg("__off"), pybind11::arg("__way"), pybind11::arg("__mode"));
		cl.def("pubseekpos", [](std::streambuf &o, class std::fpos<__mbstate_t> const & a0) -> std::fpos<__mbstate_t> { return o.pubseekpos(a0); }, "", pybind11::arg("__sp"));
		cl.def("pubseekpos", (class std::fpos<__mbstate_t> (std::streambuf::*)(class std::fpos<__mbstate_t>, enum std::_Ios_Openmode)) &std::basic_streambuf<char, std::char_traits<char> >::pubseekpos, "C++: std::basic_streambuf<char, std::char_traits<char> >::pubseekpos(class std::fpos<__mbstate_t>, enum std::_Ios_Openmode) --> class std::fpos<__mbstate_t>", pybind11::arg("__sp"), pybind11::arg("__mode"));
		cl.def("pubsync", (int (std::streambuf::*)()) &std::basic_streambuf<char, std::char_traits<char> >::pubsync, "C++: std::basic_streambuf<char, std::char_traits<char> >::pubsync() --> int");
		cl.def("in_avail", (long (std::streambuf::*)()) &std::basic_streambuf<char, std::char_traits<char> >::in_avail, "C++: std::basic_streambuf<char, std::char_traits<char> >::in_avail() --> long");
		cl.def("snextc", (int (std::streambuf::*)()) &std::basic_streambuf<char, std::char_traits<char> >::snextc, "C++: std::basic_streambuf<char, std::char_traits<char> >::snextc() --> int");
		cl.def("sbumpc", (int (std::streambuf::*)()) &std::basic_streambuf<char, std::char_traits<char> >::sbumpc, "C++: std::basic_streambuf<char, std::char_traits<char> >::sbumpc() --> int");
		cl.def("sgetc", (int (std::streambuf::*)()) &std::basic_streambuf<char, std::char_traits<char> >::sgetc, "C++: std::basic_streambuf<char, std::char_traits<char> >::sgetc() --> int");
		cl.def("sgetn", (long (std::streambuf::*)(char *, long)) &std::basic_streambuf<char, std::char_traits<char> >::sgetn, "C++: std::basic_streambuf<char, std::char_traits<char> >::sgetn(char *, long) --> long", pybind11::arg("__s"), pybind11::arg("__n"));
		cl.def("sputbackc", (int (std::streambuf::*)(char)) &std::basic_streambuf<char, std::char_traits<char> >::sputbackc, "C++: std::basic_streambuf<char, std::char_traits<char> >::sputbackc(char) --> int", pybind11::arg("__c"));
		cl.def("sungetc", (int (std::streambuf::*)()) &std::basic_streambuf<char, std::char_traits<char> >::sungetc, "C++: std::basic_streambuf<char, std::char_traits<char> >::sungetc() --> int");
		cl.def("sputc", (int (std::streambuf::*)(char)) &std::basic_streambuf<char, std::char_traits<char> >::sputc, "C++: std::basic_streambuf<char, std::char_traits<char> >::sputc(char) --> int", pybind11::arg("__c"));
		cl.def("sputn", (long (std::streambuf::*)(const char *, long)) &std::basic_streambuf<char, std::char_traits<char> >::sputn, "C++: std::basic_streambuf<char, std::char_traits<char> >::sputn(const char *, long) --> long", pybind11::arg("__s"), pybind11::arg("__n"));
	}
}
