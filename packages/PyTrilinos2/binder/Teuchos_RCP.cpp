#include <KokkosCompat_ClassicNodeAPI_Wrapper.hpp> // Kokkos::Compat::KokkosDeviceWrapperNode
#include <KokkosSparse_CrsMatrix.hpp> // KokkosSparse::CrsMatrix
#include <KokkosSparse_CrsMatrix.hpp> // KokkosSparse::SparseRowView
#include <KokkosSparse_CrsMatrix.hpp> // KokkosSparse::SparseRowViewConst
#include <Kokkos_AnonymousSpace.hpp> // Kokkos::AnonymousSpace
#include <Kokkos_Concepts.hpp> // Kokkos::Device
#include <Kokkos_DualView.hpp> // Kokkos::DualView
#include <Kokkos_HostSpace.hpp> // Kokkos::HostSpace
#include <Kokkos_Layout.hpp> // Kokkos::LayoutLeft
#include <Kokkos_Layout.hpp> // Kokkos::LayoutRight
#include <Kokkos_Layout.hpp> // Kokkos::LayoutStride
#include <Kokkos_MemoryTraits.hpp> // Kokkos::MemoryTraits
#include <Kokkos_Pair.hpp> // Kokkos::pair
#include <Kokkos_ScratchSpace.hpp> // Kokkos::ScratchMemorySpace
#include <Kokkos_Serial.hpp> // Kokkos::Serial
#include <Kokkos_StaticCrsGraph.hpp> // Kokkos::StaticCrsGraph
#include <Kokkos_View.hpp> // Kokkos::View
#include <Kokkos_View.hpp> // Kokkos::ViewTraits
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
#include <PyTrilinos2_Teuchos_Custom.hpp>
#include <Teuchos_Array.hpp> // Teuchos::Array
#include <Teuchos_ArrayRCPDecl.hpp> // Teuchos::ArrayRCP
#include <Teuchos_ArrayView.hpp> // Teuchos::ArrayView
#include <Teuchos_ArrayViewDecl.hpp> // Teuchos::ArrayView
#include <Teuchos_BLAS_types.hpp> // Teuchos::ETransp
#include <Teuchos_Comm.hpp> // Teuchos::Comm
#include <Teuchos_Comm.hpp> // Teuchos::CommRequest
#include <Teuchos_Comm.hpp> // Teuchos::CommStatus
#include <Teuchos_DataAccess.hpp> // Teuchos::DataAccess
#include <Teuchos_DefaultSerialComm.hpp> // Teuchos::SerialComm
#include <Teuchos_DefaultSerialComm.hpp> // Teuchos::SerialCommStatus
#include <Teuchos_ENull.hpp> // Teuchos::ENull
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
#include <Teuchos_RCP.hpp> // Teuchos::RCP_createNewDeallocRCPNodeRawPtr
#include <Teuchos_RCP.hpp> // Teuchos::RCP_createNewRCPNodeRawPtr
#include <Teuchos_RCP.hpp> // Teuchos::RCP_createNewRCPNodeRawPtrNonowned
#include <Teuchos_RCPDecl.hpp> // Teuchos::DeallocDelete
#include <Teuchos_RCPDecl.hpp> // Teuchos::ERCPUndefinedWeakNoDealloc
#include <Teuchos_RCPDecl.hpp> // Teuchos::ERCPWeakNoDealloc
#include <Teuchos_RCPDecl.hpp> // Teuchos::EmbeddedObjDealloc
#include <Teuchos_RCPDecl.hpp> // Teuchos::RCP
#include <Teuchos_RCPNode.hpp> // Teuchos::EPrePostDestruction
#include <Teuchos_RCPNode.hpp> // Teuchos::ERCPNodeLookup
#include <Teuchos_RCPNode.hpp> // Teuchos::ERCPStrength
#include <Teuchos_RCPNode.hpp> // Teuchos::RCPNode
#include <Teuchos_RCPNode.hpp> // Teuchos::RCPNodeHandle
#include <Teuchos_Range1D.hpp> // 
#include <Teuchos_Range1D.hpp> // Teuchos::Range1D
#include <Teuchos_ReductionOp.hpp> // Teuchos::ValueTypeReductionOp
#include <Teuchos_SerializationTraits.hpp> // Teuchos::SerializationTraits
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
#include <Tpetra_CrsGraph_decl.hpp> // Tpetra::Details::CrsPadding
#include <Tpetra_CrsMatrix_decl.hpp> // Tpetra::CrsMatrix
#include <Tpetra_Details_CrsPadding.hpp> // Tpetra::Details::CrsPadding
#include <Tpetra_Details_FixedHashTable_decl.hpp> // Tpetra::Details::FixedHashTable
#include <Tpetra_Details_LocalMap.hpp> // Tpetra::Details::LocalMap
#include <Tpetra_Details_WrappedDualView.hpp> // Tpetra::Details::WrappedDualView
#include <Tpetra_Directory_decl.hpp> // Tpetra::Directory
#include <Tpetra_Export_decl.hpp> // Tpetra::Export
#include <Tpetra_Import_decl.hpp> // Tpetra::Import
#include <Tpetra_LocalCrsMatrixOperator_decl.hpp> // Tpetra::LocalCrsMatrixOperator
#include <Tpetra_Map_decl.hpp> // Tpetra::Map
#include <Tpetra_MultiVector_decl.hpp> // Tpetra::MultiVector
#include <Tpetra_RowGraph_decl.hpp> // Tpetra::RowGraph
#include <Tpetra_RowGraph_fwd.hpp> // Tpetra::RowGraph
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
#include <Xpetra_Matrix_fwd.hpp> // Xpetra::Matrix
#include <Xpetra_MultiVector_decl.hpp> // Xpetra::MultiVector
#include <Xpetra_MultiVector_fwd.hpp> // Xpetra::MultiVector
#include <Xpetra_Operator.hpp> // Xpetra::Operator
#include <Xpetra_TpetraBlockCrsMatrix_decl.hpp> // Xpetra::TpetraBlockCrsMatrix
#include <Xpetra_TpetraCrsMatrix_decl.hpp> // Xpetra::TpetraCrsMatrix
#include <Xpetra_TpetraMultiVector_decl.hpp> // Xpetra::TpetraMultiVector
#include <cwchar> // (anonymous)
#include <deque> // std::_Deque_iterator
#include <fstream> // std::basic_filebuf
#include <fstream> // std::basic_ofstream
#include <functional> // std::less
#include <impl/Kokkos_SharedAlloc.hpp> // Kokkos::Impl::SharedAllocationRecord
#include <impl/Kokkos_SharedAlloc.hpp> // Kokkos::Impl::SharedAllocationTracker
#include <impl/Kokkos_ViewCtor.hpp> // Kokkos::Impl::ViewCtorProp
#include <impl/Kokkos_ViewMapping.hpp> // Kokkos::Impl::ALL_t
#include <impl/Kokkos_ViewMapping.hpp> // Kokkos::Impl::ViewMapping
#include <ios> // std::_Ios_Openmode
#include <ios> // std::_Ios_Seekdir
#include <ios> // std::fpos
#include <iterator> // __gnu_cxx::__normal_iterator
#include <locale> // std::locale
#include <map> // std::_Rb_tree_const_iterator
#include <map> // std::_Rb_tree_iterator
#include <map> // std::map
#include <memory> // std::allocator
#include <memory> // std::default_delete
#include <memory> // std::shared_ptr
#include <memory> // std::unique_ptr
#include <ostream> // std::basic_ostream
#include <sstream> // std::basic_ostringstream
#include <sstream> // std::basic_stringbuf
#include <streambuf> // std::basic_streambuf
#include <string> // std::basic_string
#include <string> // std::char_traits
#include <typeinfo> // std::type_info
#include <utility> // std::pair
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

void bind_Teuchos_RCP(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	// Teuchos::RCP_createNewRCPNodeRawPtrNonowned(class Teuchos::ParameterEntry *) file:Teuchos_RCP.hpp line:75
	M("Teuchos").def("RCP_createNewRCPNodeRawPtrNonowned", (class Teuchos::RCPNode * (*)(class Teuchos::ParameterEntry *)) &Teuchos::RCP_createNewRCPNodeRawPtrNonowned<Teuchos::ParameterEntry>, "C++: Teuchos::RCP_createNewRCPNodeRawPtrNonowned(class Teuchos::ParameterEntry *) --> class Teuchos::RCPNode *", pybind11::return_value_policy::automatic, pybind11::arg("p"));

	// Teuchos::RCP_createNewRCPNodeRawPtrNonowned(const class Teuchos::ParameterEntry *) file:Teuchos_RCP.hpp line:75
	M("Teuchos").def("RCP_createNewRCPNodeRawPtrNonowned", (class Teuchos::RCPNode * (*)(const class Teuchos::ParameterEntry *)) &Teuchos::RCP_createNewRCPNodeRawPtrNonowned<const Teuchos::ParameterEntry>, "C++: Teuchos::RCP_createNewRCPNodeRawPtrNonowned(const class Teuchos::ParameterEntry *) --> class Teuchos::RCPNode *", pybind11::return_value_policy::automatic, pybind11::arg("p"));

	// Teuchos::RCP_createNewRCPNodeRawPtrNonowned(const class Tpetra::MultiVector<double, int, long long> *) file:Teuchos_RCP.hpp line:75
	M("Teuchos").def("RCP_createNewRCPNodeRawPtrNonowned", (class Teuchos::RCPNode * (*)(const class Tpetra::MultiVector<double, int, long long> *)) &Teuchos::RCP_createNewRCPNodeRawPtrNonowned<const Tpetra::MultiVector<double, int, long long>>, "C++: Teuchos::RCP_createNewRCPNodeRawPtrNonowned(const class Tpetra::MultiVector<double, int, long long> *) --> class Teuchos::RCPNode *", pybind11::return_value_policy::automatic, pybind11::arg("p"));

	// Teuchos::RCP_createNewRCPNodeRawPtrNonowned(const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > *) file:Teuchos_RCP.hpp line:75
	M("Teuchos").def("RCP_createNewRCPNodeRawPtrNonowned", (class Teuchos::RCPNode * (*)(const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > *)) &Teuchos::RCP_createNewRCPNodeRawPtrNonowned<const Tpetra::Import<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >>, "C++: Teuchos::RCP_createNewRCPNodeRawPtrNonowned(const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > *) --> class Teuchos::RCPNode *", pybind11::return_value_policy::automatic, pybind11::arg("p"));

	// Teuchos::RCP_createNewRCPNodeRawPtrNonowned(const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > *) file:Teuchos_RCP.hpp line:75
	M("Teuchos").def("RCP_createNewRCPNodeRawPtrNonowned", (class Teuchos::RCPNode * (*)(const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > *)) &Teuchos::RCP_createNewRCPNodeRawPtrNonowned<const Tpetra::Export<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >>, "C++: Teuchos::RCP_createNewRCPNodeRawPtrNonowned(const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > *) --> class Teuchos::RCPNode *", pybind11::return_value_policy::automatic, pybind11::arg("p"));

	// Teuchos::RCP_createNewRCPNodeRawPtr(class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > *, bool) file:Teuchos_RCP.hpp line:91
	M("Teuchos").def("RCP_createNewRCPNodeRawPtr", (class Teuchos::RCPNode * (*)(class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > *, bool)) &Teuchos::RCP_createNewRCPNodeRawPtr<Teuchos::basic_FancyOStream<char, std::char_traits<char> >>, "C++: Teuchos::RCP_createNewRCPNodeRawPtr(class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > *, bool) --> class Teuchos::RCPNode *", pybind11::return_value_policy::automatic, pybind11::arg("p"), pybind11::arg("has_ownership_in"));

	// Teuchos::RCP_createNewRCPNodeRawPtr(class Teuchos::ParameterList *, bool) file:Teuchos_RCP.hpp line:91
	M("Teuchos").def("RCP_createNewRCPNodeRawPtr", (class Teuchos::RCPNode * (*)(class Teuchos::ParameterList *, bool)) &Teuchos::RCP_createNewRCPNodeRawPtr<Teuchos::ParameterList>, "C++: Teuchos::RCP_createNewRCPNodeRawPtr(class Teuchos::ParameterList *, bool) --> class Teuchos::RCPNode *", pybind11::return_value_policy::automatic, pybind11::arg("p"), pybind11::arg("has_ownership_in"));

	// Teuchos::RCP_createNewRCPNodeRawPtr(class Teuchos::Time *, bool) file:Teuchos_RCP.hpp line:91
	M("Teuchos").def("RCP_createNewRCPNodeRawPtr", (class Teuchos::RCPNode * (*)(class Teuchos::Time *, bool)) &Teuchos::RCP_createNewRCPNodeRawPtr<Teuchos::Time>, "C++: Teuchos::RCP_createNewRCPNodeRawPtr(class Teuchos::Time *, bool) --> class Teuchos::RCPNode *", pybind11::return_value_policy::automatic, pybind11::arg("p"), pybind11::arg("has_ownership_in"));

	// Teuchos::RCP_createNewRCPNodeRawPtr(class Teuchos::TimeMonitorSurrogateImpl *, bool) file:Teuchos_RCP.hpp line:91
	M("Teuchos").def("RCP_createNewRCPNodeRawPtr", (class Teuchos::RCPNode * (*)(class Teuchos::TimeMonitorSurrogateImpl *, bool)) &Teuchos::RCP_createNewRCPNodeRawPtr<Teuchos::TimeMonitorSurrogateImpl>, "C++: Teuchos::RCP_createNewRCPNodeRawPtr(class Teuchos::TimeMonitorSurrogateImpl *, bool) --> class Teuchos::RCPNode *", pybind11::return_value_policy::automatic, pybind11::arg("p"), pybind11::arg("has_ownership_in"));

	// Teuchos::RCP_createNewRCPNodeRawPtr(class MueLu::NoFactory *, bool) file:Teuchos_RCP.hpp line:91
	M("Teuchos").def("RCP_createNewRCPNodeRawPtr", (class Teuchos::RCPNode * (*)(class MueLu::NoFactory *, bool)) &Teuchos::RCP_createNewRCPNodeRawPtr<MueLu::NoFactory>, "C++: Teuchos::RCP_createNewRCPNodeRawPtr(class MueLu::NoFactory *, bool) --> class Teuchos::RCPNode *", pybind11::return_value_policy::automatic, pybind11::arg("p"), pybind11::arg("has_ownership_in"));

	// Teuchos::RCP_createNewRCPNodeRawPtr(class MueLu::MLParameterListInterpreter<double, int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > *, bool) file:Teuchos_RCP.hpp line:91
	M("Teuchos").def("RCP_createNewRCPNodeRawPtr", (class Teuchos::RCPNode * (*)(class MueLu::MLParameterListInterpreter<double, int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > *, bool)) &Teuchos::RCP_createNewRCPNodeRawPtr<MueLu::MLParameterListInterpreter<double, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >>, "C++: Teuchos::RCP_createNewRCPNodeRawPtr(class MueLu::MLParameterListInterpreter<double, int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > *, bool) --> class Teuchos::RCPNode *", pybind11::return_value_policy::automatic, pybind11::arg("p"), pybind11::arg("has_ownership_in"));

	// Teuchos::RCP_createNewRCPNodeRawPtr(class MueLu::ParameterListInterpreter<double, int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > *, bool) file:Teuchos_RCP.hpp line:91
	M("Teuchos").def("RCP_createNewRCPNodeRawPtr", (class Teuchos::RCPNode * (*)(class MueLu::ParameterListInterpreter<double, int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > *, bool)) &Teuchos::RCP_createNewRCPNodeRawPtr<MueLu::ParameterListInterpreter<double, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >>, "C++: Teuchos::RCP_createNewRCPNodeRawPtr(class MueLu::ParameterListInterpreter<double, int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > *, bool) --> class Teuchos::RCPNode *", pybind11::return_value_policy::automatic, pybind11::arg("p"), pybind11::arg("has_ownership_in"));

	// Teuchos::RCP_createNewRCPNodeRawPtr(class std::basic_ofstream<char> *, bool) file:Teuchos_RCP.hpp line:91
	M("Teuchos").def("RCP_createNewRCPNodeRawPtr", (class Teuchos::RCPNode * (*)(class std::basic_ofstream<char> *, bool)) &Teuchos::RCP_createNewRCPNodeRawPtr<std::basic_ofstream<char>>, "C++: Teuchos::RCP_createNewRCPNodeRawPtr(class std::basic_ofstream<char> *, bool) --> class Teuchos::RCPNode *", pybind11::return_value_policy::automatic, pybind11::arg("p"), pybind11::arg("has_ownership_in"));

	// Teuchos::RCP_createNewRCPNodeRawPtr(class MueLu::TpetraOperator<double, int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > *, bool) file:Teuchos_RCP.hpp line:91
	M("Teuchos").def("RCP_createNewRCPNodeRawPtr", (class Teuchos::RCPNode * (*)(class MueLu::TpetraOperator<double, int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > *, bool)) &Teuchos::RCP_createNewRCPNodeRawPtr<MueLu::TpetraOperator<double, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >>, "C++: Teuchos::RCP_createNewRCPNodeRawPtr(class MueLu::TpetraOperator<double, int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > *, bool) --> class Teuchos::RCPNode *", pybind11::return_value_policy::automatic, pybind11::arg("p"), pybind11::arg("has_ownership_in"));

	// Teuchos::RCP_createNewRCPNodeRawPtr(class Teuchos::SerializationTraits<int, unsigned long> *, bool) file:Teuchos_RCP.hpp line:91
	M("Teuchos").def("RCP_createNewRCPNodeRawPtr", (class Teuchos::RCPNode * (*)(class Teuchos::SerializationTraits<int, unsigned long> *, bool)) &Teuchos::RCP_createNewRCPNodeRawPtr<Teuchos::SerializationTraits<int, unsigned long>>, "C++: Teuchos::RCP_createNewRCPNodeRawPtr(class Teuchos::SerializationTraits<int, unsigned long> *, bool) --> class Teuchos::RCPNode *", pybind11::return_value_policy::automatic, pybind11::arg("p"), pybind11::arg("has_ownership_in"));

	// Teuchos::RCP_createNewRCPNodeRawPtr(class Teuchos::SerializationTraits<int, long long> *, bool) file:Teuchos_RCP.hpp line:91
	M("Teuchos").def("RCP_createNewRCPNodeRawPtr", (class Teuchos::RCPNode * (*)(class Teuchos::SerializationTraits<int, long long> *, bool)) &Teuchos::RCP_createNewRCPNodeRawPtr<Teuchos::SerializationTraits<int, long long>>, "C++: Teuchos::RCP_createNewRCPNodeRawPtr(class Teuchos::SerializationTraits<int, long long> *, bool) --> class Teuchos::RCPNode *", pybind11::return_value_policy::automatic, pybind11::arg("p"), pybind11::arg("has_ownership_in"));

	// Teuchos::RCP_createNewRCPNodeRawPtr(const class Teuchos::ValueTypeReductionOp<int, long long> *, bool) file:Teuchos_RCP.hpp line:91
	M("Teuchos").def("RCP_createNewRCPNodeRawPtr", (class Teuchos::RCPNode * (*)(const class Teuchos::ValueTypeReductionOp<int, long long> *, bool)) &Teuchos::RCP_createNewRCPNodeRawPtr<const Teuchos::ValueTypeReductionOp<int, long long>>, "C++: Teuchos::RCP_createNewRCPNodeRawPtr(const class Teuchos::ValueTypeReductionOp<int, long long> *, bool) --> class Teuchos::RCPNode *", pybind11::return_value_policy::automatic, pybind11::arg("p"), pybind11::arg("has_ownership_in"));

	// Teuchos::RCP_createNewRCPNodeRawPtr(class Teuchos::SerialCommStatus<int> *, bool) file:Teuchos_RCP.hpp line:91
	M("Teuchos").def("RCP_createNewRCPNodeRawPtr", (class Teuchos::RCPNode * (*)(class Teuchos::SerialCommStatus<int> *, bool)) &Teuchos::RCP_createNewRCPNodeRawPtr<Teuchos::SerialCommStatus<int>>, "C++: Teuchos::RCP_createNewRCPNodeRawPtr(class Teuchos::SerialCommStatus<int> *, bool) --> class Teuchos::RCPNode *", pybind11::return_value_policy::automatic, pybind11::arg("p"), pybind11::arg("has_ownership_in"));

	// Teuchos::RCP_createNewRCPNodeRawPtr(class Teuchos::SerialComm<int> *, bool) file:Teuchos_RCP.hpp line:91
	M("Teuchos").def("RCP_createNewRCPNodeRawPtr", (class Teuchos::RCPNode * (*)(class Teuchos::SerialComm<int> *, bool)) &Teuchos::RCP_createNewRCPNodeRawPtr<Teuchos::SerialComm<int>>, "C++: Teuchos::RCP_createNewRCPNodeRawPtr(class Teuchos::SerialComm<int> *, bool) --> class Teuchos::RCPNode *", pybind11::return_value_policy::automatic, pybind11::arg("p"), pybind11::arg("has_ownership_in"));

	// Teuchos::RCP_createNewRCPNodeRawPtr(const class Teuchos::Comm<int> *, bool) file:Teuchos_RCP.hpp line:91
	M("Teuchos").def("RCP_createNewRCPNodeRawPtr", (class Teuchos::RCPNode * (*)(const class Teuchos::Comm<int> *, bool)) &Teuchos::RCP_createNewRCPNodeRawPtr<const Teuchos::Comm<int>>, "C++: Teuchos::RCP_createNewRCPNodeRawPtr(const class Teuchos::Comm<int> *, bool) --> class Teuchos::RCPNode *", pybind11::return_value_policy::automatic, pybind11::arg("p"), pybind11::arg("has_ownership_in"));

	// Teuchos::RCP_createNewRCPNodeRawPtr(class Teuchos::basic_OSTab<char, struct std::char_traits<char> > *, bool) file:Teuchos_RCP.hpp line:91
	M("Teuchos").def("RCP_createNewRCPNodeRawPtr", (class Teuchos::RCPNode * (*)(class Teuchos::basic_OSTab<char, struct std::char_traits<char> > *, bool)) &Teuchos::RCP_createNewRCPNodeRawPtr<Teuchos::basic_OSTab<char, std::char_traits<char> >>, "C++: Teuchos::RCP_createNewRCPNodeRawPtr(class Teuchos::basic_OSTab<char, struct std::char_traits<char> > *, bool) --> class Teuchos::RCPNode *", pybind11::return_value_policy::automatic, pybind11::arg("p"), pybind11::arg("has_ownership_in"));

	// Teuchos::RCP_createNewRCPNodeRawPtr(class Tpetra::Map<int, long long> *, bool) file:Teuchos_RCP.hpp line:91
	M("Teuchos").def("RCP_createNewRCPNodeRawPtr", (class Teuchos::RCPNode * (*)(class Tpetra::Map<int, long long> *, bool)) &Teuchos::RCP_createNewRCPNodeRawPtr<Tpetra::Map<int, long long>>, "C++: Teuchos::RCP_createNewRCPNodeRawPtr(class Tpetra::Map<int, long long> *, bool) --> class Teuchos::RCPNode *", pybind11::return_value_policy::automatic, pybind11::arg("p"), pybind11::arg("has_ownership_in"));

	// Teuchos::RCP_createNewRCPNodeRawPtr(class std::basic_ostringstream<char> *, bool) file:Teuchos_RCP.hpp line:91
	M("Teuchos").def("RCP_createNewRCPNodeRawPtr", (class Teuchos::RCPNode * (*)(class std::basic_ostringstream<char> *, bool)) &Teuchos::RCP_createNewRCPNodeRawPtr<std::basic_ostringstream<char>>, "C++: Teuchos::RCP_createNewRCPNodeRawPtr(class std::basic_ostringstream<char> *, bool) --> class Teuchos::RCPNode *", pybind11::return_value_policy::automatic, pybind11::arg("p"), pybind11::arg("has_ownership_in"));

	// Teuchos::RCP_createNewRCPNodeRawPtr(const class Tpetra::Map<int, long long> *, bool) file:Teuchos_RCP.hpp line:91
	M("Teuchos").def("RCP_createNewRCPNodeRawPtr", (class Teuchos::RCPNode * (*)(const class Tpetra::Map<int, long long> *, bool)) &Teuchos::RCP_createNewRCPNodeRawPtr<const Tpetra::Map<int, long long>>, "C++: Teuchos::RCP_createNewRCPNodeRawPtr(const class Tpetra::Map<int, long long> *, bool) --> class Teuchos::RCPNode *", pybind11::return_value_policy::automatic, pybind11::arg("p"), pybind11::arg("has_ownership_in"));

	// Teuchos::RCP_createNewRCPNodeRawPtr(class Tpetra::MultiVector<double, int, long long> *, bool) file:Teuchos_RCP.hpp line:91
	M("Teuchos").def("RCP_createNewRCPNodeRawPtr", (class Teuchos::RCPNode * (*)(class Tpetra::MultiVector<double, int, long long> *, bool)) &Teuchos::RCP_createNewRCPNodeRawPtr<Tpetra::MultiVector<double, int, long long>>, "C++: Teuchos::RCP_createNewRCPNodeRawPtr(class Tpetra::MultiVector<double, int, long long> *, bool) --> class Teuchos::RCPNode *", pybind11::return_value_policy::automatic, pybind11::arg("p"), pybind11::arg("has_ownership_in"));

	// Teuchos::RCP_createNewRCPNodeRawPtr(class Tpetra::CrsGraph<int, long long> *, bool) file:Teuchos_RCP.hpp line:91
	M("Teuchos").def("RCP_createNewRCPNodeRawPtr", (class Teuchos::RCPNode * (*)(class Tpetra::CrsGraph<int, long long> *, bool)) &Teuchos::RCP_createNewRCPNodeRawPtr<Tpetra::CrsGraph<int, long long>>, "C++: Teuchos::RCP_createNewRCPNodeRawPtr(class Tpetra::CrsGraph<int, long long> *, bool) --> class Teuchos::RCPNode *", pybind11::return_value_policy::automatic, pybind11::arg("p"), pybind11::arg("has_ownership_in"));

	// Teuchos::RCP_createNewRCPNodeRawPtr(class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > *, bool) file:Teuchos_RCP.hpp line:91
	M("Teuchos").def("RCP_createNewRCPNodeRawPtr", (class Teuchos::RCPNode * (*)(class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > *, bool)) &Teuchos::RCP_createNewRCPNodeRawPtr<Tpetra::Import<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >>, "C++: Teuchos::RCP_createNewRCPNodeRawPtr(class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > *, bool) --> class Teuchos::RCPNode *", pybind11::return_value_policy::automatic, pybind11::arg("p"), pybind11::arg("has_ownership_in"));

	// Teuchos::RCP_createNewRCPNodeRawPtr(class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > *, bool) file:Teuchos_RCP.hpp line:91
	M("Teuchos").def("RCP_createNewRCPNodeRawPtr", (class Teuchos::RCPNode * (*)(class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > *, bool)) &Teuchos::RCP_createNewRCPNodeRawPtr<Tpetra::Export<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >>, "C++: Teuchos::RCP_createNewRCPNodeRawPtr(class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > *, bool) --> class Teuchos::RCPNode *", pybind11::return_value_policy::automatic, pybind11::arg("p"), pybind11::arg("has_ownership_in"));

	// Teuchos::RCP_createNewRCPNodeRawPtr(class Tpetra::CrsMatrix<double, int, long long> *, bool) file:Teuchos_RCP.hpp line:91
	M("Teuchos").def("RCP_createNewRCPNodeRawPtr", (class Teuchos::RCPNode * (*)(class Tpetra::CrsMatrix<double, int, long long> *, bool)) &Teuchos::RCP_createNewRCPNodeRawPtr<Tpetra::CrsMatrix<double, int, long long>>, "C++: Teuchos::RCP_createNewRCPNodeRawPtr(class Tpetra::CrsMatrix<double, int, long long> *, bool) --> class Teuchos::RCPNode *", pybind11::return_value_policy::automatic, pybind11::arg("p"), pybind11::arg("has_ownership_in"));

	// Teuchos::RCP_createNewRCPNodeRawPtr(class Teuchos::SerializationTraits<int, char> *, bool) file:Teuchos_RCP.hpp line:91
	M("Teuchos").def("RCP_createNewRCPNodeRawPtr", (class Teuchos::RCPNode * (*)(class Teuchos::SerializationTraits<int, char> *, bool)) &Teuchos::RCP_createNewRCPNodeRawPtr<Teuchos::SerializationTraits<int, char>>, "C++: Teuchos::RCP_createNewRCPNodeRawPtr(class Teuchos::SerializationTraits<int, char> *, bool) --> class Teuchos::RCPNode *", pybind11::return_value_policy::automatic, pybind11::arg("p"), pybind11::arg("has_ownership_in"));

	// Teuchos::RCP_createNewDeallocRCPNodeRawPtr(class Teuchos::ParameterList *, class Teuchos::EmbeddedObjDealloc<class Teuchos::ParameterList, class Teuchos::RCP<class Teuchos::ParameterList>, class Teuchos::DeallocDelete<class Teuchos::ParameterList> >, bool) file:Teuchos_RCP.hpp line:99
	M("Teuchos").def("RCP_createNewDeallocRCPNodeRawPtr", (class Teuchos::RCPNode * (*)(class Teuchos::ParameterList *, class Teuchos::EmbeddedObjDealloc<class Teuchos::ParameterList, class Teuchos::RCP<class Teuchos::ParameterList>, class Teuchos::DeallocDelete<class Teuchos::ParameterList> >, bool)) &Teuchos::RCP_createNewDeallocRCPNodeRawPtr<Teuchos::ParameterList,Teuchos::EmbeddedObjDealloc<Teuchos::ParameterList, Teuchos::RCP<Teuchos::ParameterList>, Teuchos::DeallocDelete<Teuchos::ParameterList> >>, "C++: Teuchos::RCP_createNewDeallocRCPNodeRawPtr(class Teuchos::ParameterList *, class Teuchos::EmbeddedObjDealloc<class Teuchos::ParameterList, class Teuchos::RCP<class Teuchos::ParameterList>, class Teuchos::DeallocDelete<class Teuchos::ParameterList> >, bool) --> class Teuchos::RCPNode *", pybind11::return_value_policy::automatic, pybind11::arg("p"), pybind11::arg("dealloc"), pybind11::arg("has_ownership_in"));

	// Teuchos::RCP_createNewDeallocRCPNodeRawPtr(const class Teuchos::ParameterList *, class Teuchos::EmbeddedObjDealloc<const class Teuchos::ParameterList, class Teuchos::RCP<const class Teuchos::ParameterList>, class Teuchos::DeallocDelete<const class Teuchos::ParameterList> >, bool) file:Teuchos_RCP.hpp line:99
	M("Teuchos").def("RCP_createNewDeallocRCPNodeRawPtr", (class Teuchos::RCPNode * (*)(const class Teuchos::ParameterList *, class Teuchos::EmbeddedObjDealloc<const class Teuchos::ParameterList, class Teuchos::RCP<const class Teuchos::ParameterList>, class Teuchos::DeallocDelete<const class Teuchos::ParameterList> >, bool)) &Teuchos::RCP_createNewDeallocRCPNodeRawPtr<const Teuchos::ParameterList,Teuchos::EmbeddedObjDealloc<const Teuchos::ParameterList, Teuchos::RCP<const Teuchos::ParameterList>, Teuchos::DeallocDelete<const Teuchos::ParameterList> >>, "C++: Teuchos::RCP_createNewDeallocRCPNodeRawPtr(const class Teuchos::ParameterList *, class Teuchos::EmbeddedObjDealloc<const class Teuchos::ParameterList, class Teuchos::RCP<const class Teuchos::ParameterList>, class Teuchos::DeallocDelete<const class Teuchos::ParameterList> >, bool) --> class Teuchos::RCPNode *", pybind11::return_value_policy::automatic, pybind11::arg("p"), pybind11::arg("dealloc"), pybind11::arg("has_ownership_in"));

}
