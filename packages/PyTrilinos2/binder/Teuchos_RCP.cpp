#include <KokkosCompat_ClassicNodeAPI_Wrapper.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <Kokkos_AnonymousSpace.hpp>
#include <Kokkos_Concepts.hpp>
#include <Kokkos_DualView.hpp>
#include <Kokkos_HostSpace.hpp>
#include <Kokkos_Layout.hpp>
#include <Kokkos_MemoryTraits.hpp>
#include <Kokkos_Pair.hpp>
#include <Kokkos_ScratchSpace.hpp>
#include <Kokkos_Serial.hpp>
#include <Kokkos_StaticCrsGraph.hpp>
#include <Kokkos_View.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_ArrayRCPDecl.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_ArrayViewDecl.hpp>
#include <Teuchos_BLAS_types.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_DataAccess.hpp>
#include <Teuchos_DefaultSerialComm.hpp>
#include <Teuchos_ENull.hpp>
#include <Teuchos_FancyOStream.hpp>
#include <Teuchos_FilteredIterator.hpp>
#include <PyTrilinos2_Teuchos_Custom.hpp>
#include <Teuchos_ParameterEntry.hpp>
#include <Teuchos_ParameterEntryValidator.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_ParameterListModifier.hpp>
#include <Teuchos_PtrDecl.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <Teuchos_RCPNode.hpp>
#include <Teuchos_Range1D.hpp>
#include <Teuchos_ReductionOp.hpp>
#include <Teuchos_SerializationTraits.hpp>
#include <Teuchos_StringIndexedOrderedValueObjectContainer.hpp>
#include <Teuchos_Time.hpp>
#include <Teuchos_TimeMonitor.hpp>
#include <Teuchos_VerbosityLevel.hpp>
#include <Teuchos_any.hpp>
#include <Tpetra_Access.hpp>
#include <Tpetra_CombineMode.hpp>
#include <Tpetra_ConfigDefs.hpp>
#include <Tpetra_CrsGraph_decl.hpp>
#include <Tpetra_CrsMatrix_decl.hpp>
#include <Tpetra_Details_CrsPadding.hpp>
#include <Tpetra_Details_FixedHashTable_decl.hpp>
#include <Tpetra_Details_LocalMap.hpp>
#include <Tpetra_Details_WrappedDualView.hpp>
#include <Tpetra_Directory_decl.hpp>
#include <Tpetra_Export_decl.hpp>
#include <Tpetra_Import_decl.hpp>
#include <Tpetra_LocalCrsMatrixOperator_decl.hpp>
#include <Tpetra_Map_decl.hpp>
#include <Tpetra_MultiVector_decl.hpp>
#include <Tpetra_RowGraph_decl.hpp>
#include <Tpetra_RowMatrix_decl.hpp>
#include <Tpetra_SrcDistObject.hpp>
#include <Tpetra_Vector_decl.hpp>
#include <cwchar>
#include <deque>
#include <impl/Kokkos_SharedAlloc.hpp>
#include <impl/Kokkos_ViewCtor.hpp>
#include <impl/Kokkos_ViewMapping.hpp>
#include <ios>
#include <iterator>
#include <locale>
#include <memory>
#include <ostream>
#include <sstream>
#include <streambuf>
#include <string>
#include <typeinfo>
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

	// Teuchos::RCP_createNewRCPNodeRawPtrNonowned(const class Tpetra::Vector<double, int, long long> *) file:Teuchos_RCP.hpp line:75
	M("Teuchos").def("RCP_createNewRCPNodeRawPtrNonowned", (class Teuchos::RCPNode * (*)(const class Tpetra::Vector<double, int, long long> *)) &Teuchos::RCP_createNewRCPNodeRawPtrNonowned<const Tpetra::Vector<double, int, long long>>, "C++: Teuchos::RCP_createNewRCPNodeRawPtrNonowned(const class Tpetra::Vector<double, int, long long> *) --> class Teuchos::RCPNode *", pybind11::return_value_policy::automatic, pybind11::arg("p"));

	// Teuchos::RCP_createNewRCPNodeRawPtr(class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > *, bool) file:Teuchos_RCP.hpp line:91
	M("Teuchos").def("RCP_createNewRCPNodeRawPtr", (class Teuchos::RCPNode * (*)(class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > *, bool)) &Teuchos::RCP_createNewRCPNodeRawPtr<Teuchos::basic_FancyOStream<char, std::char_traits<char> >>, "C++: Teuchos::RCP_createNewRCPNodeRawPtr(class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > *, bool) --> class Teuchos::RCPNode *", pybind11::return_value_policy::automatic, pybind11::arg("p"), pybind11::arg("has_ownership_in"));

	// Teuchos::RCP_createNewRCPNodeRawPtr(class Teuchos::ParameterList *, bool) file:Teuchos_RCP.hpp line:91
	M("Teuchos").def("RCP_createNewRCPNodeRawPtr", (class Teuchos::RCPNode * (*)(class Teuchos::ParameterList *, bool)) &Teuchos::RCP_createNewRCPNodeRawPtr<Teuchos::ParameterList>, "C++: Teuchos::RCP_createNewRCPNodeRawPtr(class Teuchos::ParameterList *, bool) --> class Teuchos::RCPNode *", pybind11::return_value_policy::automatic, pybind11::arg("p"), pybind11::arg("has_ownership_in"));

	// Teuchos::RCP_createNewRCPNodeRawPtr(class Teuchos::Time *, bool) file:Teuchos_RCP.hpp line:91
	M("Teuchos").def("RCP_createNewRCPNodeRawPtr", (class Teuchos::RCPNode * (*)(class Teuchos::Time *, bool)) &Teuchos::RCP_createNewRCPNodeRawPtr<Teuchos::Time>, "C++: Teuchos::RCP_createNewRCPNodeRawPtr(class Teuchos::Time *, bool) --> class Teuchos::RCPNode *", pybind11::return_value_policy::automatic, pybind11::arg("p"), pybind11::arg("has_ownership_in"));

	// Teuchos::RCP_createNewRCPNodeRawPtr(class Teuchos::TimeMonitorSurrogateImpl *, bool) file:Teuchos_RCP.hpp line:91
	M("Teuchos").def("RCP_createNewRCPNodeRawPtr", (class Teuchos::RCPNode * (*)(class Teuchos::TimeMonitorSurrogateImpl *, bool)) &Teuchos::RCP_createNewRCPNodeRawPtr<Teuchos::TimeMonitorSurrogateImpl>, "C++: Teuchos::RCP_createNewRCPNodeRawPtr(class Teuchos::TimeMonitorSurrogateImpl *, bool) --> class Teuchos::RCPNode *", pybind11::return_value_policy::automatic, pybind11::arg("p"), pybind11::arg("has_ownership_in"));

	// Teuchos::RCP_createNewRCPNodeRawPtr(class Tpetra::Directory<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > *, bool) file:Teuchos_RCP.hpp line:91
	M("Teuchos").def("RCP_createNewRCPNodeRawPtr", (class Teuchos::RCPNode * (*)(class Tpetra::Directory<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > *, bool)) &Teuchos::RCP_createNewRCPNodeRawPtr<Tpetra::Directory<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >>, "C++: Teuchos::RCP_createNewRCPNodeRawPtr(class Tpetra::Directory<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > *, bool) --> class Teuchos::RCPNode *", pybind11::return_value_policy::automatic, pybind11::arg("p"), pybind11::arg("has_ownership_in"));

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

	// Teuchos::RCP_createNewRCPNodeRawPtr(class Tpetra::Vector<double, int, long long> *, bool) file:Teuchos_RCP.hpp line:91
	M("Teuchos").def("RCP_createNewRCPNodeRawPtr", (class Teuchos::RCPNode * (*)(class Tpetra::Vector<double, int, long long> *, bool)) &Teuchos::RCP_createNewRCPNodeRawPtr<Tpetra::Vector<double, int, long long>>, "C++: Teuchos::RCP_createNewRCPNodeRawPtr(class Tpetra::Vector<double, int, long long> *, bool) --> class Teuchos::RCPNode *", pybind11::return_value_policy::automatic, pybind11::arg("p"), pybind11::arg("has_ownership_in"));

	// Teuchos::RCP_createNewRCPNodeRawPtr(class Teuchos::SerializationTraits<int, char> *, bool) file:Teuchos_RCP.hpp line:91
	M("Teuchos").def("RCP_createNewRCPNodeRawPtr", (class Teuchos::RCPNode * (*)(class Teuchos::SerializationTraits<int, char> *, bool)) &Teuchos::RCP_createNewRCPNodeRawPtr<Teuchos::SerializationTraits<int, char>>, "C++: Teuchos::RCP_createNewRCPNodeRawPtr(class Teuchos::SerializationTraits<int, char> *, bool) --> class Teuchos::RCPNode *", pybind11::return_value_policy::automatic, pybind11::arg("p"), pybind11::arg("has_ownership_in"));

	// Teuchos::RCP_createNewDeallocRCPNodeRawPtr(class Teuchos::ParameterList *, class Teuchos::EmbeddedObjDealloc<class Teuchos::ParameterList, class Teuchos::RCP<class Teuchos::ParameterList>, class Teuchos::DeallocDelete<class Teuchos::ParameterList> >, bool) file:Teuchos_RCP.hpp line:99
	M("Teuchos").def("RCP_createNewDeallocRCPNodeRawPtr", (class Teuchos::RCPNode * (*)(class Teuchos::ParameterList *, class Teuchos::EmbeddedObjDealloc<class Teuchos::ParameterList, class Teuchos::RCP<class Teuchos::ParameterList>, class Teuchos::DeallocDelete<class Teuchos::ParameterList> >, bool)) &Teuchos::RCP_createNewDeallocRCPNodeRawPtr<Teuchos::ParameterList,Teuchos::EmbeddedObjDealloc<Teuchos::ParameterList, Teuchos::RCP<Teuchos::ParameterList>, Teuchos::DeallocDelete<Teuchos::ParameterList> >>, "C++: Teuchos::RCP_createNewDeallocRCPNodeRawPtr(class Teuchos::ParameterList *, class Teuchos::EmbeddedObjDealloc<class Teuchos::ParameterList, class Teuchos::RCP<class Teuchos::ParameterList>, class Teuchos::DeallocDelete<class Teuchos::ParameterList> >, bool) --> class Teuchos::RCPNode *", pybind11::return_value_policy::automatic, pybind11::arg("p"), pybind11::arg("dealloc"), pybind11::arg("has_ownership_in"));

	// Teuchos::RCP_createNewDeallocRCPNodeRawPtr(const class Teuchos::ParameterList *, class Teuchos::EmbeddedObjDealloc<const class Teuchos::ParameterList, class Teuchos::RCP<const class Teuchos::ParameterList>, class Teuchos::DeallocDelete<const class Teuchos::ParameterList> >, bool) file:Teuchos_RCP.hpp line:99
	M("Teuchos").def("RCP_createNewDeallocRCPNodeRawPtr", (class Teuchos::RCPNode * (*)(const class Teuchos::ParameterList *, class Teuchos::EmbeddedObjDealloc<const class Teuchos::ParameterList, class Teuchos::RCP<const class Teuchos::ParameterList>, class Teuchos::DeallocDelete<const class Teuchos::ParameterList> >, bool)) &Teuchos::RCP_createNewDeallocRCPNodeRawPtr<const Teuchos::ParameterList,Teuchos::EmbeddedObjDealloc<const Teuchos::ParameterList, Teuchos::RCP<const Teuchos::ParameterList>, Teuchos::DeallocDelete<const Teuchos::ParameterList> >>, "C++: Teuchos::RCP_createNewDeallocRCPNodeRawPtr(const class Teuchos::ParameterList *, class Teuchos::EmbeddedObjDealloc<const class Teuchos::ParameterList, class Teuchos::RCP<const class Teuchos::ParameterList>, class Teuchos::DeallocDelete<const class Teuchos::ParameterList> >, bool) --> class Teuchos::RCPNode *", pybind11::return_value_policy::automatic, pybind11::arg("p"), pybind11::arg("dealloc"), pybind11::arg("has_ownership_in"));

}
