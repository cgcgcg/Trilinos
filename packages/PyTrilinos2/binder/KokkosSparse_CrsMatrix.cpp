#include <KokkosSparse_CrsMatrix.hpp> // KokkosSparse::CrsMatrix
#include <Kokkos_AnonymousSpace.hpp> // Kokkos::AnonymousSpace
#include <Kokkos_Concepts.hpp> // Kokkos::Device
#include <Kokkos_HostSpace.hpp> // 
#include <Kokkos_HostSpace.hpp> // Kokkos::HostSpace
#include <Kokkos_Layout.hpp> // Kokkos::LayoutLeft
#include <Kokkos_Layout.hpp> // Kokkos::LayoutRight
#include <Kokkos_ScratchSpace.hpp> // Kokkos::ScratchMemorySpace
#include <Kokkos_Serial.hpp> // Kokkos::Serial
#include <Kokkos_View.hpp> // Kokkos::View
#include <Kokkos_View.hpp> // Kokkos::ViewTraits
#include <impl/Kokkos_SharedAlloc.hpp> // Kokkos::Impl::SharedAllocationRecord
#include <impl/Kokkos_SharedAlloc.hpp> // Kokkos::Impl::SharedAllocationTracker
#include <impl/Kokkos_ViewCtor.hpp> // Kokkos::Impl::ViewCtorProp
#include <impl/Kokkos_ViewMapping.hpp> // Kokkos::Impl::ViewMapping
#include <iterator> // __gnu_cxx::__normal_iterator
#include <memory> // std::allocator
#include <sstream> // __str__
#include <string> // std::basic_string
#include <string> // std::char_traits

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

void bind_KokkosSparse_CrsMatrix(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // KokkosSparse::CrsMatrix file:KokkosSparse_CrsMatrix.hpp line:375
		pybind11::class_<KokkosSparse::CrsMatrix<double,int,Kokkos::HostSpace,void,unsigned long>, Teuchos::RCP<KokkosSparse::CrsMatrix<double,int,Kokkos::HostSpace,void,unsigned long>>> cl(M("KokkosSparse"), "CrsMatrix_double_int_Kokkos_HostSpace_void_unsigned_long_t", "");
		cl.def( pybind11::init( [](){ return new KokkosSparse::CrsMatrix<double,int,Kokkos::HostSpace,void,unsigned long>(); } ) );
		cl.def( pybind11::init<const std::string &, int, int, unsigned long, double *, int *, int *>(), pybind11::arg(""), pybind11::arg("nrows"), pybind11::arg("ncols"), pybind11::arg("annz"), pybind11::arg("val"), pybind11::arg("rowmap"), pybind11::arg("cols") );

		cl.def( pybind11::init( [](KokkosSparse::CrsMatrix<double,int,Kokkos::HostSpace,void,unsigned long> const &o){ return new KokkosSparse::CrsMatrix<double,int,Kokkos::HostSpace,void,unsigned long>(o); } ) );
		cl.def_readwrite("graph", &KokkosSparse::CrsMatrix<double,int,Kokkos::HostSpace,void,unsigned long>::graph);
		cl.def_readwrite("values", &KokkosSparse::CrsMatrix<double,int,Kokkos::HostSpace,void,unsigned long>::values);
		cl.def_readwrite("dev_config", &KokkosSparse::CrsMatrix<double,int,Kokkos::HostSpace,void,unsigned long>::dev_config);
		cl.def("numRows", (int (KokkosSparse::CrsMatrix<double,int,Kokkos::HostSpace,void,unsigned long>::*)() const) &KokkosSparse::CrsMatrix<double, int, Kokkos::HostSpace, void, unsigned long>::numRows, "C++: KokkosSparse::CrsMatrix<double, int, Kokkos::HostSpace, void, unsigned long>::numRows() const --> int");
		cl.def("numCols", (int (KokkosSparse::CrsMatrix<double,int,Kokkos::HostSpace,void,unsigned long>::*)() const) &KokkosSparse::CrsMatrix<double, int, Kokkos::HostSpace, void, unsigned long>::numCols, "C++: KokkosSparse::CrsMatrix<double, int, Kokkos::HostSpace, void, unsigned long>::numCols() const --> int");
		cl.def("numPointRows", (int (KokkosSparse::CrsMatrix<double,int,Kokkos::HostSpace,void,unsigned long>::*)() const) &KokkosSparse::CrsMatrix<double, int, Kokkos::HostSpace, void, unsigned long>::numPointRows, "C++: KokkosSparse::CrsMatrix<double, int, Kokkos::HostSpace, void, unsigned long>::numPointRows() const --> int");
		cl.def("numPointCols", (int (KokkosSparse::CrsMatrix<double,int,Kokkos::HostSpace,void,unsigned long>::*)() const) &KokkosSparse::CrsMatrix<double, int, Kokkos::HostSpace, void, unsigned long>::numPointCols, "C++: KokkosSparse::CrsMatrix<double, int, Kokkos::HostSpace, void, unsigned long>::numPointCols() const --> int");
		cl.def("nnz", (unsigned long (KokkosSparse::CrsMatrix<double,int,Kokkos::HostSpace,void,unsigned long>::*)() const) &KokkosSparse::CrsMatrix<double, int, Kokkos::HostSpace, void, unsigned long>::nnz, "C++: KokkosSparse::CrsMatrix<double, int, Kokkos::HostSpace, void, unsigned long>::nnz() const --> unsigned long");
	}
}
