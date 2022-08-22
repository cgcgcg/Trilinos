#include <KokkosSparse_CrsMatrix.hpp>
#include <Kokkos_AnonymousSpace.hpp>
#include <Kokkos_Concepts.hpp>
#include <Kokkos_HostSpace.hpp>
#include <Kokkos_Layout.hpp>
#include <Kokkos_ScratchSpace.hpp>
#include <Kokkos_Serial.hpp>
#include <Kokkos_View.hpp>
#include <impl/Kokkos_SharedAlloc.hpp>
#include <impl/Kokkos_ViewCtor.hpp>
#include <impl/Kokkos_ViewMapping.hpp>
#include <iterator>
#include <memory>
#include <sstream> // __str__
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

void bind_KokkosSparse_CrsMatrix(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // KokkosSparse::SparseRowView file:KokkosSparse_CrsMatrix.hpp line:167
		pybind11::class_<KokkosSparse::SparseRowView<KokkosSparse::CrsMatrix<double, int, Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>, void, unsigned long>>, Teuchos::RCP<KokkosSparse::SparseRowView<KokkosSparse::CrsMatrix<double, int, Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>, void, unsigned long>>>> cl(M("KokkosSparse"), "SparseRowView_KokkosSparse_CrsMatrix_double_int_Kokkos_Device_Kokkos_Serial_Kokkos_HostSpace_void_unsigned_long_t", "");
		cl.def( pybind11::init<double *const, int *const, const int &, const int &>(), pybind11::arg("values"), pybind11::arg("colidx__"), pybind11::arg("stride"), pybind11::arg("count") );

		cl.def( pybind11::init( [](KokkosSparse::SparseRowView<KokkosSparse::CrsMatrix<double, int, Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>, void, unsigned long>> const &o){ return new KokkosSparse::SparseRowView<KokkosSparse::CrsMatrix<double, int, Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>, void, unsigned long>>(o); } ) );
		cl.def_readonly("length", &KokkosSparse::SparseRowView<KokkosSparse::CrsMatrix<double, int, Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>, void, unsigned long>>::length);
		cl.def("value", (double & (KokkosSparse::SparseRowView<KokkosSparse::CrsMatrix<double, int, Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>, void, unsigned long>>::*)(const int &) const) &KokkosSparse::SparseRowView<KokkosSparse::CrsMatrix<double, int, Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>, void, unsigned long> >::value, "C++: KokkosSparse::SparseRowView<KokkosSparse::CrsMatrix<double, int, Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>, void, unsigned long> >::value(const int &) const --> double &", pybind11::return_value_policy::automatic, pybind11::arg("i"));
		cl.def("colidx", (int & (KokkosSparse::SparseRowView<KokkosSparse::CrsMatrix<double, int, Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>, void, unsigned long>>::*)(const int &) const) &KokkosSparse::SparseRowView<KokkosSparse::CrsMatrix<double, int, Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>, void, unsigned long> >::colidx, "C++: KokkosSparse::SparseRowView<KokkosSparse::CrsMatrix<double, int, Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>, void, unsigned long> >::colidx(const int &) const --> int &", pybind11::return_value_policy::automatic, pybind11::arg("i"));
	}
	{ // KokkosSparse::SparseRowViewConst file:KokkosSparse_CrsMatrix.hpp line:267
		pybind11::class_<KokkosSparse::SparseRowViewConst<KokkosSparse::CrsMatrix<double, int, Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>, void, unsigned long>>, Teuchos::RCP<KokkosSparse::SparseRowViewConst<KokkosSparse::CrsMatrix<double, int, Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>, void, unsigned long>>>> cl(M("KokkosSparse"), "SparseRowViewConst_KokkosSparse_CrsMatrix_double_int_Kokkos_Device_Kokkos_Serial_Kokkos_HostSpace_void_unsigned_long_t", "");
		cl.def( pybind11::init<const double *const, const int *const, const int &, const int &>(), pybind11::arg("values"), pybind11::arg("colidx__"), pybind11::arg("stride"), pybind11::arg("count") );

		cl.def( pybind11::init( [](KokkosSparse::SparseRowViewConst<KokkosSparse::CrsMatrix<double, int, Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>, void, unsigned long>> const &o){ return new KokkosSparse::SparseRowViewConst<KokkosSparse::CrsMatrix<double, int, Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>, void, unsigned long>>(o); } ) );
		cl.def_readonly("length", &KokkosSparse::SparseRowViewConst<KokkosSparse::CrsMatrix<double, int, Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>, void, unsigned long>>::length);
		cl.def("value", (const double & (KokkosSparse::SparseRowViewConst<KokkosSparse::CrsMatrix<double, int, Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>, void, unsigned long>>::*)(const int &) const) &KokkosSparse::SparseRowViewConst<KokkosSparse::CrsMatrix<double, int, Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>, void, unsigned long> >::value, "C++: KokkosSparse::SparseRowViewConst<KokkosSparse::CrsMatrix<double, int, Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>, void, unsigned long> >::value(const int &) const --> const double &", pybind11::return_value_policy::automatic, pybind11::arg("i"));
		cl.def("colidx", (const int & (KokkosSparse::SparseRowViewConst<KokkosSparse::CrsMatrix<double, int, Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>, void, unsigned long>>::*)(const int &) const) &KokkosSparse::SparseRowViewConst<KokkosSparse::CrsMatrix<double, int, Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>, void, unsigned long> >::colidx, "C++: KokkosSparse::SparseRowViewConst<KokkosSparse::CrsMatrix<double, int, Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>, void, unsigned long> >::colidx(const int &) const --> const int &", pybind11::return_value_policy::automatic, pybind11::arg("i"));
	}
	{ // KokkosSparse::CrsMatrix file:KokkosSparse_CrsMatrix.hpp line:375
		pybind11::class_<KokkosSparse::CrsMatrix<double,int,Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>,void,unsigned long>, Teuchos::RCP<KokkosSparse::CrsMatrix<double,int,Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>,void,unsigned long>>> cl(M("KokkosSparse"), "CrsMatrix_double_int_Kokkos_Device_Kokkos_Serial_Kokkos_HostSpace_void_unsigned_long_t", "");
		cl.def( pybind11::init( [](){ return new KokkosSparse::CrsMatrix<double,int,Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>,void,unsigned long>(); } ) );
		cl.def( pybind11::init<const std::string &, int, int, unsigned long, double *, int *, int *>(), pybind11::arg(""), pybind11::arg("nrows"), pybind11::arg("ncols"), pybind11::arg("annz"), pybind11::arg("val"), pybind11::arg("rowmap"), pybind11::arg("cols") );

		cl.def( pybind11::init( [](KokkosSparse::CrsMatrix<double,int,Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>,void,unsigned long> const &o){ return new KokkosSparse::CrsMatrix<double,int,Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>,void,unsigned long>(o); } ) );
		cl.def_readwrite("graph", &KokkosSparse::CrsMatrix<double,int,Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>,void,unsigned long>::graph);
		cl.def_readwrite("values", &KokkosSparse::CrsMatrix<double,int,Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>,void,unsigned long>::values);
		cl.def_readwrite("dev_config", &KokkosSparse::CrsMatrix<double,int,Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>,void,unsigned long>::dev_config);
		cl.def("assign", (class KokkosSparse::CrsMatrix<double, int, struct Kokkos::Device<class Kokkos::Serial, class Kokkos::HostSpace>, void, unsigned long> & (KokkosSparse::CrsMatrix<double,int,Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>,void,unsigned long>::*)(const class KokkosSparse::CrsMatrix<double, int, struct Kokkos::Device<class Kokkos::Serial, class Kokkos::HostSpace>, void, unsigned long> &)) &KokkosSparse::CrsMatrix<double, int, Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>, void, unsigned long>::operator=<double,int,Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>,void,unsigned long>, "C++: KokkosSparse::CrsMatrix<double, int, Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>, void, unsigned long>::operator=(const class KokkosSparse::CrsMatrix<double, int, struct Kokkos::Device<class Kokkos::Serial, class Kokkos::HostSpace>, void, unsigned long> &) --> class KokkosSparse::CrsMatrix<double, int, struct Kokkos::Device<class Kokkos::Serial, class Kokkos::HostSpace>, void, unsigned long> &", pybind11::return_value_policy::automatic, pybind11::arg("mtx"));
		cl.def("numRows", (int (KokkosSparse::CrsMatrix<double,int,Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>,void,unsigned long>::*)() const) &KokkosSparse::CrsMatrix<double, int, Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>, void, unsigned long>::numRows, "C++: KokkosSparse::CrsMatrix<double, int, Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>, void, unsigned long>::numRows() const --> int");
		cl.def("numCols", (int (KokkosSparse::CrsMatrix<double,int,Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>,void,unsigned long>::*)() const) &KokkosSparse::CrsMatrix<double, int, Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>, void, unsigned long>::numCols, "C++: KokkosSparse::CrsMatrix<double, int, Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>, void, unsigned long>::numCols() const --> int");
		cl.def("numPointRows", (int (KokkosSparse::CrsMatrix<double,int,Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>,void,unsigned long>::*)() const) &KokkosSparse::CrsMatrix<double, int, Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>, void, unsigned long>::numPointRows, "C++: KokkosSparse::CrsMatrix<double, int, Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>, void, unsigned long>::numPointRows() const --> int");
		cl.def("numPointCols", (int (KokkosSparse::CrsMatrix<double,int,Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>,void,unsigned long>::*)() const) &KokkosSparse::CrsMatrix<double, int, Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>, void, unsigned long>::numPointCols, "C++: KokkosSparse::CrsMatrix<double, int, Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>, void, unsigned long>::numPointCols() const --> int");
		cl.def("nnz", (unsigned long (KokkosSparse::CrsMatrix<double,int,Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>,void,unsigned long>::*)() const) &KokkosSparse::CrsMatrix<double, int, Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>, void, unsigned long>::nnz, "C++: KokkosSparse::CrsMatrix<double, int, Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>, void, unsigned long>::nnz() const --> unsigned long");
		cl.def("row", (struct KokkosSparse::SparseRowView<class KokkosSparse::CrsMatrix<double, int, struct Kokkos::Device<class Kokkos::Serial, class Kokkos::HostSpace>, void, unsigned long> > (KokkosSparse::CrsMatrix<double,int,Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>,void,unsigned long>::*)(const int) const) &KokkosSparse::CrsMatrix<double, int, Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>, void, unsigned long>::row, "C++: KokkosSparse::CrsMatrix<double, int, Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>, void, unsigned long>::row(const int) const --> struct KokkosSparse::SparseRowView<class KokkosSparse::CrsMatrix<double, int, struct Kokkos::Device<class Kokkos::Serial, class Kokkos::HostSpace>, void, unsigned long> >", pybind11::arg("i"));
		cl.def("rowConst", (struct KokkosSparse::SparseRowViewConst<class KokkosSparse::CrsMatrix<double, int, struct Kokkos::Device<class Kokkos::Serial, class Kokkos::HostSpace>, void, unsigned long> > (KokkosSparse::CrsMatrix<double,int,Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>,void,unsigned long>::*)(const int) const) &KokkosSparse::CrsMatrix<double, int, Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>, void, unsigned long>::rowConst, "C++: KokkosSparse::CrsMatrix<double, int, Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>, void, unsigned long>::rowConst(const int) const --> struct KokkosSparse::SparseRowViewConst<class KokkosSparse::CrsMatrix<double, int, struct Kokkos::Device<class Kokkos::Serial, class Kokkos::HostSpace>, void, unsigned long> >", pybind11::arg("i"));
		cl.def("assign", (class KokkosSparse::CrsMatrix<double, int, struct Kokkos::Device<class Kokkos::Serial, class Kokkos::HostSpace>, void, unsigned long> & (KokkosSparse::CrsMatrix<double,int,Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>,void,unsigned long>::*)(const class KokkosSparse::CrsMatrix<double, int, struct Kokkos::Device<class Kokkos::Serial, class Kokkos::HostSpace>, void, unsigned long> &)) &KokkosSparse::CrsMatrix<double, int, Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>, void, unsigned long>::operator=, "C++: KokkosSparse::CrsMatrix<double, int, Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>, void, unsigned long>::operator=(const class KokkosSparse::CrsMatrix<double, int, struct Kokkos::Device<class Kokkos::Serial, class Kokkos::HostSpace>, void, unsigned long> &) --> class KokkosSparse::CrsMatrix<double, int, struct Kokkos::Device<class Kokkos::Serial, class Kokkos::HostSpace>, void, unsigned long> &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
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
