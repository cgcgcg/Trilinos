#include <Teuchos_ArrayRCPDecl.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_ArrayViewDecl.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_ENull.hpp>
#include <Teuchos_EReductionType.hpp>
#include <Teuchos_Include_Pybind11.hpp>
#include <Teuchos_PtrDecl.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <Teuchos_RCPNode.hpp>
#include <Teuchos_ReductionOp.hpp>
#include <Teuchos_any.hpp>
#include <memory>
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

void bind_Teuchos_CommHelpers_1(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	// Teuchos::createOp(const enum Teuchos::EReductionType) file:Teuchos_CommHelpers.hpp line:1117
	M("Teuchos").def("createOp", (class Teuchos::ValueTypeReductionOp<int, long long> * (*)(const enum Teuchos::EReductionType)) &Teuchos::createOp<int,long long>, "C++: Teuchos::createOp(const enum Teuchos::EReductionType) --> class Teuchos::ValueTypeReductionOp<int, long long> *", pybind11::return_value_policy::automatic, pybind11::arg("reductType"));

	// Teuchos::ireceive(const class Teuchos::Comm<int> &, const class Teuchos::ArrayRCP<double> &, const int) file:Teuchos_CommHelpers.hpp line:1580
	M("Teuchos").def("ireceive", (class Teuchos::RCP<class Teuchos::CommRequest<int> > (*)(const class Teuchos::Comm<int> &, const class Teuchos::ArrayRCP<double> &, const int)) &Teuchos::ireceive<int,double>, "C++: Teuchos::ireceive(const class Teuchos::Comm<int> &, const class Teuchos::ArrayRCP<double> &, const int) --> class Teuchos::RCP<class Teuchos::CommRequest<int> >", pybind11::arg("comm"), pybind11::arg("recvBuffer"), pybind11::arg("sourceRank"));

	// Teuchos::ireceive(const class Teuchos::ArrayRCP<double> &, const int, const int, const class Teuchos::Comm<int> &) file:Teuchos_CommHelpers.hpp line:1585
	M("Teuchos").def("ireceive", (class Teuchos::RCP<class Teuchos::CommRequest<int> > (*)(const class Teuchos::ArrayRCP<double> &, const int, const int, const class Teuchos::Comm<int> &)) &Teuchos::ireceive<int,double>, "C++: Teuchos::ireceive(const class Teuchos::ArrayRCP<double> &, const int, const int, const class Teuchos::Comm<int> &) --> class Teuchos::RCP<class Teuchos::CommRequest<int> >", pybind11::arg("recvBuffer"), pybind11::arg("sourceRank"), pybind11::arg("tag"), pybind11::arg("comm"));

	// Teuchos::isend(const class Teuchos::ArrayRCP<const double> &, const int, const int, const class Teuchos::Comm<int> &) file:Teuchos_CommHelpers.hpp line:1604
	M("Teuchos").def("isend", (class Teuchos::RCP<class Teuchos::CommRequest<int> > (*)(const class Teuchos::ArrayRCP<const double> &, const int, const int, const class Teuchos::Comm<int> &)) &Teuchos::isend<int,double>, "C++: Teuchos::isend(const class Teuchos::ArrayRCP<const double> &, const int, const int, const class Teuchos::Comm<int> &) --> class Teuchos::RCP<class Teuchos::CommRequest<int> >", pybind11::arg("sendBuffer"), pybind11::arg("destRank"), pybind11::arg("tag"), pybind11::arg("comm"));

	// Teuchos::ireceive(const class Teuchos::Comm<int> &, const class Teuchos::ArrayRCP<long long> &, const int) file:Teuchos_CommHelpers.hpp line:1675
	M("Teuchos").def("ireceive", (class Teuchos::RCP<class Teuchos::CommRequest<int> > (*)(const class Teuchos::Comm<int> &, const class Teuchos::ArrayRCP<long long> &, const int)) &Teuchos::ireceive<int,long long>, "C++: Teuchos::ireceive(const class Teuchos::Comm<int> &, const class Teuchos::ArrayRCP<long long> &, const int) --> class Teuchos::RCP<class Teuchos::CommRequest<int> >", pybind11::arg("comm"), pybind11::arg("recvBuffer"), pybind11::arg("sourceRank"));

	// Teuchos::ireceive(const class Teuchos::ArrayRCP<long long> &, const int, const int, const class Teuchos::Comm<int> &) file:Teuchos_CommHelpers.hpp line:1680
	M("Teuchos").def("ireceive", (class Teuchos::RCP<class Teuchos::CommRequest<int> > (*)(const class Teuchos::ArrayRCP<long long> &, const int, const int, const class Teuchos::Comm<int> &)) &Teuchos::ireceive<int,long long>, "C++: Teuchos::ireceive(const class Teuchos::ArrayRCP<long long> &, const int, const int, const class Teuchos::Comm<int> &) --> class Teuchos::RCP<class Teuchos::CommRequest<int> >", pybind11::arg("recvBuffer"), pybind11::arg("sourceRank"), pybind11::arg("tag"), pybind11::arg("comm"));

	// Teuchos::isend(const class Teuchos::ArrayRCP<const long long> &, const int, const int, const class Teuchos::Comm<int> &) file:Teuchos_CommHelpers.hpp line:1699
	M("Teuchos").def("isend", (class Teuchos::RCP<class Teuchos::CommRequest<int> > (*)(const class Teuchos::ArrayRCP<const long long> &, const int, const int, const class Teuchos::Comm<int> &)) &Teuchos::isend<int,long long>, "C++: Teuchos::isend(const class Teuchos::ArrayRCP<const long long> &, const int, const int, const class Teuchos::Comm<int> &) --> class Teuchos::RCP<class Teuchos::CommRequest<int> >", pybind11::arg("sendBuffer"), pybind11::arg("destRank"), pybind11::arg("tag"), pybind11::arg("comm"));

	// Teuchos::ireceive(const class Teuchos::Comm<int> &, const class Teuchos::ArrayRCP<unsigned long> &, const int) file:Teuchos_CommHelpers.hpp line:1843
	M("Teuchos").def("ireceive", (class Teuchos::RCP<class Teuchos::CommRequest<int> > (*)(const class Teuchos::Comm<int> &, const class Teuchos::ArrayRCP<unsigned long> &, const int)) &Teuchos::ireceive<int,unsigned long>, "C++: Teuchos::ireceive(const class Teuchos::Comm<int> &, const class Teuchos::ArrayRCP<unsigned long> &, const int) --> class Teuchos::RCP<class Teuchos::CommRequest<int> >", pybind11::arg("comm"), pybind11::arg("recvBuffer"), pybind11::arg("sourceRank"));

	// Teuchos::ireceive(const class Teuchos::ArrayRCP<unsigned long> &, const int, const int, const class Teuchos::Comm<int> &) file:Teuchos_CommHelpers.hpp line:1848
	M("Teuchos").def("ireceive", (class Teuchos::RCP<class Teuchos::CommRequest<int> > (*)(const class Teuchos::ArrayRCP<unsigned long> &, const int, const int, const class Teuchos::Comm<int> &)) &Teuchos::ireceive<int,unsigned long>, "C++: Teuchos::ireceive(const class Teuchos::ArrayRCP<unsigned long> &, const int, const int, const class Teuchos::Comm<int> &) --> class Teuchos::RCP<class Teuchos::CommRequest<int> >", pybind11::arg("recvBuffer"), pybind11::arg("sourceRank"), pybind11::arg("tag"), pybind11::arg("comm"));

	// Teuchos::isend(const class Teuchos::ArrayRCP<const unsigned long> &, const int, const int, const class Teuchos::Comm<int> &) file:Teuchos_CommHelpers.hpp line:1867
	M("Teuchos").def("isend", (class Teuchos::RCP<class Teuchos::CommRequest<int> > (*)(const class Teuchos::ArrayRCP<const unsigned long> &, const int, const int, const class Teuchos::Comm<int> &)) &Teuchos::isend<int,unsigned long>, "C++: Teuchos::isend(const class Teuchos::ArrayRCP<const unsigned long> &, const int, const int, const class Teuchos::Comm<int> &) --> class Teuchos::RCP<class Teuchos::CommRequest<int> >", pybind11::arg("sendBuffer"), pybind11::arg("destRank"), pybind11::arg("tag"), pybind11::arg("comm"));

	// Teuchos::ireceive(const class Teuchos::Comm<int> &, const class Teuchos::ArrayRCP<int> &, const int) file:Teuchos_CommHelpers.hpp line:1948
	M("Teuchos").def("ireceive", (class Teuchos::RCP<class Teuchos::CommRequest<int> > (*)(const class Teuchos::Comm<int> &, const class Teuchos::ArrayRCP<int> &, const int)) &Teuchos::ireceive<int,int>, "C++: Teuchos::ireceive(const class Teuchos::Comm<int> &, const class Teuchos::ArrayRCP<int> &, const int) --> class Teuchos::RCP<class Teuchos::CommRequest<int> >", pybind11::arg("comm"), pybind11::arg("recvBuffer"), pybind11::arg("sourceRank"));

	// Teuchos::ireceive(const class Teuchos::ArrayRCP<int> &, const int, const int, const class Teuchos::Comm<int> &) file:Teuchos_CommHelpers.hpp line:1953
	M("Teuchos").def("ireceive", (class Teuchos::RCP<class Teuchos::CommRequest<int> > (*)(const class Teuchos::ArrayRCP<int> &, const int, const int, const class Teuchos::Comm<int> &)) &Teuchos::ireceive<int,int>, "C++: Teuchos::ireceive(const class Teuchos::ArrayRCP<int> &, const int, const int, const class Teuchos::Comm<int> &) --> class Teuchos::RCP<class Teuchos::CommRequest<int> >", pybind11::arg("recvBuffer"), pybind11::arg("sourceRank"), pybind11::arg("tag"), pybind11::arg("comm"));

	// Teuchos::isend(const class Teuchos::ArrayRCP<const int> &, const int, const int, const class Teuchos::Comm<int> &) file:Teuchos_CommHelpers.hpp line:1972
	M("Teuchos").def("isend", (class Teuchos::RCP<class Teuchos::CommRequest<int> > (*)(const class Teuchos::ArrayRCP<const int> &, const int, const int, const class Teuchos::Comm<int> &)) &Teuchos::isend<int,int>, "C++: Teuchos::isend(const class Teuchos::ArrayRCP<const int> &, const int, const int, const class Teuchos::Comm<int> &) --> class Teuchos::RCP<class Teuchos::CommRequest<int> >", pybind11::arg("sendBuffer"), pybind11::arg("destRank"), pybind11::arg("tag"), pybind11::arg("comm"));

}
