#include <PyTrilinos2_Teuchos_Custom.hpp>
#include <Teuchos_ArrayRCPDecl.hpp>
#include <Teuchos_ArrayViewDecl.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_ENull.hpp>
#include <Teuchos_EReductionType.hpp>
#include <Teuchos_PtrDecl.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <Teuchos_RCPNode.hpp>
#include <Teuchos_any.hpp>
#include <iterator>
#include <memory>
#include <sstream> // __str__
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

void bind_Teuchos_CommHelpers(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	// Teuchos::broadcast(const class Teuchos::Comm<int> &, const int, const class Teuchos::Ptr<unsigned long> &) file:Teuchos_CommHelpers.hpp line:1235
	M("Teuchos").def("broadcast", (void (*)(const class Teuchos::Comm<int> &, const int, const class Teuchos::Ptr<unsigned long> &)) &Teuchos::broadcast<int,unsigned long>, "C++: Teuchos::broadcast(const class Teuchos::Comm<int> &, const int, const class Teuchos::Ptr<unsigned long> &) --> void", pybind11::arg("comm"), pybind11::arg("rootRank"), pybind11::arg("object"));

	// Teuchos::broadcast(const class Teuchos::Comm<int> &, const int, const class Teuchos::Ptr<long long> &) file:Teuchos_CommHelpers.hpp line:1235
	M("Teuchos").def("broadcast", (void (*)(const class Teuchos::Comm<int> &, const int, const class Teuchos::Ptr<long long> &)) &Teuchos::broadcast<int,long long>, "C++: Teuchos::broadcast(const class Teuchos::Comm<int> &, const int, const class Teuchos::Ptr<long long> &) --> void", pybind11::arg("comm"), pybind11::arg("rootRank"), pybind11::arg("object"));

	// Teuchos::reduceAll(const class Teuchos::Comm<int> &, const enum Teuchos::EReductionType, const int &, const class Teuchos::Ptr<int> &) file:Teuchos_CommHelpers.hpp line:2111
	M("Teuchos").def("reduceAll", (void (*)(const class Teuchos::Comm<int> &, const enum Teuchos::EReductionType, const int &, const class Teuchos::Ptr<int> &)) &Teuchos::reduceAll<int,int>, "C++: Teuchos::reduceAll(const class Teuchos::Comm<int> &, const enum Teuchos::EReductionType, const int &, const class Teuchos::Ptr<int> &) --> void", pybind11::arg("comm"), pybind11::arg("reductType"), pybind11::arg("send"), pybind11::arg("globalReduct"));

	// Teuchos::reduceAll(const class Teuchos::Comm<int> &, const enum Teuchos::EReductionType, const unsigned long &, const class Teuchos::Ptr<unsigned long> &) file:Teuchos_CommHelpers.hpp line:2111
	M("Teuchos").def("reduceAll", (void (*)(const class Teuchos::Comm<int> &, const enum Teuchos::EReductionType, const unsigned long &, const class Teuchos::Ptr<unsigned long> &)) &Teuchos::reduceAll<int,unsigned long>, "C++: Teuchos::reduceAll(const class Teuchos::Comm<int> &, const enum Teuchos::EReductionType, const unsigned long &, const class Teuchos::Ptr<unsigned long> &) --> void", pybind11::arg("comm"), pybind11::arg("reductType"), pybind11::arg("send"), pybind11::arg("globalReduct"));

	// Teuchos::reduceAll(const class Teuchos::Comm<int> &, const enum Teuchos::EReductionType, const long long &, const class Teuchos::Ptr<long long> &) file:Teuchos_CommHelpers.hpp line:2111
	M("Teuchos").def("reduceAll", (void (*)(const class Teuchos::Comm<int> &, const enum Teuchos::EReductionType, const long long &, const class Teuchos::Ptr<long long> &)) &Teuchos::reduceAll<int,long long>, "C++: Teuchos::reduceAll(const class Teuchos::Comm<int> &, const enum Teuchos::EReductionType, const long long &, const class Teuchos::Ptr<long long> &) --> void", pybind11::arg("comm"), pybind11::arg("reductType"), pybind11::arg("send"), pybind11::arg("globalReduct"));

	// Teuchos::reduceAll(const class Teuchos::Comm<int> &, const enum Teuchos::EReductionType, const double &, const class Teuchos::Ptr<double> &) file:Teuchos_CommHelpers.hpp line:2111
	M("Teuchos").def("reduceAll", (void (*)(const class Teuchos::Comm<int> &, const enum Teuchos::EReductionType, const double &, const class Teuchos::Ptr<double> &)) &Teuchos::reduceAll<int,double>, "C++: Teuchos::reduceAll(const class Teuchos::Comm<int> &, const enum Teuchos::EReductionType, const double &, const class Teuchos::Ptr<double> &) --> void", pybind11::arg("comm"), pybind11::arg("reductType"), pybind11::arg("send"), pybind11::arg("globalReduct"));

	// Teuchos::scan(const class Teuchos::Comm<int> &, const enum Teuchos::EReductionType, const long long &, const class Teuchos::Ptr<long long> &) file:Teuchos_CommHelpers.hpp line:2249
	M("Teuchos").def("scan", (void (*)(const class Teuchos::Comm<int> &, const enum Teuchos::EReductionType, const long long &, const class Teuchos::Ptr<long long> &)) &Teuchos::scan<int,long long>, "C++: Teuchos::scan(const class Teuchos::Comm<int> &, const enum Teuchos::EReductionType, const long long &, const class Teuchos::Ptr<long long> &) --> void", pybind11::arg("comm"), pybind11::arg("reductType"), pybind11::arg("send"), pybind11::arg("scanReduct"));

	// Teuchos::isend(const class Teuchos::ArrayRCP<const char> &, const int, const int, const class Teuchos::Comm<int> &) file:Teuchos_CommHelpers.hpp line:2589
	M("Teuchos").def("isend", (class Teuchos::RCP<class Teuchos::CommRequest<int> > (*)(const class Teuchos::ArrayRCP<const char> &, const int, const int, const class Teuchos::Comm<int> &)) &Teuchos::isend<int,char>, "C++: Teuchos::isend(const class Teuchos::ArrayRCP<const char> &, const int, const int, const class Teuchos::Comm<int> &) --> class Teuchos::RCP<class Teuchos::CommRequest<int> >", pybind11::arg("sendBuffer"), pybind11::arg("destRank"), pybind11::arg("tag"), pybind11::arg("comm"));

	// Teuchos::ireceive(const class Teuchos::ArrayRCP<char> &, const int, const int, const class Teuchos::Comm<int> &) file:Teuchos_CommHelpers.hpp line:2662
	M("Teuchos").def("ireceive", (class Teuchos::RCP<class Teuchos::CommRequest<int> > (*)(const class Teuchos::ArrayRCP<char> &, const int, const int, const class Teuchos::Comm<int> &)) &Teuchos::ireceive<int,char>, "C++: Teuchos::ireceive(const class Teuchos::ArrayRCP<char> &, const int, const int, const class Teuchos::Comm<int> &) --> class Teuchos::RCP<class Teuchos::CommRequest<int> >", pybind11::arg("recvBuffer"), pybind11::arg("sourceRank"), pybind11::arg("tag"), pybind11::arg("comm"));

}
