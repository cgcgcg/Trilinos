#include <Tpetra_Util.hpp> // Tpetra::SortDetails::isAlreadySorted
#include <Tpetra_Util.hpp> // Tpetra::SortDetails::sh_sort2
#include <Tpetra_Util.hpp> // Tpetra::SortDetails::sh_sort3
#include <iterator> // __gnu_cxx::__normal_iterator
#include <memory> // std::allocator
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

void bind_Tpetra_Util(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	// Tpetra::SortDetails::isAlreadySorted(int *const &, int *const &) file:Tpetra_Util.hpp line:256
	M("Tpetra::SortDetails").def("isAlreadySorted", (bool (*)(int *const &, int *const &)) &Tpetra::SortDetails::isAlreadySorted<int *>, "C++: Tpetra::SortDetails::isAlreadySorted(int *const &, int *const &) --> bool", pybind11::arg("first"), pybind11::arg("last"));

	// Tpetra::SortDetails::sh_sort2(int *const &, int *const &, double *const &, double *const &) file:Tpetra_Util.hpp line:470
	M("Tpetra::SortDetails").def("sh_sort2", (void (*)(int *const &, int *const &, double *const &, double *const &)) &Tpetra::SortDetails::sh_sort2<int *,double *>, "C++: Tpetra::SortDetails::sh_sort2(int *const &, int *const &, double *const &, double *const &) --> void", pybind11::arg("first1"), pybind11::arg("last1"), pybind11::arg("first2"), pybind11::arg(""));

}
