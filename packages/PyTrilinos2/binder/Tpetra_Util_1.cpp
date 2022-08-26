#include <Tpetra_Util.hpp> // Tpetra::merge2
#include <Tpetra_Util.hpp> // Tpetra::sort2
#include <Tpetra_Util.hpp> // Tpetra::sort3
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

void bind_Tpetra_Util_1(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	// Tpetra::sort2(int *const &, int *const &, double *const &) file:Tpetra_Util.hpp line:519
	M("Tpetra").def("sort2", (void (*)(int *const &, int *const &, double *const &)) &Tpetra::sort2<int *,double *>, "C++: Tpetra::sort2(int *const &, int *const &, double *const &) --> void", pybind11::arg("first1"), pybind11::arg("last1"), pybind11::arg("first2"));

}
