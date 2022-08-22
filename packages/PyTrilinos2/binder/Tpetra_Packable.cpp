#include <Teuchos_Array.hpp>
#include <Teuchos_ArrayViewDecl.hpp>
#include <Teuchos_ENull.hpp>
#include <Teuchos_RCPNode.hpp>
#include <Tpetra_Packable.hpp>
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

void bind_Tpetra_Packable(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // Tpetra::Packable file:Tpetra_Packable.hpp line:96
		pybind11::class_<Tpetra::Packable<long long,int>, Teuchos::RCP<Tpetra::Packable<long long,int>>, PyCallBack_Tpetra_Packable_long_long_int_t> cl(M("Tpetra"), "Packable_long_long_int_t", "");
		cl.def(pybind11::init<PyCallBack_Tpetra_Packable_long_long_int_t const &>());
		cl.def( pybind11::init( [](){ return new PyCallBack_Tpetra_Packable_long_long_int_t(); } ) );
		cl.def("pack", (void (Tpetra::Packable<long long,int>::*)(const class Teuchos::ArrayView<const int> &, class Teuchos::Array<long long> &, const class Teuchos::ArrayView<unsigned long> &, unsigned long &) const) &Tpetra::Packable<long long, int>::pack, "C++: Tpetra::Packable<long long, int>::pack(const class Teuchos::ArrayView<const int> &, class Teuchos::Array<long long> &, const class Teuchos::ArrayView<unsigned long> &, unsigned long &) const --> void", pybind11::arg("exportLIDs"), pybind11::arg("exports"), pybind11::arg("numPacketsPerLID"), pybind11::arg("constantNumPackets"));
		cl.def("assign", (class Tpetra::Packable<long long, int> & (Tpetra::Packable<long long,int>::*)(const class Tpetra::Packable<long long, int> &)) &Tpetra::Packable<long long, int>::operator=, "C++: Tpetra::Packable<long long, int>::operator=(const class Tpetra::Packable<long long, int> &) --> class Tpetra::Packable<long long, int> &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // Tpetra::Packable file:Tpetra_Packable.hpp line:96
		pybind11::class_<Tpetra::Packable<char,int>, Teuchos::RCP<Tpetra::Packable<char,int>>, PyCallBack_Tpetra_Packable_char_int_t> cl(M("Tpetra"), "Packable_char_int_t", "");
		cl.def(pybind11::init<PyCallBack_Tpetra_Packable_char_int_t const &>());
		cl.def( pybind11::init( [](){ return new PyCallBack_Tpetra_Packable_char_int_t(); } ) );
		cl.def("pack", (void (Tpetra::Packable<char,int>::*)(const class Teuchos::ArrayView<const int> &, class Teuchos::Array<char> &, const class Teuchos::ArrayView<unsigned long> &, unsigned long &) const) &Tpetra::Packable<char, int>::pack, "C++: Tpetra::Packable<char, int>::pack(const class Teuchos::ArrayView<const int> &, class Teuchos::Array<char> &, const class Teuchos::ArrayView<unsigned long> &, unsigned long &) const --> void", pybind11::arg("exportLIDs"), pybind11::arg("exports"), pybind11::arg("numPacketsPerLID"), pybind11::arg("constantNumPackets"));
		cl.def("assign", (class Tpetra::Packable<char, int> & (Tpetra::Packable<char,int>::*)(const class Tpetra::Packable<char, int> &)) &Tpetra::Packable<char, int>::operator=, "C++: Tpetra::Packable<char, int>::operator=(const class Tpetra::Packable<char, int> &) --> class Tpetra::Packable<char, int> &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
}
