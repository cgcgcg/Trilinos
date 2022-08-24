#include <PyTrilinos2_Teuchos_Custom.hpp>
#include <Teuchos_BLAS_types.hpp>
#include <Teuchos_DataAccess.hpp>
#include <Teuchos_Range1D.hpp>
#include <cwchar>
#include <ios>
#include <locale>
#include <ostream>
#include <sstream> // __str__
#include <streambuf>
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

void bind_Teuchos_BLAS_types(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	// Teuchos::ESide file:Teuchos_BLAS_types.hpp line:88
	pybind11::enum_<Teuchos::ESide>(M("Teuchos"), "ESide", pybind11::arithmetic(), "")
		.value("LEFT_SIDE", Teuchos::LEFT_SIDE)
		.value("RIGHT_SIDE", Teuchos::RIGHT_SIDE)
		.export_values();

;

	// Teuchos::ETransp file:Teuchos_BLAS_types.hpp line:93
	pybind11::enum_<Teuchos::ETransp>(M("Teuchos"), "ETransp", pybind11::arithmetic(), "")
		.value("NO_TRANS", Teuchos::NO_TRANS)
		.value("TRANS", Teuchos::TRANS)
		.value("CONJ_TRANS", Teuchos::CONJ_TRANS)
		.export_values();

;

	// Teuchos::EUplo file:Teuchos_BLAS_types.hpp line:99
	pybind11::enum_<Teuchos::EUplo>(M("Teuchos"), "EUplo", pybind11::arithmetic(), "")
		.value("UPPER_TRI", Teuchos::UPPER_TRI)
		.value("LOWER_TRI", Teuchos::LOWER_TRI)
		.value("UNDEF_TRI", Teuchos::UNDEF_TRI)
		.export_values();

;

	// Teuchos::EDiag file:Teuchos_BLAS_types.hpp line:105
	pybind11::enum_<Teuchos::EDiag>(M("Teuchos"), "EDiag", pybind11::arithmetic(), "")
		.value("UNIT_DIAG", Teuchos::UNIT_DIAG)
		.value("NON_UNIT_DIAG", Teuchos::NON_UNIT_DIAG)
		.export_values();

;

	// Teuchos::EType file:Teuchos_BLAS_types.hpp line:110
	pybind11::enum_<Teuchos::EType>(M("Teuchos"), "EType", pybind11::arithmetic(), "")
		.value("FULL", Teuchos::FULL)
		.value("LOWER", Teuchos::LOWER)
		.value("UPPER", Teuchos::UPPER)
		.value("HESSENBERG", Teuchos::HESSENBERG)
		.value("SYM_BAND_L", Teuchos::SYM_BAND_L)
		.value("SYM_BAND_U", Teuchos::SYM_BAND_U)
		.value("BAND", Teuchos::BAND)
		.export_values();

;

	// Teuchos::DataAccess file:Teuchos_DataAccess.hpp line:60
	pybind11::enum_<Teuchos::DataAccess>(M("Teuchos"), "DataAccess", pybind11::arithmetic(), "If set to Copy, user data will be copied at construction.\n      If set to View, user data will be encapsulated and used throughout\n      the life of the object.")
		.value("Copy", Teuchos::Copy)
		.value("View", Teuchos::View)
		.export_values();

;

	{ // Teuchos::Range1D file:Teuchos_Range1D.hpp line:88
		pybind11::class_<Teuchos::Range1D, Teuchos::RCP<Teuchos::Range1D>> cl(M("Teuchos"), "Range1D", "Subregion Index Range Class.\n\n The class %Range1D encapsulates a 1-D, zero-based, range of\n non-negative indexes.  It is used to index into vectors and matrices and\n return subregions of them respectively.\n\n Constructing using Range1D() yields a range that represents the\n entire dimension of an object [0, max_ubound] (an entire\n std::vector, all the rows in a matrix, or all the columns in a matrix\n etc.).\n\n Constructing using \n\n an invalid range [0,-2] with size() == -1.  Once\n constructed with Range1D(INVALID), a %Range1D object can\n pass through many other operations that may change %lbound() and\n %ubound() but will never change size()==-1.\n\n Constructing using \n\n yields a finite-dimensional zero-based range.  The validity of constructed\n range will only be checked if TEUCHOS_DEBUG is defined.\n\n There are many \n\n used with %Range1D objects.\n\n The default copy constructor and assignment operator functions are allowed\n since they have the correct semantics.");
		cl.def( pybind11::init( [](){ return new Teuchos::Range1D(); } ) );
		cl.def( pybind11::init<enum Teuchos::Range1D::EInvalidRange>(), pybind11::arg("") );

		cl.def( pybind11::init<long, long>(), pybind11::arg("lbound"), pybind11::arg("ubound") );

		cl.def( pybind11::init( [](Teuchos::Range1D const &o){ return new Teuchos::Range1D(o); } ) );

		pybind11::enum_<Teuchos::Range1D::EInvalidRange>(cl, "EInvalidRange", pybind11::arithmetic(), ". ")
			.value("INVALID", Teuchos::Range1D::INVALID)
			.export_values();

		cl.def("full_range", (bool (Teuchos::Range1D::*)() const) &Teuchos::Range1D::full_range, "Returns  if the range represents the entire region. \n\nC++: Teuchos::Range1D::full_range() const --> bool");
		cl.def("lbound", (long (Teuchos::Range1D::*)() const) &Teuchos::Range1D::lbound, "Return lower bound of the range \n\nC++: Teuchos::Range1D::lbound() const --> long");
		cl.def("ubound", (long (Teuchos::Range1D::*)() const) &Teuchos::Range1D::ubound, "Return upper bound of the range \n\nC++: Teuchos::Range1D::ubound() const --> long");
		cl.def("size", (long (Teuchos::Range1D::*)() const) &Teuchos::Range1D::size, "Return the size of the range (ubound() - lbound() + 1) \n\nC++: Teuchos::Range1D::size() const --> long");
		cl.def("in_range", (bool (Teuchos::Range1D::*)(long) const) &Teuchos::Range1D::in_range, "Return true if the index is in range \n\nC++: Teuchos::Range1D::in_range(long) const --> bool", pybind11::arg("i"));
		cl.def("__iadd__", (class Teuchos::Range1D & (Teuchos::Range1D::*)(long)) &Teuchos::Range1D::operator+=, "Increment the range by a constant\n\n  this->lbound() + incr >= 0 (throws \n   \n\nC++: Teuchos::Range1D::operator+=(long) --> class Teuchos::Range1D &", pybind11::return_value_policy::automatic, pybind11::arg("incr"));
		cl.def("__isub__", (class Teuchos::Range1D & (Teuchos::Range1D::*)(long)) &Teuchos::Range1D::operator-=, "Deincrement the range by a constant.\n\n  this->lbound() - incr >= 0 (throws \n   \n\nC++: Teuchos::Range1D::operator-=(long) --> class Teuchos::Range1D &", pybind11::return_value_policy::automatic, pybind11::arg("incr"));

		cl.def("__str__", [](Teuchos::Range1D const &o) -> std::string { std::ostringstream s; s << o; return s.str(); } );
	}
	// Teuchos::full_range(const class Teuchos::Range1D &, long, long) file:Teuchos_Range1D.hpp line:272
	M("Teuchos").def("full_range", (class Teuchos::Range1D (*)(const class Teuchos::Range1D &, long, long)) &Teuchos::full_range, "Return a bounded index range from a potentially unbounded index\n range.\n\n Return a index range of lbound to ubound if rng.full_range() == true\n , otherwise just return a copy of rng.\n\n Postconditions: \n	 [rng.full_range() == true] return.lbound() == lbound\n	 [rng.full_range() == true] return.ubound() == ubound\n	 [rng.full_range() == false] return.lbound() == rng.lbound()\n	 [rng.full_range() == false] return.ubound() == rng.ubound()\n	\n\n \n\n  \n\nC++: Teuchos::full_range(const class Teuchos::Range1D &, long, long) --> class Teuchos::Range1D", pybind11::arg("rng"), pybind11::arg("lbound"), pybind11::arg("ubound"));

}
