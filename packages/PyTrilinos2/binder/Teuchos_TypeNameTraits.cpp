#include <KokkosCompat_ClassicNodeAPI_Wrapper.hpp>
#include <KokkosCompat_View.hpp>
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
#include <Teuchos_OpaqueWrapper.hpp>
#include <Teuchos_ParameterEntry.hpp>
#include <Teuchos_ParameterEntryValidator.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_ParameterListModifier.hpp>
#include <Teuchos_PtrDecl.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <Teuchos_RCPNode.hpp>
#include <Teuchos_Range1D.hpp>
#include <Teuchos_ReductionOp.hpp>
#include <Teuchos_SerializationTraits.hpp>
#include <Teuchos_StringIndexedOrderedValueObjectContainer.hpp>
#include <Teuchos_TestForException.hpp>
#include <Teuchos_Time.hpp>
#include <Teuchos_TimeMonitor.hpp>
#include <Teuchos_TypeNameTraits.hpp>
#include <Teuchos_VerbosityLevel.hpp>
#include <Teuchos_any.hpp>
#include <Teuchos_iostream_helpers.hpp>
#include <Tpetra_Access.hpp>
#include <Tpetra_CombineMode.hpp>
#include <Tpetra_ConfigDefs.hpp>
#include <Tpetra_CrsGraph_decl.hpp>
#include <Tpetra_CrsMatrix_decl.hpp>
#include <Tpetra_Details_CrsPadding.hpp>
#include <Tpetra_Details_DistributorPlan.hpp>
#include <Tpetra_Details_LocalMap.hpp>
#include <Tpetra_Details_Transfer_decl.hpp>
#include <Tpetra_Details_WrappedDualView.hpp>
#include <Tpetra_Directory_decl.hpp>
#include <Tpetra_Distributor.hpp>
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
#include <istream>
#include <iterator>
#include <locale>
#include <memory>
#include <mpi.h>
#include <ostream>
#include <sstream>
#include <sstream> // __str__
#include <stdexcept>
#include <streambuf>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>
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

// Teuchos::any::holder file:Teuchos_any.hpp line:268
struct PyCallBack_Teuchos_any_holder_int_t : public Teuchos::any::holder<int> {
	using Teuchos::any::holder<int>::holder;

	const class std::type_info & type() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::any::holder<int> *>(this), "type");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<const class std::type_info &>::value) {
				static pybind11::detail::override_caster_t<const class std::type_info &> caster;
				return pybind11::detail::cast_ref<const class std::type_info &>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<const class std::type_info &>(std::move(o));
		}
		return holder::type();
	}
	std::string typeName() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::any::holder<int> *>(this), "typeName");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<std::string>::value) {
				static pybind11::detail::override_caster_t<std::string> caster;
				return pybind11::detail::cast_ref<std::string>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<std::string>(std::move(o));
		}
		return holder::typeName();
	}
	class Teuchos::any::placeholder * clone() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::any::holder<int> *>(this), "clone");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<class Teuchos::any::placeholder *>::value) {
				static pybind11::detail::override_caster_t<class Teuchos::any::placeholder *> caster;
				return pybind11::detail::cast_ref<class Teuchos::any::placeholder *>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<class Teuchos::any::placeholder *>(std::move(o));
		}
		return holder::clone();
	}
	bool same(const class Teuchos::any::placeholder & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::any::holder<int> *>(this), "same");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return holder::same(a0);
	}
};

// Teuchos::any::holder file:Teuchos_any.hpp line:268
struct PyCallBack_Teuchos_any_holder_double_t : public Teuchos::any::holder<double> {
	using Teuchos::any::holder<double>::holder;

	const class std::type_info & type() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::any::holder<double> *>(this), "type");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<const class std::type_info &>::value) {
				static pybind11::detail::override_caster_t<const class std::type_info &> caster;
				return pybind11::detail::cast_ref<const class std::type_info &>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<const class std::type_info &>(std::move(o));
		}
		return holder::type();
	}
	std::string typeName() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::any::holder<double> *>(this), "typeName");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<std::string>::value) {
				static pybind11::detail::override_caster_t<std::string> caster;
				return pybind11::detail::cast_ref<std::string>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<std::string>(std::move(o));
		}
		return holder::typeName();
	}
	class Teuchos::any::placeholder * clone() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::any::holder<double> *>(this), "clone");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<class Teuchos::any::placeholder *>::value) {
				static pybind11::detail::override_caster_t<class Teuchos::any::placeholder *> caster;
				return pybind11::detail::cast_ref<class Teuchos::any::placeholder *>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<class Teuchos::any::placeholder *>(std::move(o));
		}
		return holder::clone();
	}
	bool same(const class Teuchos::any::placeholder & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::any::holder<double> *>(this), "same");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return holder::same(a0);
	}
};

// Teuchos::any::holder file:Teuchos_any.hpp line:268
struct PyCallBack_Teuchos_any_holder_std_string_t : public Teuchos::any::holder<std::string> {
	using Teuchos::any::holder<std::string>::holder;

	const class std::type_info & type() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::any::holder<std::string> *>(this), "type");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<const class std::type_info &>::value) {
				static pybind11::detail::override_caster_t<const class std::type_info &> caster;
				return pybind11::detail::cast_ref<const class std::type_info &>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<const class std::type_info &>(std::move(o));
		}
		return holder::type();
	}
	std::string typeName() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::any::holder<std::string> *>(this), "typeName");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<std::string>::value) {
				static pybind11::detail::override_caster_t<std::string> caster;
				return pybind11::detail::cast_ref<std::string>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<std::string>(std::move(o));
		}
		return holder::typeName();
	}
	class Teuchos::any::placeholder * clone() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::any::holder<std::string> *>(this), "clone");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<class Teuchos::any::placeholder *>::value) {
				static pybind11::detail::override_caster_t<class Teuchos::any::placeholder *> caster;
				return pybind11::detail::cast_ref<class Teuchos::any::placeholder *>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<class Teuchos::any::placeholder *>(std::move(o));
		}
		return holder::clone();
	}
	bool same(const class Teuchos::any::placeholder & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::any::holder<std::string> *>(this), "same");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return holder::same(a0);
	}
};

// Teuchos::any::holder file:Teuchos_any.hpp line:268
struct PyCallBack_Teuchos_any_holder_Teuchos_ParameterList_t : public Teuchos::any::holder<Teuchos::ParameterList> {
	using Teuchos::any::holder<Teuchos::ParameterList>::holder;

	const class std::type_info & type() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::any::holder<Teuchos::ParameterList> *>(this), "type");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<const class std::type_info &>::value) {
				static pybind11::detail::override_caster_t<const class std::type_info &> caster;
				return pybind11::detail::cast_ref<const class std::type_info &>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<const class std::type_info &>(std::move(o));
		}
		return holder::type();
	}
	std::string typeName() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::any::holder<Teuchos::ParameterList> *>(this), "typeName");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<std::string>::value) {
				static pybind11::detail::override_caster_t<std::string> caster;
				return pybind11::detail::cast_ref<std::string>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<std::string>(std::move(o));
		}
		return holder::typeName();
	}
	class Teuchos::any::placeholder * clone() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::any::holder<Teuchos::ParameterList> *>(this), "clone");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<class Teuchos::any::placeholder *>::value) {
				static pybind11::detail::override_caster_t<class Teuchos::any::placeholder *> caster;
				return pybind11::detail::cast_ref<class Teuchos::any::placeholder *>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<class Teuchos::any::placeholder *>(std::move(o));
		}
		return holder::clone();
	}
	bool same(const class Teuchos::any::placeholder & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::any::holder<Teuchos::ParameterList> *>(this), "same");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return holder::same(a0);
	}
};

// Teuchos::any::holder file:Teuchos_any.hpp line:268
struct PyCallBack_Teuchos_any_holder_bool_t : public Teuchos::any::holder<bool> {
	using Teuchos::any::holder<bool>::holder;

	const class std::type_info & type() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::any::holder<bool> *>(this), "type");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<const class std::type_info &>::value) {
				static pybind11::detail::override_caster_t<const class std::type_info &> caster;
				return pybind11::detail::cast_ref<const class std::type_info &>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<const class std::type_info &>(std::move(o));
		}
		return holder::type();
	}
	std::string typeName() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::any::holder<bool> *>(this), "typeName");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<std::string>::value) {
				static pybind11::detail::override_caster_t<std::string> caster;
				return pybind11::detail::cast_ref<std::string>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<std::string>(std::move(o));
		}
		return holder::typeName();
	}
	class Teuchos::any::placeholder * clone() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::any::holder<bool> *>(this), "clone");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<class Teuchos::any::placeholder *>::value) {
				static pybind11::detail::override_caster_t<class Teuchos::any::placeholder *> caster;
				return pybind11::detail::cast_ref<class Teuchos::any::placeholder *>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<class Teuchos::any::placeholder *>(std::move(o));
		}
		return holder::clone();
	}
	bool same(const class Teuchos::any::placeholder & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::any::holder<bool> *>(this), "same");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return holder::same(a0);
	}
};

// Teuchos::any::holder file:Teuchos_any.hpp line:268
struct PyCallBack_Teuchos_any_holder_Teuchos_ArrayRCP_char_t : public Teuchos::any::holder<Teuchos::ArrayRCP<char>> {
	using Teuchos::any::holder<Teuchos::ArrayRCP<char>>::holder;

	const class std::type_info & type() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::any::holder<Teuchos::ArrayRCP<char>> *>(this), "type");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<const class std::type_info &>::value) {
				static pybind11::detail::override_caster_t<const class std::type_info &> caster;
				return pybind11::detail::cast_ref<const class std::type_info &>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<const class std::type_info &>(std::move(o));
		}
		return holder::type();
	}
	std::string typeName() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::any::holder<Teuchos::ArrayRCP<char>> *>(this), "typeName");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<std::string>::value) {
				static pybind11::detail::override_caster_t<std::string> caster;
				return pybind11::detail::cast_ref<std::string>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<std::string>(std::move(o));
		}
		return holder::typeName();
	}
	class Teuchos::any::placeholder * clone() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::any::holder<Teuchos::ArrayRCP<char>> *>(this), "clone");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<class Teuchos::any::placeholder *>::value) {
				static pybind11::detail::override_caster_t<class Teuchos::any::placeholder *> caster;
				return pybind11::detail::cast_ref<class Teuchos::any::placeholder *>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<class Teuchos::any::placeholder *>(std::move(o));
		}
		return holder::clone();
	}
	bool same(const class Teuchos::any::placeholder & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::any::holder<Teuchos::ArrayRCP<char>> *>(this), "same");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return holder::same(a0);
	}
};

// Teuchos::any::holder file:Teuchos_any.hpp line:268
struct PyCallBack_Teuchos_any_holder_Teuchos_ArrayRCP_const_char_t : public Teuchos::any::holder<Teuchos::ArrayRCP<const char>> {
	using Teuchos::any::holder<Teuchos::ArrayRCP<const char>>::holder;

	const class std::type_info & type() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::any::holder<Teuchos::ArrayRCP<const char>> *>(this), "type");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<const class std::type_info &>::value) {
				static pybind11::detail::override_caster_t<const class std::type_info &> caster;
				return pybind11::detail::cast_ref<const class std::type_info &>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<const class std::type_info &>(std::move(o));
		}
		return holder::type();
	}
	std::string typeName() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::any::holder<Teuchos::ArrayRCP<const char>> *>(this), "typeName");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<std::string>::value) {
				static pybind11::detail::override_caster_t<std::string> caster;
				return pybind11::detail::cast_ref<std::string>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<std::string>(std::move(o));
		}
		return holder::typeName();
	}
	class Teuchos::any::placeholder * clone() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::any::holder<Teuchos::ArrayRCP<const char>> *>(this), "clone");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<class Teuchos::any::placeholder *>::value) {
				static pybind11::detail::override_caster_t<class Teuchos::any::placeholder *> caster;
				return pybind11::detail::cast_ref<class Teuchos::any::placeholder *>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<class Teuchos::any::placeholder *>(std::move(o));
		}
		return holder::clone();
	}
	bool same(const class Teuchos::any::placeholder & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::any::holder<Teuchos::ArrayRCP<const char>> *>(this), "same");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return holder::same(a0);
	}
};

// Teuchos::bad_any_cast file:Teuchos_any.hpp line:324
struct PyCallBack_Teuchos_bad_any_cast : public Teuchos::bad_any_cast {
	using Teuchos::bad_any_cast::bad_any_cast;

	const char * what() const noexcept override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Teuchos::bad_any_cast *>(this), "what");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<const char *>::value) {
				static pybind11::detail::override_caster_t<const char *> caster;
				return pybind11::detail::cast_ref<const char *>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<const char *>(std::move(o));
		}
		return runtime_error::what();
	}
};

void bind_Teuchos_TypeNameTraits(std::function< pybind11::module &(std::string const &namespace_) > &M)
{

	def_Teuchos_functions(M("Teuchos"));
	// Teuchos::demangleName(const std::string &) file:Teuchos_TypeNameTraits.hpp line:77
	M("Teuchos").def("demangleName", (std::string (*)(const std::string &)) &Teuchos::demangleName, "Demangle a C++ name if valid.\n\n The name must have come from typeid(...).name() in order to be\n valid name to pass to this function.\n\n \n\n \n\nC++: Teuchos::demangleName(const std::string &) --> std::string", pybind11::arg("mangledName"));

	// Teuchos::typeName(const class Teuchos::any::placeholder &) file:Teuchos_TypeNameTraits.hpp line:115
	M("Teuchos").def("typeName", (std::string (*)(const class Teuchos::any::placeholder &)) &Teuchos::typeName<Teuchos::any::placeholder>, "C++: Teuchos::typeName(const class Teuchos::any::placeholder &) --> std::string", pybind11::arg("t"));

	// Teuchos::typeName(const class Tpetra::MultiVector<double, int, long long> &) file:Teuchos_TypeNameTraits.hpp line:115
	M("Teuchos").def("typeName", (std::string (*)(const class Tpetra::MultiVector<double, int, long long> &)) &Teuchos::typeName<Tpetra::MultiVector<double, int, long long>>, "C++: Teuchos::typeName(const class Tpetra::MultiVector<double, int, long long> &) --> std::string", pybind11::arg("t"));

	// Teuchos::typeName(const class Tpetra::CrsGraph<int, long long> &) file:Teuchos_TypeNameTraits.hpp line:115
	M("Teuchos").def("typeName", (std::string (*)(const class Tpetra::CrsGraph<int, long long> &)) &Teuchos::typeName<Tpetra::CrsGraph<int, long long>>, "C++: Teuchos::typeName(const class Tpetra::CrsGraph<int, long long> &) --> std::string", pybind11::arg("t"));

	// Teuchos::typeName(const class Tpetra::CrsMatrix<double, int, long long> &) file:Teuchos_TypeNameTraits.hpp line:115
	M("Teuchos").def("typeName", (std::string (*)(const class Tpetra::CrsMatrix<double, int, long long> &)) &Teuchos::typeName<Tpetra::CrsMatrix<double, int, long long>>, "C++: Teuchos::typeName(const class Tpetra::CrsMatrix<double, int, long long> &) --> std::string", pybind11::arg("t"));

	// Teuchos::typeName(const class Teuchos::RCPNodeTmpl<class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> >, class Teuchos::DeallocDelete<class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > > > &) file:Teuchos_TypeNameTraits.hpp line:115
	M("Teuchos").def("typeName", (std::string (*)(const class Teuchos::RCPNodeTmpl<class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> >, class Teuchos::DeallocDelete<class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > > > &)) &Teuchos::typeName<Teuchos::RCPNodeTmpl<Teuchos::basic_FancyOStream<char, std::char_traits<char> >, Teuchos::DeallocDelete<Teuchos::basic_FancyOStream<char, std::char_traits<char> > > >>, "C++: Teuchos::typeName(const class Teuchos::RCPNodeTmpl<class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> >, class Teuchos::DeallocDelete<class Teuchos::basic_FancyOStream<char, struct std::char_traits<char> > > > &) --> std::string", pybind11::arg("t"));

	// Teuchos::typeName(const class Teuchos::RCPNodeTmpl<class Teuchos::ParameterList, class Teuchos::DeallocDelete<class Teuchos::ParameterList> > &) file:Teuchos_TypeNameTraits.hpp line:115
	M("Teuchos").def("typeName", (std::string (*)(const class Teuchos::RCPNodeTmpl<class Teuchos::ParameterList, class Teuchos::DeallocDelete<class Teuchos::ParameterList> > &)) &Teuchos::typeName<Teuchos::RCPNodeTmpl<Teuchos::ParameterList, Teuchos::DeallocDelete<Teuchos::ParameterList> >>, "C++: Teuchos::typeName(const class Teuchos::RCPNodeTmpl<class Teuchos::ParameterList, class Teuchos::DeallocDelete<class Teuchos::ParameterList> > &) --> std::string", pybind11::arg("t"));

	// Teuchos::typeName(const class Teuchos::RCPNodeTmpl<class Teuchos::ParameterEntry, class Teuchos::DeallocNull<class Teuchos::ParameterEntry> > &) file:Teuchos_TypeNameTraits.hpp line:115
	M("Teuchos").def("typeName", (std::string (*)(const class Teuchos::RCPNodeTmpl<class Teuchos::ParameterEntry, class Teuchos::DeallocNull<class Teuchos::ParameterEntry> > &)) &Teuchos::typeName<Teuchos::RCPNodeTmpl<Teuchos::ParameterEntry, Teuchos::DeallocNull<Teuchos::ParameterEntry> >>, "C++: Teuchos::typeName(const class Teuchos::RCPNodeTmpl<class Teuchos::ParameterEntry, class Teuchos::DeallocNull<class Teuchos::ParameterEntry> > &) --> std::string", pybind11::arg("t"));

	// Teuchos::typeName(const class Teuchos::RCPNodeTmpl<const class Teuchos::ParameterEntry, class Teuchos::DeallocNull<const class Teuchos::ParameterEntry> > &) file:Teuchos_TypeNameTraits.hpp line:115
	M("Teuchos").def("typeName", (std::string (*)(const class Teuchos::RCPNodeTmpl<const class Teuchos::ParameterEntry, class Teuchos::DeallocNull<const class Teuchos::ParameterEntry> > &)) &Teuchos::typeName<Teuchos::RCPNodeTmpl<const Teuchos::ParameterEntry, Teuchos::DeallocNull<const Teuchos::ParameterEntry> >>, "C++: Teuchos::typeName(const class Teuchos::RCPNodeTmpl<const class Teuchos::ParameterEntry, class Teuchos::DeallocNull<const class Teuchos::ParameterEntry> > &) --> std::string", pybind11::arg("t"));

	// Teuchos::typeName(const class Teuchos::RCPNodeTmpl<class Teuchos::ParameterList, class Teuchos::EmbeddedObjDealloc<class Teuchos::ParameterList, class Teuchos::RCP<class Teuchos::ParameterList>, class Teuchos::DeallocDelete<class Teuchos::ParameterList> > > &) file:Teuchos_TypeNameTraits.hpp line:115
	M("Teuchos").def("typeName", (std::string (*)(const class Teuchos::RCPNodeTmpl<class Teuchos::ParameterList, class Teuchos::EmbeddedObjDealloc<class Teuchos::ParameterList, class Teuchos::RCP<class Teuchos::ParameterList>, class Teuchos::DeallocDelete<class Teuchos::ParameterList> > > &)) &Teuchos::typeName<Teuchos::RCPNodeTmpl<Teuchos::ParameterList, Teuchos::EmbeddedObjDealloc<Teuchos::ParameterList, Teuchos::RCP<Teuchos::ParameterList>, Teuchos::DeallocDelete<Teuchos::ParameterList> > >>, "C++: Teuchos::typeName(const class Teuchos::RCPNodeTmpl<class Teuchos::ParameterList, class Teuchos::EmbeddedObjDealloc<class Teuchos::ParameterList, class Teuchos::RCP<class Teuchos::ParameterList>, class Teuchos::DeallocDelete<class Teuchos::ParameterList> > > &) --> std::string", pybind11::arg("t"));

	// Teuchos::typeName(const class Teuchos::RCPNodeTmpl<const class Teuchos::ParameterList, class Teuchos::EmbeddedObjDealloc<const class Teuchos::ParameterList, class Teuchos::RCP<const class Teuchos::ParameterList>, class Teuchos::DeallocDelete<const class Teuchos::ParameterList> > > &) file:Teuchos_TypeNameTraits.hpp line:115
	M("Teuchos").def("typeName", (std::string (*)(const class Teuchos::RCPNodeTmpl<const class Teuchos::ParameterList, class Teuchos::EmbeddedObjDealloc<const class Teuchos::ParameterList, class Teuchos::RCP<const class Teuchos::ParameterList>, class Teuchos::DeallocDelete<const class Teuchos::ParameterList> > > &)) &Teuchos::typeName<Teuchos::RCPNodeTmpl<const Teuchos::ParameterList, Teuchos::EmbeddedObjDealloc<const Teuchos::ParameterList, Teuchos::RCP<const Teuchos::ParameterList>, Teuchos::DeallocDelete<const Teuchos::ParameterList> > >>, "C++: Teuchos::typeName(const class Teuchos::RCPNodeTmpl<const class Teuchos::ParameterList, class Teuchos::EmbeddedObjDealloc<const class Teuchos::ParameterList, class Teuchos::RCP<const class Teuchos::ParameterList>, class Teuchos::DeallocDelete<const class Teuchos::ParameterList> > > &) --> std::string", pybind11::arg("t"));

	// Teuchos::typeName(const class Teuchos::RCPNodeTmpl<class Teuchos::Time, class Teuchos::DeallocDelete<class Teuchos::Time> > &) file:Teuchos_TypeNameTraits.hpp line:115
	M("Teuchos").def("typeName", (std::string (*)(const class Teuchos::RCPNodeTmpl<class Teuchos::Time, class Teuchos::DeallocDelete<class Teuchos::Time> > &)) &Teuchos::typeName<Teuchos::RCPNodeTmpl<Teuchos::Time, Teuchos::DeallocDelete<Teuchos::Time> >>, "C++: Teuchos::typeName(const class Teuchos::RCPNodeTmpl<class Teuchos::Time, class Teuchos::DeallocDelete<class Teuchos::Time> > &) --> std::string", pybind11::arg("t"));

	// Teuchos::typeName(const class Teuchos::RCPNodeTmpl<class Teuchos::TimeMonitorSurrogateImpl, class Teuchos::DeallocDelete<class Teuchos::TimeMonitorSurrogateImpl> > &) file:Teuchos_TypeNameTraits.hpp line:115
	M("Teuchos").def("typeName", (std::string (*)(const class Teuchos::RCPNodeTmpl<class Teuchos::TimeMonitorSurrogateImpl, class Teuchos::DeallocDelete<class Teuchos::TimeMonitorSurrogateImpl> > &)) &Teuchos::typeName<Teuchos::RCPNodeTmpl<Teuchos::TimeMonitorSurrogateImpl, Teuchos::DeallocDelete<Teuchos::TimeMonitorSurrogateImpl> >>, "C++: Teuchos::typeName(const class Teuchos::RCPNodeTmpl<class Teuchos::TimeMonitorSurrogateImpl, class Teuchos::DeallocDelete<class Teuchos::TimeMonitorSurrogateImpl> > &) --> std::string", pybind11::arg("t"));

	// Teuchos::typeName(const class Teuchos::RCPNodeTmpl<class Tpetra::Directory<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> >, class Teuchos::DeallocDelete<class Tpetra::Directory<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > > > &) file:Teuchos_TypeNameTraits.hpp line:115
	M("Teuchos").def("typeName", (std::string (*)(const class Teuchos::RCPNodeTmpl<class Tpetra::Directory<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> >, class Teuchos::DeallocDelete<class Tpetra::Directory<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > > > &)) &Teuchos::typeName<Teuchos::RCPNodeTmpl<Tpetra::Directory<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >, Teuchos::DeallocDelete<Tpetra::Directory<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> > > >>, "C++: Teuchos::typeName(const class Teuchos::RCPNodeTmpl<class Tpetra::Directory<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> >, class Teuchos::DeallocDelete<class Tpetra::Directory<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > > > &) --> std::string", pybind11::arg("t"));

	// Teuchos::typeName(const class Teuchos::RCPNodeTmpl<class Teuchos::SerializationTraits<int, unsigned long>, class Teuchos::DeallocDelete<class Teuchos::SerializationTraits<int, unsigned long> > > &) file:Teuchos_TypeNameTraits.hpp line:115
	M("Teuchos").def("typeName", (std::string (*)(const class Teuchos::RCPNodeTmpl<class Teuchos::SerializationTraits<int, unsigned long>, class Teuchos::DeallocDelete<class Teuchos::SerializationTraits<int, unsigned long> > > &)) &Teuchos::typeName<Teuchos::RCPNodeTmpl<Teuchos::SerializationTraits<int, unsigned long>, Teuchos::DeallocDelete<Teuchos::SerializationTraits<int, unsigned long> > >>, "C++: Teuchos::typeName(const class Teuchos::RCPNodeTmpl<class Teuchos::SerializationTraits<int, unsigned long>, class Teuchos::DeallocDelete<class Teuchos::SerializationTraits<int, unsigned long> > > &) --> std::string", pybind11::arg("t"));

	// Teuchos::typeName(const class Teuchos::RCPNodeTmpl<class Teuchos::SerializationTraits<int, long long>, class Teuchos::DeallocDelete<class Teuchos::SerializationTraits<int, long long> > > &) file:Teuchos_TypeNameTraits.hpp line:115
	M("Teuchos").def("typeName", (std::string (*)(const class Teuchos::RCPNodeTmpl<class Teuchos::SerializationTraits<int, long long>, class Teuchos::DeallocDelete<class Teuchos::SerializationTraits<int, long long> > > &)) &Teuchos::typeName<Teuchos::RCPNodeTmpl<Teuchos::SerializationTraits<int, long long>, Teuchos::DeallocDelete<Teuchos::SerializationTraits<int, long long> > >>, "C++: Teuchos::typeName(const class Teuchos::RCPNodeTmpl<class Teuchos::SerializationTraits<int, long long>, class Teuchos::DeallocDelete<class Teuchos::SerializationTraits<int, long long> > > &) --> std::string", pybind11::arg("t"));

	// Teuchos::typeName(const class Teuchos::RCPNodeTmpl<const class Teuchos::ValueTypeReductionOp<int, long long>, class Teuchos::DeallocDelete<const class Teuchos::ValueTypeReductionOp<int, long long> > > &) file:Teuchos_TypeNameTraits.hpp line:115
	M("Teuchos").def("typeName", (std::string (*)(const class Teuchos::RCPNodeTmpl<const class Teuchos::ValueTypeReductionOp<int, long long>, class Teuchos::DeallocDelete<const class Teuchos::ValueTypeReductionOp<int, long long> > > &)) &Teuchos::typeName<Teuchos::RCPNodeTmpl<const Teuchos::ValueTypeReductionOp<int, long long>, Teuchos::DeallocDelete<const Teuchos::ValueTypeReductionOp<int, long long> > >>, "C++: Teuchos::typeName(const class Teuchos::RCPNodeTmpl<const class Teuchos::ValueTypeReductionOp<int, long long>, class Teuchos::DeallocDelete<const class Teuchos::ValueTypeReductionOp<int, long long> > > &) --> std::string", pybind11::arg("t"));

	// Teuchos::typeName(const class Teuchos::RCPNodeTmpl<class Teuchos::SerialCommStatus<int>, class Teuchos::DeallocDelete<class Teuchos::SerialCommStatus<int> > > &) file:Teuchos_TypeNameTraits.hpp line:115
	M("Teuchos").def("typeName", (std::string (*)(const class Teuchos::RCPNodeTmpl<class Teuchos::SerialCommStatus<int>, class Teuchos::DeallocDelete<class Teuchos::SerialCommStatus<int> > > &)) &Teuchos::typeName<Teuchos::RCPNodeTmpl<Teuchos::SerialCommStatus<int>, Teuchos::DeallocDelete<Teuchos::SerialCommStatus<int> > >>, "C++: Teuchos::typeName(const class Teuchos::RCPNodeTmpl<class Teuchos::SerialCommStatus<int>, class Teuchos::DeallocDelete<class Teuchos::SerialCommStatus<int> > > &) --> std::string", pybind11::arg("t"));

	// Teuchos::typeName(const class Teuchos::RCPNodeTmpl<class Teuchos::SerialComm<int>, class Teuchos::DeallocDelete<class Teuchos::SerialComm<int> > > &) file:Teuchos_TypeNameTraits.hpp line:115
	M("Teuchos").def("typeName", (std::string (*)(const class Teuchos::RCPNodeTmpl<class Teuchos::SerialComm<int>, class Teuchos::DeallocDelete<class Teuchos::SerialComm<int> > > &)) &Teuchos::typeName<Teuchos::RCPNodeTmpl<Teuchos::SerialComm<int>, Teuchos::DeallocDelete<Teuchos::SerialComm<int> > >>, "C++: Teuchos::typeName(const class Teuchos::RCPNodeTmpl<class Teuchos::SerialComm<int>, class Teuchos::DeallocDelete<class Teuchos::SerialComm<int> > > &) --> std::string", pybind11::arg("t"));

	// Teuchos::typeName(const class Teuchos::RCPNodeTmpl<const class Teuchos::Comm<int>, class Teuchos::DeallocDelete<const class Teuchos::Comm<int> > > &) file:Teuchos_TypeNameTraits.hpp line:115
	M("Teuchos").def("typeName", (std::string (*)(const class Teuchos::RCPNodeTmpl<const class Teuchos::Comm<int>, class Teuchos::DeallocDelete<const class Teuchos::Comm<int> > > &)) &Teuchos::typeName<Teuchos::RCPNodeTmpl<const Teuchos::Comm<int>, Teuchos::DeallocDelete<const Teuchos::Comm<int> > >>, "C++: Teuchos::typeName(const class Teuchos::RCPNodeTmpl<const class Teuchos::Comm<int>, class Teuchos::DeallocDelete<const class Teuchos::Comm<int> > > &) --> std::string", pybind11::arg("t"));

	// Teuchos::typeName(const class Teuchos::RCPNodeTmpl<class Teuchos::basic_OSTab<char, struct std::char_traits<char> >, class Teuchos::DeallocDelete<class Teuchos::basic_OSTab<char, struct std::char_traits<char> > > > &) file:Teuchos_TypeNameTraits.hpp line:115
	M("Teuchos").def("typeName", (std::string (*)(const class Teuchos::RCPNodeTmpl<class Teuchos::basic_OSTab<char, struct std::char_traits<char> >, class Teuchos::DeallocDelete<class Teuchos::basic_OSTab<char, struct std::char_traits<char> > > > &)) &Teuchos::typeName<Teuchos::RCPNodeTmpl<Teuchos::basic_OSTab<char, std::char_traits<char> >, Teuchos::DeallocDelete<Teuchos::basic_OSTab<char, std::char_traits<char> > > >>, "C++: Teuchos::typeName(const class Teuchos::RCPNodeTmpl<class Teuchos::basic_OSTab<char, struct std::char_traits<char> >, class Teuchos::DeallocDelete<class Teuchos::basic_OSTab<char, struct std::char_traits<char> > > > &) --> std::string", pybind11::arg("t"));

	// Teuchos::typeName(const class Teuchos::RCPNodeTmpl<class Tpetra::Map<int, long long>, class Teuchos::DeallocDelete<class Tpetra::Map<int, long long> > > &) file:Teuchos_TypeNameTraits.hpp line:115
	M("Teuchos").def("typeName", (std::string (*)(const class Teuchos::RCPNodeTmpl<class Tpetra::Map<int, long long>, class Teuchos::DeallocDelete<class Tpetra::Map<int, long long> > > &)) &Teuchos::typeName<Teuchos::RCPNodeTmpl<Tpetra::Map<int, long long>, Teuchos::DeallocDelete<Tpetra::Map<int, long long> > >>, "C++: Teuchos::typeName(const class Teuchos::RCPNodeTmpl<class Tpetra::Map<int, long long>, class Teuchos::DeallocDelete<class Tpetra::Map<int, long long> > > &) --> std::string", pybind11::arg("t"));

	// Teuchos::typeName(const class Teuchos::RCPNodeTmpl<class std::basic_ostringstream<char>, class Teuchos::DeallocDelete<class std::basic_ostringstream<char> > > &) file:Teuchos_TypeNameTraits.hpp line:115
	M("Teuchos").def("typeName", (std::string (*)(const class Teuchos::RCPNodeTmpl<class std::basic_ostringstream<char>, class Teuchos::DeallocDelete<class std::basic_ostringstream<char> > > &)) &Teuchos::typeName<Teuchos::RCPNodeTmpl<std::basic_ostringstream<char>, Teuchos::DeallocDelete<std::basic_ostringstream<char> > >>, "C++: Teuchos::typeName(const class Teuchos::RCPNodeTmpl<class std::basic_ostringstream<char>, class Teuchos::DeallocDelete<class std::basic_ostringstream<char> > > &) --> std::string", pybind11::arg("t"));

	// Teuchos::typeName(const class Teuchos::RCPNodeTmpl<const class Tpetra::Map<int, long long>, class Teuchos::DeallocDelete<const class Tpetra::Map<int, long long> > > &) file:Teuchos_TypeNameTraits.hpp line:115
	M("Teuchos").def("typeName", (std::string (*)(const class Teuchos::RCPNodeTmpl<const class Tpetra::Map<int, long long>, class Teuchos::DeallocDelete<const class Tpetra::Map<int, long long> > > &)) &Teuchos::typeName<Teuchos::RCPNodeTmpl<const Tpetra::Map<int, long long>, Teuchos::DeallocDelete<const Tpetra::Map<int, long long> > >>, "C++: Teuchos::typeName(const class Teuchos::RCPNodeTmpl<const class Tpetra::Map<int, long long>, class Teuchos::DeallocDelete<const class Tpetra::Map<int, long long> > > &) --> std::string", pybind11::arg("t"));

	// Teuchos::typeName(const class Teuchos::RCPNodeTmpl<class Tpetra::MultiVector<double, int, long long>, class Teuchos::DeallocDelete<class Tpetra::MultiVector<double, int, long long> > > &) file:Teuchos_TypeNameTraits.hpp line:115
	M("Teuchos").def("typeName", (std::string (*)(const class Teuchos::RCPNodeTmpl<class Tpetra::MultiVector<double, int, long long>, class Teuchos::DeallocDelete<class Tpetra::MultiVector<double, int, long long> > > &)) &Teuchos::typeName<Teuchos::RCPNodeTmpl<Tpetra::MultiVector<double, int, long long>, Teuchos::DeallocDelete<Tpetra::MultiVector<double, int, long long> > >>, "C++: Teuchos::typeName(const class Teuchos::RCPNodeTmpl<class Tpetra::MultiVector<double, int, long long>, class Teuchos::DeallocDelete<class Tpetra::MultiVector<double, int, long long> > > &) --> std::string", pybind11::arg("t"));

	// Teuchos::typeName(const class Teuchos::RCPNodeTmpl<class Teuchos::ArrayRCP<const double>, class Teuchos::DeallocArrayDelete<class Teuchos::ArrayRCP<const double> > > &) file:Teuchos_TypeNameTraits.hpp line:115
	M("Teuchos").def("typeName", (std::string (*)(const class Teuchos::RCPNodeTmpl<class Teuchos::ArrayRCP<const double>, class Teuchos::DeallocArrayDelete<class Teuchos::ArrayRCP<const double> > > &)) &Teuchos::typeName<Teuchos::RCPNodeTmpl<Teuchos::ArrayRCP<const double>, Teuchos::DeallocArrayDelete<Teuchos::ArrayRCP<const double> > >>, "C++: Teuchos::typeName(const class Teuchos::RCPNodeTmpl<class Teuchos::ArrayRCP<const double>, class Teuchos::DeallocArrayDelete<class Teuchos::ArrayRCP<const double> > > &) --> std::string", pybind11::arg("t"));

	// Teuchos::typeName(const class Teuchos::RCPNodeTmpl<class Teuchos::ArrayRCP<double>, class Teuchos::DeallocArrayDelete<class Teuchos::ArrayRCP<double> > > &) file:Teuchos_TypeNameTraits.hpp line:115
	M("Teuchos").def("typeName", (std::string (*)(const class Teuchos::RCPNodeTmpl<class Teuchos::ArrayRCP<double>, class Teuchos::DeallocArrayDelete<class Teuchos::ArrayRCP<double> > > &)) &Teuchos::typeName<Teuchos::RCPNodeTmpl<Teuchos::ArrayRCP<double>, Teuchos::DeallocArrayDelete<Teuchos::ArrayRCP<double> > >>, "C++: Teuchos::typeName(const class Teuchos::RCPNodeTmpl<class Teuchos::ArrayRCP<double>, class Teuchos::DeallocArrayDelete<class Teuchos::ArrayRCP<double> > > &) --> std::string", pybind11::arg("t"));

	// Teuchos::typeName(const class Teuchos::RCPNodeTmpl<const class Tpetra::MultiVector<double, int, long long>, class Teuchos::DeallocNull<const class Tpetra::MultiVector<double, int, long long> > > &) file:Teuchos_TypeNameTraits.hpp line:115
	M("Teuchos").def("typeName", (std::string (*)(const class Teuchos::RCPNodeTmpl<const class Tpetra::MultiVector<double, int, long long>, class Teuchos::DeallocNull<const class Tpetra::MultiVector<double, int, long long> > > &)) &Teuchos::typeName<Teuchos::RCPNodeTmpl<const Tpetra::MultiVector<double, int, long long>, Teuchos::DeallocNull<const Tpetra::MultiVector<double, int, long long> > >>, "C++: Teuchos::typeName(const class Teuchos::RCPNodeTmpl<const class Tpetra::MultiVector<double, int, long long>, class Teuchos::DeallocNull<const class Tpetra::MultiVector<double, int, long long> > > &) --> std::string", pybind11::arg("t"));

	// Teuchos::typeName(const class Teuchos::RCPNodeTmpl<class Tpetra::CrsGraph<int, long long>, class Teuchos::DeallocDelete<class Tpetra::CrsGraph<int, long long> > > &) file:Teuchos_TypeNameTraits.hpp line:115
	M("Teuchos").def("typeName", (std::string (*)(const class Teuchos::RCPNodeTmpl<class Tpetra::CrsGraph<int, long long>, class Teuchos::DeallocDelete<class Tpetra::CrsGraph<int, long long> > > &)) &Teuchos::typeName<Teuchos::RCPNodeTmpl<Tpetra::CrsGraph<int, long long>, Teuchos::DeallocDelete<Tpetra::CrsGraph<int, long long> > >>, "C++: Teuchos::typeName(const class Teuchos::RCPNodeTmpl<class Tpetra::CrsGraph<int, long long>, class Teuchos::DeallocDelete<class Tpetra::CrsGraph<int, long long> > > &) --> std::string", pybind11::arg("t"));

	// Teuchos::typeName(const class Teuchos::RCPNodeTmpl<unsigned long, class Teuchos::DeallocArrayDelete<unsigned long> > &) file:Teuchos_TypeNameTraits.hpp line:115
	M("Teuchos").def("typeName", (std::string (*)(const class Teuchos::RCPNodeTmpl<unsigned long, class Teuchos::DeallocArrayDelete<unsigned long> > &)) &Teuchos::typeName<Teuchos::RCPNodeTmpl<unsigned long, Teuchos::DeallocArrayDelete<unsigned long> >>, "C++: Teuchos::typeName(const class Teuchos::RCPNodeTmpl<unsigned long, class Teuchos::DeallocArrayDelete<unsigned long> > &) --> std::string", pybind11::arg("t"));

	// Teuchos::typeName(const class Teuchos::RCPNodeTmpl<class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> >, class Teuchos::DeallocDelete<class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > > > &) file:Teuchos_TypeNameTraits.hpp line:115
	M("Teuchos").def("typeName", (std::string (*)(const class Teuchos::RCPNodeTmpl<class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> >, class Teuchos::DeallocDelete<class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > > > &)) &Teuchos::typeName<Teuchos::RCPNodeTmpl<Tpetra::Import<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >, Teuchos::DeallocDelete<Tpetra::Import<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> > > >>, "C++: Teuchos::typeName(const class Teuchos::RCPNodeTmpl<class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> >, class Teuchos::DeallocDelete<class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > > > &) --> std::string", pybind11::arg("t"));

	// Teuchos::typeName(const class Teuchos::RCPNodeTmpl<class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> >, class Teuchos::DeallocDelete<class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > > > &) file:Teuchos_TypeNameTraits.hpp line:115
	M("Teuchos").def("typeName", (std::string (*)(const class Teuchos::RCPNodeTmpl<class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> >, class Teuchos::DeallocDelete<class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > > > &)) &Teuchos::typeName<Teuchos::RCPNodeTmpl<Tpetra::Export<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >, Teuchos::DeallocDelete<Tpetra::Export<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> > > >>, "C++: Teuchos::typeName(const class Teuchos::RCPNodeTmpl<class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> >, class Teuchos::DeallocDelete<class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > > > &) --> std::string", pybind11::arg("t"));

	// Teuchos::typeName(const class Teuchos::RCPNodeTmpl<const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> >, class Teuchos::DeallocNull<const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > > > &) file:Teuchos_TypeNameTraits.hpp line:115
	M("Teuchos").def("typeName", (std::string (*)(const class Teuchos::RCPNodeTmpl<const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> >, class Teuchos::DeallocNull<const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > > > &)) &Teuchos::typeName<Teuchos::RCPNodeTmpl<const Tpetra::Import<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >, Teuchos::DeallocNull<const Tpetra::Import<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> > > >>, "C++: Teuchos::typeName(const class Teuchos::RCPNodeTmpl<const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> >, class Teuchos::DeallocNull<const class Tpetra::Import<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > > > &) --> std::string", pybind11::arg("t"));

	// Teuchos::typeName(const class Teuchos::RCPNodeTmpl<const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> >, class Teuchos::DeallocNull<const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > > > &) file:Teuchos_TypeNameTraits.hpp line:115
	M("Teuchos").def("typeName", (std::string (*)(const class Teuchos::RCPNodeTmpl<const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> >, class Teuchos::DeallocNull<const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > > > &)) &Teuchos::typeName<Teuchos::RCPNodeTmpl<const Tpetra::Export<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >, Teuchos::DeallocNull<const Tpetra::Export<int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> > > >>, "C++: Teuchos::typeName(const class Teuchos::RCPNodeTmpl<const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> >, class Teuchos::DeallocNull<const class Tpetra::Export<int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > > > &) --> std::string", pybind11::arg("t"));

	// Teuchos::typeName(const class Tpetra::MultiVector<int, int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &) file:Teuchos_TypeNameTraits.hpp line:115
	M("Teuchos").def("typeName", (std::string (*)(const class Tpetra::MultiVector<int, int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &)) &Teuchos::typeName<Tpetra::MultiVector<int, int, long long, Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace> >>, "C++: Teuchos::typeName(const class Tpetra::MultiVector<int, int, long long, class Kokkos::Compat::KokkosDeviceWrapperNode<class Kokkos::Serial, class Kokkos::HostSpace> > &) --> std::string", pybind11::arg("t"));

	// Teuchos::typeName(const class Teuchos::RCPNodeTmpl<long long, class Teuchos::DeallocArrayDelete<long long> > &) file:Teuchos_TypeNameTraits.hpp line:115
	M("Teuchos").def("typeName", (std::string (*)(const class Teuchos::RCPNodeTmpl<long long, class Teuchos::DeallocArrayDelete<long long> > &)) &Teuchos::typeName<Teuchos::RCPNodeTmpl<long long, Teuchos::DeallocArrayDelete<long long> >>, "C++: Teuchos::typeName(const class Teuchos::RCPNodeTmpl<long long, class Teuchos::DeallocArrayDelete<long long> > &) --> std::string", pybind11::arg("t"));

	// Teuchos::typeName(const class Teuchos::RCPNodeTmpl<int, class Teuchos::DeallocArrayDelete<int> > &) file:Teuchos_TypeNameTraits.hpp line:115
	M("Teuchos").def("typeName", (std::string (*)(const class Teuchos::RCPNodeTmpl<int, class Teuchos::DeallocArrayDelete<int> > &)) &Teuchos::typeName<Teuchos::RCPNodeTmpl<int, Teuchos::DeallocArrayDelete<int> >>, "C++: Teuchos::typeName(const class Teuchos::RCPNodeTmpl<int, class Teuchos::DeallocArrayDelete<int> > &) --> std::string", pybind11::arg("t"));

	// Teuchos::typeName(const class Teuchos::RCPNodeTmpl<class Tpetra::CrsMatrix<double, int, long long>, class Teuchos::DeallocDelete<class Tpetra::CrsMatrix<double, int, long long> > > &) file:Teuchos_TypeNameTraits.hpp line:115
	M("Teuchos").def("typeName", (std::string (*)(const class Teuchos::RCPNodeTmpl<class Tpetra::CrsMatrix<double, int, long long>, class Teuchos::DeallocDelete<class Tpetra::CrsMatrix<double, int, long long> > > &)) &Teuchos::typeName<Teuchos::RCPNodeTmpl<Tpetra::CrsMatrix<double, int, long long>, Teuchos::DeallocDelete<Tpetra::CrsMatrix<double, int, long long> > >>, "C++: Teuchos::typeName(const class Teuchos::RCPNodeTmpl<class Tpetra::CrsMatrix<double, int, long long>, class Teuchos::DeallocDelete<class Tpetra::CrsMatrix<double, int, long long> > > &) --> std::string", pybind11::arg("t"));

	// Teuchos::typeName(const class Teuchos::RCPNodeTmpl<class Tpetra::Vector<double, int, long long>, class Teuchos::DeallocDelete<class Tpetra::Vector<double, int, long long> > > &) file:Teuchos_TypeNameTraits.hpp line:115
	M("Teuchos").def("typeName", (std::string (*)(const class Teuchos::RCPNodeTmpl<class Tpetra::Vector<double, int, long long>, class Teuchos::DeallocDelete<class Tpetra::Vector<double, int, long long> > > &)) &Teuchos::typeName<Teuchos::RCPNodeTmpl<Tpetra::Vector<double, int, long long>, Teuchos::DeallocDelete<Tpetra::Vector<double, int, long long> > >>, "C++: Teuchos::typeName(const class Teuchos::RCPNodeTmpl<class Tpetra::Vector<double, int, long long>, class Teuchos::DeallocDelete<class Tpetra::Vector<double, int, long long> > > &) --> std::string", pybind11::arg("t"));

	// Teuchos::typeName(const class Teuchos::RCPNodeTmpl<const class Tpetra::Vector<double, int, long long>, class Teuchos::DeallocNull<const class Tpetra::Vector<double, int, long long> > > &) file:Teuchos_TypeNameTraits.hpp line:115
	M("Teuchos").def("typeName", (std::string (*)(const class Teuchos::RCPNodeTmpl<const class Tpetra::Vector<double, int, long long>, class Teuchos::DeallocNull<const class Tpetra::Vector<double, int, long long> > > &)) &Teuchos::typeName<Teuchos::RCPNodeTmpl<const Tpetra::Vector<double, int, long long>, Teuchos::DeallocNull<const Tpetra::Vector<double, int, long long> > >>, "C++: Teuchos::typeName(const class Teuchos::RCPNodeTmpl<const class Tpetra::Vector<double, int, long long>, class Teuchos::DeallocNull<const class Tpetra::Vector<double, int, long long> > > &) --> std::string", pybind11::arg("t"));

	// Teuchos::typeName(const class Teuchos::RCPNodeTmpl<class Teuchos::SerializationTraits<int, char>, class Teuchos::DeallocDelete<class Teuchos::SerializationTraits<int, char> > > &) file:Teuchos_TypeNameTraits.hpp line:115
	M("Teuchos").def("typeName", (std::string (*)(const class Teuchos::RCPNodeTmpl<class Teuchos::SerializationTraits<int, char>, class Teuchos::DeallocDelete<class Teuchos::SerializationTraits<int, char> > > &)) &Teuchos::typeName<Teuchos::RCPNodeTmpl<Teuchos::SerializationTraits<int, char>, Teuchos::DeallocDelete<Teuchos::SerializationTraits<int, char> > >>, "C++: Teuchos::typeName(const class Teuchos::RCPNodeTmpl<class Teuchos::SerializationTraits<int, char>, class Teuchos::DeallocDelete<class Teuchos::SerializationTraits<int, char> > > &) --> std::string", pybind11::arg("t"));

	// Teuchos::typeName(const class Teuchos::RCP<class Teuchos::CommRequest<int> > &) file:Teuchos_TypeNameTraits.hpp line:115
	M("Teuchos").def("typeName", (std::string (*)(const class Teuchos::RCP<class Teuchos::CommRequest<int> > &)) &Teuchos::typeName<Teuchos::RCP<Teuchos::CommRequest<int> >>, "C++: Teuchos::typeName(const class Teuchos::RCP<class Teuchos::CommRequest<int> > &) --> std::string", pybind11::arg("t"));

	// Teuchos::typeName(const class Teuchos::RCPNodeTmpl<double, class Teuchos::DeallocArrayDelete<double> > &) file:Teuchos_TypeNameTraits.hpp line:115
	M("Teuchos").def("typeName", (std::string (*)(const class Teuchos::RCPNodeTmpl<double, class Teuchos::DeallocArrayDelete<double> > &)) &Teuchos::typeName<Teuchos::RCPNodeTmpl<double, Teuchos::DeallocArrayDelete<double> >>, "C++: Teuchos::typeName(const class Teuchos::RCPNodeTmpl<double, class Teuchos::DeallocArrayDelete<double> > &) --> std::string", pybind11::arg("t"));

	// Teuchos::typeName(const class Teuchos::Comm<int> &) file:Teuchos_TypeNameTraits.hpp line:115
	M("Teuchos").def("typeName", (std::string (*)(const class Teuchos::Comm<int> &)) &Teuchos::typeName<Teuchos::Comm<int>>, "C++: Teuchos::typeName(const class Teuchos::Comm<int> &) --> std::string", pybind11::arg("t"));

	// Teuchos::typeName(const class Teuchos::RCPNodeTmpl<struct std::pair<long long, long long>, class Teuchos::DeallocArrayDelete<struct std::pair<long long, long long> > > &) file:Teuchos_TypeNameTraits.hpp line:115
	M("Teuchos").def("typeName", (std::string (*)(const class Teuchos::RCPNodeTmpl<struct std::pair<long long, long long>, class Teuchos::DeallocArrayDelete<struct std::pair<long long, long long> > > &)) &Teuchos::typeName<Teuchos::RCPNodeTmpl<std::pair<long long, long long>, Teuchos::DeallocArrayDelete<std::pair<long long, long long> > >>, "C++: Teuchos::typeName(const class Teuchos::RCPNodeTmpl<struct std::pair<long long, long long>, class Teuchos::DeallocArrayDelete<struct std::pair<long long, long long> > > &) --> std::string", pybind11::arg("t"));

	// Teuchos::TestForException_incrThrowNumber() file:Teuchos_TestForException.hpp line:61
	M("Teuchos").def("TestForException_incrThrowNumber", (void (*)()) &Teuchos::TestForException_incrThrowNumber, "Increment the throw number.  \n\nC++: Teuchos::TestForException_incrThrowNumber() --> void");

	// Teuchos::TestForException_getThrowNumber() file:Teuchos_TestForException.hpp line:64
	M("Teuchos").def("TestForException_getThrowNumber", (int (*)()) &Teuchos::TestForException_getThrowNumber, "Increment the throw number.  \n\nC++: Teuchos::TestForException_getThrowNumber() --> int");

	// Teuchos::TestForException_break(const std::string &) file:Teuchos_TestForException.hpp line:68
	M("Teuchos").def("TestForException_break", (void (*)(const std::string &)) &Teuchos::TestForException_break, "The only purpose for this function is to set a breakpoint.\n    \n\n\nC++: Teuchos::TestForException_break(const std::string &) --> void", pybind11::arg("msg"));

	// Teuchos::TestForException_setEnableStacktrace(bool) file:Teuchos_TestForException.hpp line:72
	M("Teuchos").def("TestForException_setEnableStacktrace", (void (*)(bool)) &Teuchos::TestForException_setEnableStacktrace, "Set at runtime if stacktracing functionality is enabled when *\n    exceptions are thrown.  \n\n\nC++: Teuchos::TestForException_setEnableStacktrace(bool) --> void", pybind11::arg("enableStrackTrace"));

	// Teuchos::TestForException_getEnableStacktrace() file:Teuchos_TestForException.hpp line:76
	M("Teuchos").def("TestForException_getEnableStacktrace", (bool (*)()) &Teuchos::TestForException_getEnableStacktrace, "Get at runtime if stacktracing functionality is enabled when\n exceptions are thrown. \n\nC++: Teuchos::TestForException_getEnableStacktrace() --> bool");

	// Teuchos::TestForTermination_terminate(const std::string &) file:Teuchos_TestForException.hpp line:79
	M("Teuchos").def("TestForTermination_terminate", (void (*)(const std::string &)) &Teuchos::TestForTermination_terminate, "Prints the message to std::cerr and calls std::terminate. \n\nC++: Teuchos::TestForTermination_terminate(const std::string &) --> void", pybind11::arg("msg"));

	// Teuchos::enumIstreamExtractor(std::istream &, enum Teuchos::EVerbosityLevel &) file:Teuchos_iostream_helpers.hpp line:58
	M("Teuchos").def("enumIstreamExtractor", (std::istream & (*)(std::istream &, enum Teuchos::EVerbosityLevel &)) &Teuchos::enumIstreamExtractor<Teuchos::EVerbosityLevel>, "C++: Teuchos::enumIstreamExtractor(std::istream &, enum Teuchos::EVerbosityLevel &) --> std::istream &", pybind11::return_value_policy::automatic, pybind11::arg("std_is"), pybind11::arg("enum_value"));

	// Teuchos::EVerbosityLevel file:Teuchos_VerbosityLevel.hpp line:62
	pybind11::enum_<Teuchos::EVerbosityLevel>(M("Teuchos"), "EVerbosityLevel", pybind11::arithmetic(), "Verbosity level.\n\n \n\n ")
		.value("VERB_DEFAULT", Teuchos::VERB_DEFAULT)
		.value("VERB_NONE", Teuchos::VERB_NONE)
		.value("VERB_LOW", Teuchos::VERB_LOW)
		.value("VERB_MEDIUM", Teuchos::VERB_MEDIUM)
		.value("VERB_HIGH", Teuchos::VERB_HIGH)
		.value("VERB_EXTREME", Teuchos::VERB_EXTREME)
		.export_values();

;

	// Teuchos::toString(const enum Teuchos::EVerbosityLevel) file:Teuchos_VerbosityLevel.hpp line:80
	M("Teuchos").def("toString", (std::string (*)(const enum Teuchos::EVerbosityLevel)) &Teuchos::toString, "Return a std::string representation of the verbosity level.\n\n \n\n \n\nC++: Teuchos::toString(const enum Teuchos::EVerbosityLevel) --> std::string", pybind11::arg("verbLevel"));

	// Teuchos::includesVerbLevel(const enum Teuchos::EVerbosityLevel, const enum Teuchos::EVerbosityLevel, const bool) file:Teuchos_VerbosityLevel.hpp line:96
	M("Teuchos").def("includesVerbLevel", [](const enum Teuchos::EVerbosityLevel & a0, const enum Teuchos::EVerbosityLevel & a1) -> bool { return Teuchos::includesVerbLevel(a0, a1); }, "", pybind11::arg("verbLevel"), pybind11::arg("requestedVerbLevel"));
	M("Teuchos").def("includesVerbLevel", (bool (*)(const enum Teuchos::EVerbosityLevel, const enum Teuchos::EVerbosityLevel, const bool)) &Teuchos::includesVerbLevel, "Return true if the verbosity level includes the given level.\n\n \n\n           [in] The verbosity level that is in effect.\n \n\n\n           [in] The verbosity level the client is asking if\n           is included in verbLevel.\n \n\n\n           [in] Set to true if the level in\n           requestedVerbLevel is the default verbosity level.  In\n           this case, if verbLevel==VERB_DEFAULT, then this function\n           will return true.  The default value is false.\n\nC++: Teuchos::includesVerbLevel(const enum Teuchos::EVerbosityLevel, const enum Teuchos::EVerbosityLevel, const bool) --> bool", pybind11::arg("verbLevel"), pybind11::arg("requestedVerbLevel"), pybind11::arg("isDefaultLevel"));

	// Teuchos::incrVerbLevel(const enum Teuchos::EVerbosityLevel, const int) file:Teuchos_VerbosityLevel.hpp line:112
	M("Teuchos").def("incrVerbLevel", (enum Teuchos::EVerbosityLevel (*)(const enum Teuchos::EVerbosityLevel, const int)) &Teuchos::incrVerbLevel, "Return an increased or decreased verbosity level.\n\n \n\n           [in] The base verbosity level.\n \n\n\n           [in] The number of levels to increase (>0) or decrease (<0).\n\n See the function implementation for details on what it does!\n\nC++: Teuchos::incrVerbLevel(const enum Teuchos::EVerbosityLevel, const int) --> enum Teuchos::EVerbosityLevel", pybind11::arg("inputVerbLevel"), pybind11::arg("numLevels"));

	{ // Teuchos::any file:Teuchos_any.hpp line:154
		pybind11::class_<Teuchos::any, Teuchos::RCP<Teuchos::any>> cl(M("Teuchos"), "any", "Modified boost::any class, which is a container for a templated\n value.");
		cl.def( pybind11::init( [](){ return new Teuchos::any(); } ) );
		cl.def( pybind11::init( [](Teuchos::any const &o){ return new Teuchos::any(o); } ) );
		cl.def("assign", (class Teuchos::any & (Teuchos::any::*)(const class Teuchos::any &)) &Teuchos::any::operator=<Teuchos::any>, "C++: Teuchos::any::operator=(const class Teuchos::any &) --> class Teuchos::any &", pybind11::return_value_policy::automatic, pybind11::arg("rhs"));
		cl.def("swap", (class Teuchos::any & (Teuchos::any::*)(class Teuchos::any &)) &Teuchos::any::swap, "Method for swapping the contents of two any classes\n\nC++: Teuchos::any::swap(class Teuchos::any &) --> class Teuchos::any &", pybind11::return_value_policy::automatic, pybind11::arg("rhs"));
		cl.def("assign", (class Teuchos::any & (Teuchos::any::*)(const class Teuchos::any &)) &Teuchos::any::operator=, "Copy the value held in rhs\n\nC++: Teuchos::any::operator=(const class Teuchos::any &) --> class Teuchos::any &", pybind11::return_value_policy::automatic, pybind11::arg("rhs"));
		cl.def("empty", (bool (Teuchos::any::*)() const) &Teuchos::any::empty, "Return true if nothing is being stored\n\nC++: Teuchos::any::empty() const --> bool");
		cl.def("type", (const class std::type_info & (Teuchos::any::*)() const) &Teuchos::any::type, "Return the type of value being stored\n\nC++: Teuchos::any::type() const --> const class std::type_info &", pybind11::return_value_policy::automatic);
		cl.def("typeName", (std::string (Teuchos::any::*)() const) &Teuchos::any::typeName, "Return the name of the type\n\nC++: Teuchos::any::typeName() const --> std::string");
		cl.def("same", (bool (Teuchos::any::*)(const class Teuchos::any &) const) &Teuchos::any::same, "Return if two any objects are the same or not.\n  \n\n This function with throw an exception if\n           operator== can't be applied to the held type!\n\nC++: Teuchos::any::same(const class Teuchos::any &) const --> bool", pybind11::arg("other"));
		cl.def("access_content", (class Teuchos::any::placeholder * (Teuchos::any::*)()) &Teuchos::any::access_content, "C++: Teuchos::any::access_content() --> class Teuchos::any::placeholder *", pybind11::return_value_policy::automatic);

		cl.def("__str__", [](Teuchos::any const &o) -> std::string { std::ostringstream s; s << o; return s.str(); } );

		{ // Teuchos::any::placeholder file:Teuchos_any.hpp line:249
			auto & enclosing_class = cl;
			pybind11::class_<Teuchos::any::placeholder, Teuchos::RCP<Teuchos::any::placeholder>> cl(enclosing_class, "placeholder", ". ");
			cl.def("type", (const class std::type_info & (Teuchos::any::placeholder::*)() const) &Teuchos::any::placeholder::type, ". \n\nC++: Teuchos::any::placeholder::type() const --> const class std::type_info &", pybind11::return_value_policy::automatic);
			cl.def("typeName", (std::string (Teuchos::any::placeholder::*)() const) &Teuchos::any::placeholder::typeName, ". \n\nC++: Teuchos::any::placeholder::typeName() const --> std::string");
			cl.def("clone", (class Teuchos::any::placeholder * (Teuchos::any::placeholder::*)() const) &Teuchos::any::placeholder::clone, ". \n\nC++: Teuchos::any::placeholder::clone() const --> class Teuchos::any::placeholder *", pybind11::return_value_policy::automatic);
			cl.def("same", (bool (Teuchos::any::placeholder::*)(const class Teuchos::any::placeholder &) const) &Teuchos::any::placeholder::same, ". \n\nC++: Teuchos::any::placeholder::same(const class Teuchos::any::placeholder &) const --> bool", pybind11::arg("other"));
			cl.def("assign", (class Teuchos::any::placeholder & (Teuchos::any::placeholder::*)(const class Teuchos::any::placeholder &)) &Teuchos::any::placeholder::operator=, "C++: Teuchos::any::placeholder::operator=(const class Teuchos::any::placeholder &) --> class Teuchos::any::placeholder &", pybind11::return_value_policy::automatic, pybind11::arg(""));
		}

		{ // Teuchos::any::holder file:Teuchos_any.hpp line:268
			auto & enclosing_class = cl;
			pybind11::class_<Teuchos::any::holder<int>, Teuchos::RCP<Teuchos::any::holder<int>>, PyCallBack_Teuchos_any_holder_int_t, Teuchos::any::placeholder> cl(enclosing_class, "holder_int_t", "");
			cl.def( pybind11::init<const int &>(), pybind11::arg("value") );

			cl.def( pybind11::init( [](PyCallBack_Teuchos_any_holder_int_t const &o){ return new PyCallBack_Teuchos_any_holder_int_t(o); } ) );
			cl.def( pybind11::init( [](Teuchos::any::holder<int> const &o){ return new Teuchos::any::holder<int>(o); } ) );
			cl.def_readwrite("held", &Teuchos::any::holder<int>::held);
			cl.def("type", (const class std::type_info & (Teuchos::any::holder<int>::*)() const) &Teuchos::any::holder<int>::type, "C++: Teuchos::any::holder<int>::type() const --> const class std::type_info &", pybind11::return_value_policy::automatic);
			cl.def("typeName", (std::string (Teuchos::any::holder<int>::*)() const) &Teuchos::any::holder<int>::typeName, "C++: Teuchos::any::holder<int>::typeName() const --> std::string");
			cl.def("clone", (class Teuchos::any::placeholder * (Teuchos::any::holder<int>::*)() const) &Teuchos::any::holder<int>::clone, "C++: Teuchos::any::holder<int>::clone() const --> class Teuchos::any::placeholder *", pybind11::return_value_policy::automatic);
			cl.def("same", (bool (Teuchos::any::holder<int>::*)(const class Teuchos::any::placeholder &) const) &Teuchos::any::holder<int>::same, "C++: Teuchos::any::holder<int>::same(const class Teuchos::any::placeholder &) const --> bool", pybind11::arg("other"));
			cl.def("assign", (class Teuchos::any::holder<int> & (Teuchos::any::holder<int>::*)(const class Teuchos::any::holder<int> &)) &Teuchos::any::holder<int>::operator=, "C++: Teuchos::any::holder<int>::operator=(const class Teuchos::any::holder<int> &) --> class Teuchos::any::holder<int> &", pybind11::return_value_policy::automatic, pybind11::arg(""));
			cl.def("type", (const class std::type_info & (Teuchos::any::placeholder::*)() const) &Teuchos::any::placeholder::type, ". \n\nC++: Teuchos::any::placeholder::type() const --> const class std::type_info &", pybind11::return_value_policy::automatic);
			cl.def("typeName", (std::string (Teuchos::any::placeholder::*)() const) &Teuchos::any::placeholder::typeName, ". \n\nC++: Teuchos::any::placeholder::typeName() const --> std::string");
			cl.def("clone", (class Teuchos::any::placeholder * (Teuchos::any::placeholder::*)() const) &Teuchos::any::placeholder::clone, ". \n\nC++: Teuchos::any::placeholder::clone() const --> class Teuchos::any::placeholder *", pybind11::return_value_policy::automatic);
			cl.def("same", (bool (Teuchos::any::placeholder::*)(const class Teuchos::any::placeholder &) const) &Teuchos::any::placeholder::same, ". \n\nC++: Teuchos::any::placeholder::same(const class Teuchos::any::placeholder &) const --> bool", pybind11::arg("other"));
			cl.def("assign", (class Teuchos::any::placeholder & (Teuchos::any::placeholder::*)(const class Teuchos::any::placeholder &)) &Teuchos::any::placeholder::operator=, "C++: Teuchos::any::placeholder::operator=(const class Teuchos::any::placeholder &) --> class Teuchos::any::placeholder &", pybind11::return_value_policy::automatic, pybind11::arg(""));
		}

		{ // Teuchos::any::holder file:Teuchos_any.hpp line:268
			auto & enclosing_class = cl;
			pybind11::class_<Teuchos::any::holder<double>, Teuchos::RCP<Teuchos::any::holder<double>>, PyCallBack_Teuchos_any_holder_double_t, Teuchos::any::placeholder> cl(enclosing_class, "holder_double_t", "");
			cl.def( pybind11::init<const double &>(), pybind11::arg("value") );

			cl.def( pybind11::init( [](PyCallBack_Teuchos_any_holder_double_t const &o){ return new PyCallBack_Teuchos_any_holder_double_t(o); } ) );
			cl.def( pybind11::init( [](Teuchos::any::holder<double> const &o){ return new Teuchos::any::holder<double>(o); } ) );
			cl.def_readwrite("held", &Teuchos::any::holder<double>::held);
			cl.def("type", (const class std::type_info & (Teuchos::any::holder<double>::*)() const) &Teuchos::any::holder<double>::type, "C++: Teuchos::any::holder<double>::type() const --> const class std::type_info &", pybind11::return_value_policy::automatic);
			cl.def("typeName", (std::string (Teuchos::any::holder<double>::*)() const) &Teuchos::any::holder<double>::typeName, "C++: Teuchos::any::holder<double>::typeName() const --> std::string");
			cl.def("clone", (class Teuchos::any::placeholder * (Teuchos::any::holder<double>::*)() const) &Teuchos::any::holder<double>::clone, "C++: Teuchos::any::holder<double>::clone() const --> class Teuchos::any::placeholder *", pybind11::return_value_policy::automatic);
			cl.def("same", (bool (Teuchos::any::holder<double>::*)(const class Teuchos::any::placeholder &) const) &Teuchos::any::holder<double>::same, "C++: Teuchos::any::holder<double>::same(const class Teuchos::any::placeholder &) const --> bool", pybind11::arg("other"));
			cl.def("assign", (class Teuchos::any::holder<double> & (Teuchos::any::holder<double>::*)(const class Teuchos::any::holder<double> &)) &Teuchos::any::holder<double>::operator=, "C++: Teuchos::any::holder<double>::operator=(const class Teuchos::any::holder<double> &) --> class Teuchos::any::holder<double> &", pybind11::return_value_policy::automatic, pybind11::arg(""));
			cl.def("type", (const class std::type_info & (Teuchos::any::placeholder::*)() const) &Teuchos::any::placeholder::type, ". \n\nC++: Teuchos::any::placeholder::type() const --> const class std::type_info &", pybind11::return_value_policy::automatic);
			cl.def("typeName", (std::string (Teuchos::any::placeholder::*)() const) &Teuchos::any::placeholder::typeName, ". \n\nC++: Teuchos::any::placeholder::typeName() const --> std::string");
			cl.def("clone", (class Teuchos::any::placeholder * (Teuchos::any::placeholder::*)() const) &Teuchos::any::placeholder::clone, ". \n\nC++: Teuchos::any::placeholder::clone() const --> class Teuchos::any::placeholder *", pybind11::return_value_policy::automatic);
			cl.def("same", (bool (Teuchos::any::placeholder::*)(const class Teuchos::any::placeholder &) const) &Teuchos::any::placeholder::same, ". \n\nC++: Teuchos::any::placeholder::same(const class Teuchos::any::placeholder &) const --> bool", pybind11::arg("other"));
			cl.def("assign", (class Teuchos::any::placeholder & (Teuchos::any::placeholder::*)(const class Teuchos::any::placeholder &)) &Teuchos::any::placeholder::operator=, "C++: Teuchos::any::placeholder::operator=(const class Teuchos::any::placeholder &) --> class Teuchos::any::placeholder &", pybind11::return_value_policy::automatic, pybind11::arg(""));
		}

		{ // Teuchos::any::holder file:Teuchos_any.hpp line:268
			auto & enclosing_class = cl;
			pybind11::class_<Teuchos::any::holder<std::string>, Teuchos::RCP<Teuchos::any::holder<std::string>>, PyCallBack_Teuchos_any_holder_std_string_t, Teuchos::any::placeholder> cl(enclosing_class, "holder_std_string_t", "");
			cl.def( pybind11::init<const std::string &>(), pybind11::arg("value") );

			cl.def( pybind11::init( [](PyCallBack_Teuchos_any_holder_std_string_t const &o){ return new PyCallBack_Teuchos_any_holder_std_string_t(o); } ) );
			cl.def( pybind11::init( [](Teuchos::any::holder<std::string> const &o){ return new Teuchos::any::holder<std::string>(o); } ) );
			cl.def_readwrite("held", &Teuchos::any::holder<std::string>::held);
			cl.def("type", (const class std::type_info & (Teuchos::any::holder<std::string>::*)() const) &Teuchos::any::holder<std::string >::type, "C++: Teuchos::any::holder<std::string >::type() const --> const class std::type_info &", pybind11::return_value_policy::automatic);
			cl.def("typeName", (std::string (Teuchos::any::holder<std::string>::*)() const) &Teuchos::any::holder<std::string >::typeName, "C++: Teuchos::any::holder<std::string >::typeName() const --> std::string");
			cl.def("clone", (class Teuchos::any::placeholder * (Teuchos::any::holder<std::string>::*)() const) &Teuchos::any::holder<std::string >::clone, "C++: Teuchos::any::holder<std::string >::clone() const --> class Teuchos::any::placeholder *", pybind11::return_value_policy::automatic);
			cl.def("same", (bool (Teuchos::any::holder<std::string>::*)(const class Teuchos::any::placeholder &) const) &Teuchos::any::holder<std::string >::same, "C++: Teuchos::any::holder<std::string >::same(const class Teuchos::any::placeholder &) const --> bool", pybind11::arg("other"));
			cl.def("assign", (class Teuchos::any::holder<std::string > & (Teuchos::any::holder<std::string>::*)(const class Teuchos::any::holder<std::string > &)) &Teuchos::any::holder<std::string >::operator=, "C++: Teuchos::any::holder<std::string >::operator=(const class Teuchos::any::holder<std::string > &) --> class Teuchos::any::holder<std::string > &", pybind11::return_value_policy::automatic, pybind11::arg(""));
			cl.def("type", (const class std::type_info & (Teuchos::any::placeholder::*)() const) &Teuchos::any::placeholder::type, ". \n\nC++: Teuchos::any::placeholder::type() const --> const class std::type_info &", pybind11::return_value_policy::automatic);
			cl.def("typeName", (std::string (Teuchos::any::placeholder::*)() const) &Teuchos::any::placeholder::typeName, ". \n\nC++: Teuchos::any::placeholder::typeName() const --> std::string");
			cl.def("clone", (class Teuchos::any::placeholder * (Teuchos::any::placeholder::*)() const) &Teuchos::any::placeholder::clone, ". \n\nC++: Teuchos::any::placeholder::clone() const --> class Teuchos::any::placeholder *", pybind11::return_value_policy::automatic);
			cl.def("same", (bool (Teuchos::any::placeholder::*)(const class Teuchos::any::placeholder &) const) &Teuchos::any::placeholder::same, ". \n\nC++: Teuchos::any::placeholder::same(const class Teuchos::any::placeholder &) const --> bool", pybind11::arg("other"));
			cl.def("assign", (class Teuchos::any::placeholder & (Teuchos::any::placeholder::*)(const class Teuchos::any::placeholder &)) &Teuchos::any::placeholder::operator=, "C++: Teuchos::any::placeholder::operator=(const class Teuchos::any::placeholder &) --> class Teuchos::any::placeholder &", pybind11::return_value_policy::automatic, pybind11::arg(""));
		}

		{ // Teuchos::any::holder file:Teuchos_any.hpp line:268
			auto & enclosing_class = cl;
			pybind11::class_<Teuchos::any::holder<Teuchos::ParameterList>, Teuchos::RCP<Teuchos::any::holder<Teuchos::ParameterList>>, PyCallBack_Teuchos_any_holder_Teuchos_ParameterList_t, Teuchos::any::placeholder> cl(enclosing_class, "holder_Teuchos_ParameterList_t", "");
			cl.def( pybind11::init<const class Teuchos::ParameterList &>(), pybind11::arg("value") );

			cl.def( pybind11::init( [](PyCallBack_Teuchos_any_holder_Teuchos_ParameterList_t const &o){ return new PyCallBack_Teuchos_any_holder_Teuchos_ParameterList_t(o); } ) );
			cl.def( pybind11::init( [](Teuchos::any::holder<Teuchos::ParameterList> const &o){ return new Teuchos::any::holder<Teuchos::ParameterList>(o); } ) );
			cl.def_readwrite("held", &Teuchos::any::holder<Teuchos::ParameterList>::held);
			cl.def("type", (const class std::type_info & (Teuchos::any::holder<Teuchos::ParameterList>::*)() const) &Teuchos::any::holder<Teuchos::ParameterList>::type, "C++: Teuchos::any::holder<Teuchos::ParameterList>::type() const --> const class std::type_info &", pybind11::return_value_policy::automatic);
			cl.def("typeName", (std::string (Teuchos::any::holder<Teuchos::ParameterList>::*)() const) &Teuchos::any::holder<Teuchos::ParameterList>::typeName, "C++: Teuchos::any::holder<Teuchos::ParameterList>::typeName() const --> std::string");
			cl.def("clone", (class Teuchos::any::placeholder * (Teuchos::any::holder<Teuchos::ParameterList>::*)() const) &Teuchos::any::holder<Teuchos::ParameterList>::clone, "C++: Teuchos::any::holder<Teuchos::ParameterList>::clone() const --> class Teuchos::any::placeholder *", pybind11::return_value_policy::automatic);
			cl.def("same", (bool (Teuchos::any::holder<Teuchos::ParameterList>::*)(const class Teuchos::any::placeholder &) const) &Teuchos::any::holder<Teuchos::ParameterList>::same, "C++: Teuchos::any::holder<Teuchos::ParameterList>::same(const class Teuchos::any::placeholder &) const --> bool", pybind11::arg("other"));
			cl.def("assign", (class Teuchos::any::holder<class Teuchos::ParameterList> & (Teuchos::any::holder<Teuchos::ParameterList>::*)(const class Teuchos::any::holder<class Teuchos::ParameterList> &)) &Teuchos::any::holder<Teuchos::ParameterList>::operator=, "C++: Teuchos::any::holder<Teuchos::ParameterList>::operator=(const class Teuchos::any::holder<class Teuchos::ParameterList> &) --> class Teuchos::any::holder<class Teuchos::ParameterList> &", pybind11::return_value_policy::automatic, pybind11::arg(""));
			cl.def("type", (const class std::type_info & (Teuchos::any::placeholder::*)() const) &Teuchos::any::placeholder::type, ". \n\nC++: Teuchos::any::placeholder::type() const --> const class std::type_info &", pybind11::return_value_policy::automatic);
			cl.def("typeName", (std::string (Teuchos::any::placeholder::*)() const) &Teuchos::any::placeholder::typeName, ". \n\nC++: Teuchos::any::placeholder::typeName() const --> std::string");
			cl.def("clone", (class Teuchos::any::placeholder * (Teuchos::any::placeholder::*)() const) &Teuchos::any::placeholder::clone, ". \n\nC++: Teuchos::any::placeholder::clone() const --> class Teuchos::any::placeholder *", pybind11::return_value_policy::automatic);
			cl.def("same", (bool (Teuchos::any::placeholder::*)(const class Teuchos::any::placeholder &) const) &Teuchos::any::placeholder::same, ". \n\nC++: Teuchos::any::placeholder::same(const class Teuchos::any::placeholder &) const --> bool", pybind11::arg("other"));
			cl.def("assign", (class Teuchos::any::placeholder & (Teuchos::any::placeholder::*)(const class Teuchos::any::placeholder &)) &Teuchos::any::placeholder::operator=, "C++: Teuchos::any::placeholder::operator=(const class Teuchos::any::placeholder &) --> class Teuchos::any::placeholder &", pybind11::return_value_policy::automatic, pybind11::arg(""));
		}

		{ // Teuchos::any::holder file:Teuchos_any.hpp line:268
			auto & enclosing_class = cl;
			pybind11::class_<Teuchos::any::holder<bool>, Teuchos::RCP<Teuchos::any::holder<bool>>, PyCallBack_Teuchos_any_holder_bool_t, Teuchos::any::placeholder> cl(enclosing_class, "holder_bool_t", "");
			cl.def( pybind11::init<const bool &>(), pybind11::arg("value") );

			cl.def( pybind11::init( [](PyCallBack_Teuchos_any_holder_bool_t const &o){ return new PyCallBack_Teuchos_any_holder_bool_t(o); } ) );
			cl.def( pybind11::init( [](Teuchos::any::holder<bool> const &o){ return new Teuchos::any::holder<bool>(o); } ) );
			cl.def_readwrite("held", &Teuchos::any::holder<bool>::held);
			cl.def("type", (const class std::type_info & (Teuchos::any::holder<bool>::*)() const) &Teuchos::any::holder<bool>::type, "C++: Teuchos::any::holder<bool>::type() const --> const class std::type_info &", pybind11::return_value_policy::automatic);
			cl.def("typeName", (std::string (Teuchos::any::holder<bool>::*)() const) &Teuchos::any::holder<bool>::typeName, "C++: Teuchos::any::holder<bool>::typeName() const --> std::string");
			cl.def("clone", (class Teuchos::any::placeholder * (Teuchos::any::holder<bool>::*)() const) &Teuchos::any::holder<bool>::clone, "C++: Teuchos::any::holder<bool>::clone() const --> class Teuchos::any::placeholder *", pybind11::return_value_policy::automatic);
			cl.def("same", (bool (Teuchos::any::holder<bool>::*)(const class Teuchos::any::placeholder &) const) &Teuchos::any::holder<bool>::same, "C++: Teuchos::any::holder<bool>::same(const class Teuchos::any::placeholder &) const --> bool", pybind11::arg("other"));
			cl.def("assign", (class Teuchos::any::holder<bool> & (Teuchos::any::holder<bool>::*)(const class Teuchos::any::holder<bool> &)) &Teuchos::any::holder<bool>::operator=, "C++: Teuchos::any::holder<bool>::operator=(const class Teuchos::any::holder<bool> &) --> class Teuchos::any::holder<bool> &", pybind11::return_value_policy::automatic, pybind11::arg(""));
			cl.def("type", (const class std::type_info & (Teuchos::any::placeholder::*)() const) &Teuchos::any::placeholder::type, ". \n\nC++: Teuchos::any::placeholder::type() const --> const class std::type_info &", pybind11::return_value_policy::automatic);
			cl.def("typeName", (std::string (Teuchos::any::placeholder::*)() const) &Teuchos::any::placeholder::typeName, ". \n\nC++: Teuchos::any::placeholder::typeName() const --> std::string");
			cl.def("clone", (class Teuchos::any::placeholder * (Teuchos::any::placeholder::*)() const) &Teuchos::any::placeholder::clone, ". \n\nC++: Teuchos::any::placeholder::clone() const --> class Teuchos::any::placeholder *", pybind11::return_value_policy::automatic);
			cl.def("same", (bool (Teuchos::any::placeholder::*)(const class Teuchos::any::placeholder &) const) &Teuchos::any::placeholder::same, ". \n\nC++: Teuchos::any::placeholder::same(const class Teuchos::any::placeholder &) const --> bool", pybind11::arg("other"));
			cl.def("assign", (class Teuchos::any::placeholder & (Teuchos::any::placeholder::*)(const class Teuchos::any::placeholder &)) &Teuchos::any::placeholder::operator=, "C++: Teuchos::any::placeholder::operator=(const class Teuchos::any::placeholder &) --> class Teuchos::any::placeholder &", pybind11::return_value_policy::automatic, pybind11::arg(""));
		}

		{ // Teuchos::any::holder file:Teuchos_any.hpp line:268
			auto & enclosing_class = cl;
			pybind11::class_<Teuchos::any::holder<Teuchos::ArrayRCP<char>>, Teuchos::RCP<Teuchos::any::holder<Teuchos::ArrayRCP<char>>>, PyCallBack_Teuchos_any_holder_Teuchos_ArrayRCP_char_t, Teuchos::any::placeholder> cl(enclosing_class, "holder_Teuchos_ArrayRCP_char_t", "");
			cl.def( pybind11::init<const class Teuchos::ArrayRCP<char> &>(), pybind11::arg("value") );

			cl.def( pybind11::init( [](PyCallBack_Teuchos_any_holder_Teuchos_ArrayRCP_char_t const &o){ return new PyCallBack_Teuchos_any_holder_Teuchos_ArrayRCP_char_t(o); } ) );
			cl.def( pybind11::init( [](Teuchos::any::holder<Teuchos::ArrayRCP<char>> const &o){ return new Teuchos::any::holder<Teuchos::ArrayRCP<char>>(o); } ) );
			cl.def_readwrite("held", &Teuchos::any::holder<Teuchos::ArrayRCP<char>>::held);
			cl.def("type", (const class std::type_info & (Teuchos::any::holder<Teuchos::ArrayRCP<char>>::*)() const) &Teuchos::any::holder<Teuchos::ArrayRCP<char> >::type, "C++: Teuchos::any::holder<Teuchos::ArrayRCP<char> >::type() const --> const class std::type_info &", pybind11::return_value_policy::automatic);
			cl.def("typeName", (std::string (Teuchos::any::holder<Teuchos::ArrayRCP<char>>::*)() const) &Teuchos::any::holder<Teuchos::ArrayRCP<char> >::typeName, "C++: Teuchos::any::holder<Teuchos::ArrayRCP<char> >::typeName() const --> std::string");
			cl.def("clone", (class Teuchos::any::placeholder * (Teuchos::any::holder<Teuchos::ArrayRCP<char>>::*)() const) &Teuchos::any::holder<Teuchos::ArrayRCP<char> >::clone, "C++: Teuchos::any::holder<Teuchos::ArrayRCP<char> >::clone() const --> class Teuchos::any::placeholder *", pybind11::return_value_policy::automatic);
			cl.def("same", (bool (Teuchos::any::holder<Teuchos::ArrayRCP<char>>::*)(const class Teuchos::any::placeholder &) const) &Teuchos::any::holder<Teuchos::ArrayRCP<char> >::same, "C++: Teuchos::any::holder<Teuchos::ArrayRCP<char> >::same(const class Teuchos::any::placeholder &) const --> bool", pybind11::arg("other"));
			cl.def("assign", (class Teuchos::any::holder<class Teuchos::ArrayRCP<char> > & (Teuchos::any::holder<Teuchos::ArrayRCP<char>>::*)(const class Teuchos::any::holder<class Teuchos::ArrayRCP<char> > &)) &Teuchos::any::holder<Teuchos::ArrayRCP<char> >::operator=, "C++: Teuchos::any::holder<Teuchos::ArrayRCP<char> >::operator=(const class Teuchos::any::holder<class Teuchos::ArrayRCP<char> > &) --> class Teuchos::any::holder<class Teuchos::ArrayRCP<char> > &", pybind11::return_value_policy::automatic, pybind11::arg(""));
			cl.def("type", (const class std::type_info & (Teuchos::any::placeholder::*)() const) &Teuchos::any::placeholder::type, ". \n\nC++: Teuchos::any::placeholder::type() const --> const class std::type_info &", pybind11::return_value_policy::automatic);
			cl.def("typeName", (std::string (Teuchos::any::placeholder::*)() const) &Teuchos::any::placeholder::typeName, ". \n\nC++: Teuchos::any::placeholder::typeName() const --> std::string");
			cl.def("clone", (class Teuchos::any::placeholder * (Teuchos::any::placeholder::*)() const) &Teuchos::any::placeholder::clone, ". \n\nC++: Teuchos::any::placeholder::clone() const --> class Teuchos::any::placeholder *", pybind11::return_value_policy::automatic);
			cl.def("same", (bool (Teuchos::any::placeholder::*)(const class Teuchos::any::placeholder &) const) &Teuchos::any::placeholder::same, ". \n\nC++: Teuchos::any::placeholder::same(const class Teuchos::any::placeholder &) const --> bool", pybind11::arg("other"));
			cl.def("assign", (class Teuchos::any::placeholder & (Teuchos::any::placeholder::*)(const class Teuchos::any::placeholder &)) &Teuchos::any::placeholder::operator=, "C++: Teuchos::any::placeholder::operator=(const class Teuchos::any::placeholder &) --> class Teuchos::any::placeholder &", pybind11::return_value_policy::automatic, pybind11::arg(""));
		}

		{ // Teuchos::any::holder file:Teuchos_any.hpp line:268
			auto & enclosing_class = cl;
			pybind11::class_<Teuchos::any::holder<Teuchos::ArrayRCP<const char>>, Teuchos::RCP<Teuchos::any::holder<Teuchos::ArrayRCP<const char>>>, PyCallBack_Teuchos_any_holder_Teuchos_ArrayRCP_const_char_t, Teuchos::any::placeholder> cl(enclosing_class, "holder_Teuchos_ArrayRCP_const_char_t", "");
			cl.def( pybind11::init<const class Teuchos::ArrayRCP<const char> &>(), pybind11::arg("value") );

			cl.def( pybind11::init( [](PyCallBack_Teuchos_any_holder_Teuchos_ArrayRCP_const_char_t const &o){ return new PyCallBack_Teuchos_any_holder_Teuchos_ArrayRCP_const_char_t(o); } ) );
			cl.def( pybind11::init( [](Teuchos::any::holder<Teuchos::ArrayRCP<const char>> const &o){ return new Teuchos::any::holder<Teuchos::ArrayRCP<const char>>(o); } ) );
			cl.def_readwrite("held", &Teuchos::any::holder<Teuchos::ArrayRCP<const char>>::held);
			cl.def("type", (const class std::type_info & (Teuchos::any::holder<Teuchos::ArrayRCP<const char>>::*)() const) &Teuchos::any::holder<Teuchos::ArrayRCP<const char> >::type, "C++: Teuchos::any::holder<Teuchos::ArrayRCP<const char> >::type() const --> const class std::type_info &", pybind11::return_value_policy::automatic);
			cl.def("typeName", (std::string (Teuchos::any::holder<Teuchos::ArrayRCP<const char>>::*)() const) &Teuchos::any::holder<Teuchos::ArrayRCP<const char> >::typeName, "C++: Teuchos::any::holder<Teuchos::ArrayRCP<const char> >::typeName() const --> std::string");
			cl.def("clone", (class Teuchos::any::placeholder * (Teuchos::any::holder<Teuchos::ArrayRCP<const char>>::*)() const) &Teuchos::any::holder<Teuchos::ArrayRCP<const char> >::clone, "C++: Teuchos::any::holder<Teuchos::ArrayRCP<const char> >::clone() const --> class Teuchos::any::placeholder *", pybind11::return_value_policy::automatic);
			cl.def("same", (bool (Teuchos::any::holder<Teuchos::ArrayRCP<const char>>::*)(const class Teuchos::any::placeholder &) const) &Teuchos::any::holder<Teuchos::ArrayRCP<const char> >::same, "C++: Teuchos::any::holder<Teuchos::ArrayRCP<const char> >::same(const class Teuchos::any::placeholder &) const --> bool", pybind11::arg("other"));
			cl.def("assign", (class Teuchos::any::holder<class Teuchos::ArrayRCP<const char> > & (Teuchos::any::holder<Teuchos::ArrayRCP<const char>>::*)(const class Teuchos::any::holder<class Teuchos::ArrayRCP<const char> > &)) &Teuchos::any::holder<Teuchos::ArrayRCP<const char> >::operator=, "C++: Teuchos::any::holder<Teuchos::ArrayRCP<const char> >::operator=(const class Teuchos::any::holder<class Teuchos::ArrayRCP<const char> > &) --> class Teuchos::any::holder<class Teuchos::ArrayRCP<const char> > &", pybind11::return_value_policy::automatic, pybind11::arg(""));
			cl.def("type", (const class std::type_info & (Teuchos::any::placeholder::*)() const) &Teuchos::any::placeholder::type, ". \n\nC++: Teuchos::any::placeholder::type() const --> const class std::type_info &", pybind11::return_value_policy::automatic);
			cl.def("typeName", (std::string (Teuchos::any::placeholder::*)() const) &Teuchos::any::placeholder::typeName, ". \n\nC++: Teuchos::any::placeholder::typeName() const --> std::string");
			cl.def("clone", (class Teuchos::any::placeholder * (Teuchos::any::placeholder::*)() const) &Teuchos::any::placeholder::clone, ". \n\nC++: Teuchos::any::placeholder::clone() const --> class Teuchos::any::placeholder *", pybind11::return_value_policy::automatic);
			cl.def("same", (bool (Teuchos::any::placeholder::*)(const class Teuchos::any::placeholder &) const) &Teuchos::any::placeholder::same, ". \n\nC++: Teuchos::any::placeholder::same(const class Teuchos::any::placeholder &) const --> bool", pybind11::arg("other"));
			cl.def("assign", (class Teuchos::any::placeholder & (Teuchos::any::placeholder::*)(const class Teuchos::any::placeholder &)) &Teuchos::any::placeholder::operator=, "C++: Teuchos::any::placeholder::operator=(const class Teuchos::any::placeholder &) --> class Teuchos::any::placeholder &", pybind11::return_value_policy::automatic, pybind11::arg(""));
		}

	}
	{ // Teuchos::bad_any_cast file:Teuchos_any.hpp line:324
		pybind11::class_<Teuchos::bad_any_cast, Teuchos::RCP<Teuchos::bad_any_cast>, PyCallBack_Teuchos_bad_any_cast, std::runtime_error> cl(M("Teuchos"), "bad_any_cast", "Thrown if any_cast is attempted between two incompatable types.");
		cl.def( pybind11::init<const std::string>(), pybind11::arg("msg") );

		cl.def( pybind11::init( [](PyCallBack_Teuchos_bad_any_cast const &o){ return new PyCallBack_Teuchos_bad_any_cast(o); } ) );
		cl.def( pybind11::init( [](Teuchos::bad_any_cast const &o){ return new Teuchos::bad_any_cast(o); } ) );
		cl.def("assign", (class Teuchos::bad_any_cast & (Teuchos::bad_any_cast::*)(const class Teuchos::bad_any_cast &)) &Teuchos::bad_any_cast::operator=, "C++: Teuchos::bad_any_cast::operator=(const class Teuchos::bad_any_cast &) --> class Teuchos::bad_any_cast &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	// Teuchos::any_cast(class Teuchos::any &) file:Teuchos_any.hpp line:339
	M("Teuchos").def("any_cast", (int & (*)(class Teuchos::any &)) &Teuchos::any_cast<int>, "C++: Teuchos::any_cast(class Teuchos::any &) --> int &", pybind11::return_value_policy::automatic, pybind11::arg("operand"));

	// Teuchos::any_cast(class Teuchos::any &) file:Teuchos_any.hpp line:339
	M("Teuchos").def("any_cast", (double & (*)(class Teuchos::any &)) &Teuchos::any_cast<double>, "C++: Teuchos::any_cast(class Teuchos::any &) --> double &", pybind11::return_value_policy::automatic, pybind11::arg("operand"));

	// Teuchos::any_cast(class Teuchos::any &) file:Teuchos_any.hpp line:339
	M("Teuchos").def("any_cast", (std::string & (*)(class Teuchos::any &)) &Teuchos::any_cast<std::string>, "C++: Teuchos::any_cast(class Teuchos::any &) --> std::string &", pybind11::return_value_policy::automatic, pybind11::arg("operand"));

	// Teuchos::any_cast(class Teuchos::any &) file:Teuchos_any.hpp line:339
	M("Teuchos").def("any_cast", (class Teuchos::ParameterList & (*)(class Teuchos::any &)) &Teuchos::any_cast<Teuchos::ParameterList>, "C++: Teuchos::any_cast(class Teuchos::any &) --> class Teuchos::ParameterList &", pybind11::return_value_policy::automatic, pybind11::arg("operand"));

	// Teuchos::any_cast(class Teuchos::any &) file:Teuchos_any.hpp line:339
	M("Teuchos").def("any_cast", (bool & (*)(class Teuchos::any &)) &Teuchos::any_cast<bool>, "C++: Teuchos::any_cast(class Teuchos::any &) --> bool &", pybind11::return_value_policy::automatic, pybind11::arg("operand"));

	// Teuchos::any_cast(const class Teuchos::any &) file:Teuchos_any.hpp line:375
	M("Teuchos").def("any_cast", (const int & (*)(const class Teuchos::any &)) &Teuchos::any_cast<int>, "C++: Teuchos::any_cast(const class Teuchos::any &) --> const int &", pybind11::return_value_policy::automatic, pybind11::arg("operand"));

	// Teuchos::any_cast(const class Teuchos::any &) file:Teuchos_any.hpp line:375
	M("Teuchos").def("any_cast", (const double & (*)(const class Teuchos::any &)) &Teuchos::any_cast<double>, "C++: Teuchos::any_cast(const class Teuchos::any &) --> const double &", pybind11::return_value_policy::automatic, pybind11::arg("operand"));

	// Teuchos::any_cast(const class Teuchos::any &) file:Teuchos_any.hpp line:375
	M("Teuchos").def("any_cast", (const std::string & (*)(const class Teuchos::any &)) &Teuchos::any_cast<std::string>, "C++: Teuchos::any_cast(const class Teuchos::any &) --> const std::string &", pybind11::return_value_policy::automatic, pybind11::arg("operand"));

	// Teuchos::any_cast(const class Teuchos::any &) file:Teuchos_any.hpp line:375
	M("Teuchos").def("any_cast", (const class Teuchos::ParameterList & (*)(const class Teuchos::any &)) &Teuchos::any_cast<Teuchos::ParameterList>, "C++: Teuchos::any_cast(const class Teuchos::any &) --> const class Teuchos::ParameterList &", pybind11::return_value_policy::automatic, pybind11::arg("operand"));

	// Teuchos::any_cast(const class Teuchos::any &) file:Teuchos_any.hpp line:375
	M("Teuchos").def("any_cast", (const bool & (*)(const class Teuchos::any &)) &Teuchos::any_cast<bool>, "C++: Teuchos::any_cast(const class Teuchos::any &) --> const bool &", pybind11::return_value_policy::automatic, pybind11::arg("operand"));

	// Teuchos::toString(const class Teuchos::any &) file:Teuchos_any.hpp line:398
	M("Teuchos").def("toString", (std::string (*)(const class Teuchos::any &)) &Teuchos::toString, "Converts the value in any to a std::string.\n    \n\n This function with throw an exception if\n             the held type can't be printed via operator<< !\n\nC++: Teuchos::toString(const class Teuchos::any &) --> std::string", pybind11::arg("rhs"));

	// Teuchos::swap(class Teuchos::any &, class Teuchos::any &) file:Teuchos_any.hpp line:439
	M("Teuchos").def("swap", (void (*)(class Teuchos::any &, class Teuchos::any &)) &Teuchos::swap, "Special swap for other code to find via Argument Dependent Lookup\n\nC++: Teuchos::swap(class Teuchos::any &, class Teuchos::any &) --> void", pybind11::arg("a"), pybind11::arg("b"));

}
