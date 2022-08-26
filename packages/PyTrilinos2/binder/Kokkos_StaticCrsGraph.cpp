#include <Kokkos_Concepts.hpp> // Kokkos::Device
#include <Kokkos_HostSpace.hpp> // 
#include <Kokkos_HostSpace.hpp> // Kokkos::HostSpace
#include <Kokkos_Layout.hpp> // Kokkos::LayoutLeft
#include <Kokkos_MemoryTraits.hpp> // Kokkos::MemoryTraits
#include <Kokkos_Serial.hpp> // Kokkos::Serial
#include <Kokkos_StaticCrsGraph.hpp> // Kokkos::StaticCrsGraph
#include <sstream> // __str__

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

void bind_Kokkos_StaticCrsGraph(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // Kokkos::StaticCrsGraph file:Kokkos_StaticCrsGraph.hpp line:285
		pybind11::class_<Kokkos::StaticCrsGraph<int,Kokkos::LayoutLeft,Kokkos::HostSpace,Kokkos::MemoryTraits<0>,unsigned long>, Teuchos::RCP<Kokkos::StaticCrsGraph<int,Kokkos::LayoutLeft,Kokkos::HostSpace,Kokkos::MemoryTraits<0>,unsigned long>>> cl(M("Kokkos"), "StaticCrsGraph_int_Kokkos_LayoutLeft_Kokkos_HostSpace_Kokkos_MemoryTraits_0_unsigned_long_t", "");
		cl.def( pybind11::init( [](){ return new Kokkos::StaticCrsGraph<int,Kokkos::LayoutLeft,Kokkos::HostSpace,Kokkos::MemoryTraits<0>,unsigned long>(); } ) );
		cl.def( pybind11::init( [](Kokkos::StaticCrsGraph<int,Kokkos::LayoutLeft,Kokkos::HostSpace,Kokkos::MemoryTraits<0>,unsigned long> const &o){ return new Kokkos::StaticCrsGraph<int,Kokkos::LayoutLeft,Kokkos::HostSpace,Kokkos::MemoryTraits<0>,unsigned long>(o); } ) );
		cl.def_readwrite("entries", &Kokkos::StaticCrsGraph<int,Kokkos::LayoutLeft,Kokkos::HostSpace,Kokkos::MemoryTraits<0>,unsigned long>::entries);
		cl.def_readwrite("row_map", &Kokkos::StaticCrsGraph<int,Kokkos::LayoutLeft,Kokkos::HostSpace,Kokkos::MemoryTraits<0>,unsigned long>::row_map);
		cl.def_readwrite("row_block_offsets", &Kokkos::StaticCrsGraph<int,Kokkos::LayoutLeft,Kokkos::HostSpace,Kokkos::MemoryTraits<0>,unsigned long>::row_block_offsets);
		cl.def("assign", (class Kokkos::StaticCrsGraph<int, struct Kokkos::LayoutLeft, class Kokkos::HostSpace, struct Kokkos::MemoryTraits<0>, unsigned long> & (Kokkos::StaticCrsGraph<int,Kokkos::LayoutLeft,Kokkos::HostSpace,Kokkos::MemoryTraits<0>,unsigned long>::*)(const class Kokkos::StaticCrsGraph<int, struct Kokkos::LayoutLeft, class Kokkos::HostSpace, struct Kokkos::MemoryTraits<0>, unsigned long> &)) &Kokkos::StaticCrsGraph<int, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<0>, unsigned long>::operator=, "C++: Kokkos::StaticCrsGraph<int, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<0>, unsigned long>::operator=(const class Kokkos::StaticCrsGraph<int, struct Kokkos::LayoutLeft, class Kokkos::HostSpace, struct Kokkos::MemoryTraits<0>, unsigned long> &) --> class Kokkos::StaticCrsGraph<int, struct Kokkos::LayoutLeft, class Kokkos::HostSpace, struct Kokkos::MemoryTraits<0>, unsigned long> &", pybind11::return_value_policy::automatic, pybind11::arg("rhs"));
		cl.def("numRows", (unsigned long (Kokkos::StaticCrsGraph<int,Kokkos::LayoutLeft,Kokkos::HostSpace,Kokkos::MemoryTraits<0>,unsigned long>::*)() const) &Kokkos::StaticCrsGraph<int, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<0>, unsigned long>::numRows, "C++: Kokkos::StaticCrsGraph<int, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<0>, unsigned long>::numRows() const --> unsigned long");
		cl.def("is_allocated", (bool (Kokkos::StaticCrsGraph<int,Kokkos::LayoutLeft,Kokkos::HostSpace,Kokkos::MemoryTraits<0>,unsigned long>::*)() const) &Kokkos::StaticCrsGraph<int, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<0>, unsigned long>::is_allocated, "C++: Kokkos::StaticCrsGraph<int, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<0>, unsigned long>::is_allocated() const --> bool");
		cl.def("create_block_partitioning", [](Kokkos::StaticCrsGraph<int,Kokkos::LayoutLeft,Kokkos::HostSpace,Kokkos::MemoryTraits<0>,unsigned long> &o, unsigned long const & a0) -> void { return o.create_block_partitioning(a0); }, "", pybind11::arg("num_blocks"));
		cl.def("create_block_partitioning", (void (Kokkos::StaticCrsGraph<int,Kokkos::LayoutLeft,Kokkos::HostSpace,Kokkos::MemoryTraits<0>,unsigned long>::*)(unsigned long, unsigned long)) &Kokkos::StaticCrsGraph<int, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<0>, unsigned long>::create_block_partitioning, "C++: Kokkos::StaticCrsGraph<int, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<0>, unsigned long>::create_block_partitioning(unsigned long, unsigned long) --> void", pybind11::arg("num_blocks"), pybind11::arg("fix_cost_per_row"));
	}
}
