#include <map>
#include <algorithm>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

#include <pybind11/pybind11.h>

typedef std::function< pybind11::module & (std::string const &) > ModuleGetter;

void bind_std_postypes(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_std_exception(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_std_typeinfo(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_std_locale_classes(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_std_istream_tcc(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_std_sstream_tcc(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Teuchos_ParameterListExceptions(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Teuchos_TypeNameTraits(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Teuchos_ENull(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Teuchos_Ptr(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Teuchos_RCP(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Teuchos_ArrayRCPDecl(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Teuchos_FancyOStream(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Teuchos_ScalarTraits(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Teuchos_Time(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Teuchos_DefaultSerialComm(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Teuchos_CommHelpers(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Teuchos_CommHelpers_1(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Tpetra_ConfigDefs(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Tpetra_Map_def(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Tpetra_Util(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Tpetra_Util_1(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Tpetra_SrcDistObject(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Teuchos_BLAS_types(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Tpetra_Access(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Tpetra_MultiVector_decl(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Teuchos_Object(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Kokkos_StaticCrsGraph(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_KokkosSparse_CrsMatrix(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Tpetra_Packable(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Tpetra_CrsGraph_decl(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Teuchos_VerboseObject(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Tpetra_Distributor(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Teuchos_ParameterListAcceptor(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Tpetra_Details_DistributorPlan(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Tpetra_Import_decl(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Tpetra_RowMatrix_decl(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_PyTrilinos2_Tpetra_ETI(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Teuchos_SerializationTraits(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Tpetra_MultiVector_decl_1(std::function< pybind11::module &(std::string const &namespace_) > &M);


PYBIND11_MODULE(PyTrilinos2, root_module) {
	root_module.doc() = "PyTrilinos2 module";

	std::map <std::string, pybind11::module> modules;
	ModuleGetter M = [&](std::string const &namespace_) -> pybind11::module & {
		auto it = modules.find(namespace_);
		if( it == modules.end() ) throw std::runtime_error("Attempt to access pybind11::module for namespace " + namespace_ + " before it was created!!!");
		return it->second;
	};

	modules[""] = root_module;

	static std::vector<std::string> const reserved_python_words {"nonlocal", "global", };

	auto mangle_namespace_name(
		[](std::string const &ns) -> std::string {
			if ( std::find(reserved_python_words.begin(), reserved_python_words.end(), ns) == reserved_python_words.end() ) return ns;
			else return ns+'_';
		}
	);

	std::vector< std::pair<std::string, std::string> > sub_modules {
		{"", "Kokkos"},
		{"", "KokkosSparse"},
		{"", "Teuchos"},
		{"Teuchos", "Exceptions"},
		{"Teuchos", "PtrPrivateUtilityPack"},
		{"", "Tpetra"},
		{"Tpetra", "Access"},
		{"Tpetra", "Details"},
		{"Tpetra", "SortDetails"},
		{"", "std"},
	};
	for(auto &p : sub_modules ) modules[p.first.size() ? p.first+"::"+p.second : p.second] = modules[p.first].def_submodule( mangle_namespace_name(p.second).c_str(), ("Bindings for " + p.first + "::" + p.second + " namespace").c_str() );

	//pybind11::class_<std::shared_ptr<void>>(M(""), "_encapsulated_data_");

	bind_std_postypes(M);
	bind_std_exception(M);
	bind_std_typeinfo(M);
	bind_std_locale_classes(M);
	bind_std_istream_tcc(M);
	bind_std_sstream_tcc(M);
	bind_Teuchos_ParameterListExceptions(M);
	bind_Teuchos_TypeNameTraits(M);
	bind_Teuchos_ENull(M);
	bind_Teuchos_Ptr(M);
	bind_Teuchos_RCP(M);
	bind_Teuchos_ArrayRCPDecl(M);
	bind_Teuchos_FancyOStream(M);
	bind_Teuchos_ScalarTraits(M);
	bind_Teuchos_Time(M);
	bind_Teuchos_DefaultSerialComm(M);
	bind_Teuchos_CommHelpers(M);
	bind_Teuchos_CommHelpers_1(M);
	bind_Tpetra_ConfigDefs(M);
	bind_Tpetra_Map_def(M);
	bind_Tpetra_Util(M);
	bind_Tpetra_Util_1(M);
	bind_Tpetra_SrcDistObject(M);
	bind_Teuchos_BLAS_types(M);
	bind_Tpetra_Access(M);
	bind_Tpetra_MultiVector_decl(M);
	bind_Teuchos_Object(M);
	bind_Kokkos_StaticCrsGraph(M);
	bind_KokkosSparse_CrsMatrix(M);
	bind_Tpetra_Packable(M);
	bind_Tpetra_CrsGraph_decl(M);
	bind_Teuchos_VerboseObject(M);
	bind_Tpetra_Distributor(M);
	bind_Teuchos_ParameterListAcceptor(M);
	bind_Tpetra_Details_DistributorPlan(M);
	bind_Tpetra_Import_decl(M);
	bind_Tpetra_RowMatrix_decl(M);
	bind_PyTrilinos2_Tpetra_ETI(M);
	bind_Teuchos_SerializationTraits(M);
	bind_Tpetra_MultiVector_decl_1(M);

}
