#include "Teuchos_FancyOStream.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_VerbosityLevel.hpp"
#include "Tpetra_Access.hpp"
#include <Tpetra_Core.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_Vector.hpp>

#include <pybind11/embed.h>
namespace py = pybind11;
using namespace py::literals;

using map_type = Tpetra::Map<>;
using vec_type = Tpetra::Vector<>;
using scalar_type = vec_type::scalar_type;
using local_ordinal_type = vec_type::local_ordinal_type;
using global_ordinal_type = vec_type::global_ordinal_type;

int main(int argc, char *argv[]) {
  Tpetra::ScopeGuard tpetraScope(&argc, &argv);
  {
    auto comm = Tpetra::getDefaultComm();
    const Tpetra::global_size_t numGblIndices = 50;

    auto out = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));

    const global_ordinal_type indexBase = 0;
    auto map = rcp(new map_type(numGblIndices, indexBase, comm));
    auto vec = rcp(new vec_type(map));

    vec->putScalar(1.);

    vec->describe(*out, Teuchos::VERB_EXTREME);
    std::cout << vec->getLocalViewHost(Tpetra::Access::ReadOnly).data() << std::endl;

    py::scoped_interpreter guard{};
    py::module_ PyTrilinos2 = py::module_::import("PyTrilinos2");
    auto locals = py::dict("vec"_a = *vec);
    py::exec(R"(
        import numpy as np

        vec.putScalar(2.)
        print(np.info(vec.getLocalViewHost()))
        v = vec.getLocalViewHost()
        v[:] = 3.
    )", py::globals(), locals);

    vec->describe(*out, Teuchos::VERB_EXTREME);
  }
  return 0;
}
