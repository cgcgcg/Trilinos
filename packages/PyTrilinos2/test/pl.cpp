#include <iostream>
#include <Tpetra_Core.hpp>
#include "Teuchos_ParameterListExceptions.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include <Teuchos_XMLParameterListHelpers.hpp>
#include "Teuchos_stacktrace.hpp"
#include "MueLu_MasterList.hpp"

int main(int argc, char **argv) {
  Tpetra::ScopeGuard tpetraScope(&argc, &argv);
  {
    std::string xmlFileName = "test.xml";
    auto comm = Tpetra::getDefaultComm();
    auto paramList = Teuchos::rcp(new Teuchos::ParameterList());
    Teuchos::updateParametersFromXmlFileAndBroadcast(xmlFileName,
                                                     paramList.ptr(), *comm);
    const auto &validList = *MueLu::MasterList::List();
    try {
      paramList->validateParameters(validList);
    } catch (Teuchos::Exceptions::InvalidParameterName &e) {
      std::stringstream ss;
      ss << "Error message:\n";
      ss << e.what();
      ss << std::endl;
#ifdef HAVE_TEUCHOS_STACKTRACE
      ss << "Stacktrace:";
      ss << Teuchos::get_stacktrace();
      ss << std::endl;
#endif
      std::cout << ss.str();
      throw;
    }
  }
}
