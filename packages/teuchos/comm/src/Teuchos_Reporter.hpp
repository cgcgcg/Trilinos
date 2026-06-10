#ifndef TEUCHOS_REPORTER_HPP
#define TEUCHOS_REPORTER_HPP

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"
#include <string>

namespace Teuchos {

bool globalReportingEnabled();

class Reporter {
public:
  Reporter() {
    reportingEnabled_ = globalReportingEnabled();
    data_ = Teuchos::ParameterList();
  }

  Reporter (const Reporter&) = delete;
  Reporter& operator= (const Reporter&) = delete;

  void setReportingEnabled(bool state) { reportingEnabled_ = state; }

  bool isReportingEnabled() const { return reportingEnabled_; }

  Teuchos::RCP<Teuchos::ParameterList> addReport(const std::string name);

  Teuchos::ParameterList &getData() { return data_; }

#ifdef HAVE_TEUCHOSPARAMETERLIST_JSON
  nlohmann::json toJSON() const;
#endif

private:
  bool reportingEnabled_;
  Teuchos::ParameterList data_;
};

Reporter &getReporter();

} // namespace Teuchos

#endif
