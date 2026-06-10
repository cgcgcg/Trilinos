#include "Teuchos_Reporter.hpp"
#include "Teuchos_StackedTimer.hpp"
#include "Teuchos_StrUtils.hpp"
#include "Teuchos_TimeMonitor.hpp"

namespace Teuchos {

bool globalReportingEnabled() {
  static bool enableReportingDefault = false;
  return enableReportingDefault;
}

Teuchos::RCP<Teuchos::ParameterList>
Reporter::addReport(const std::string name) {
  if (reportingEnabled_) {

    auto stacked_timer = Teuchos::TimeMonitor::getStackedTimer();
    auto contextStr = stacked_timer->getRunningTimerName();
    auto context = StrUtils::splitString(contextStr, '@');

    auto pl = Teuchos::rcpFromRef(data_);
    for (auto lvl : context)
      pl = sublist(pl, lvl);

    auto &reportsList = pl->sublist(name);
    auto numReports = reportsList.numParams();
    // if ((numReports>0) && (!reportsList.isSublist("0"))) {

    // }
    auto &reportList = reportsList.sublist(std::to_string(numReports));
    return Teuchos::rcpFromRef(reportList);
  } else {
    return Teuchos::null;
  }
}

#ifdef HAVE_TEUCHOSPARAMETERLIST_JSON
nlohmann::json
Reporter::toJSON() const {
  return data_.toJSON();
}
#endif

Reporter &getReporter() {
  static Reporter reporter = Reporter();
  return reporter;
}

} // namespace Teuchos
