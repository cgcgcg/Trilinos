#include "Teuchos_Reporter.hpp"
#include "Teuchos_Assert.hpp"
#include "Teuchos_EReductionType.hpp"
#include "Teuchos_FancyOStream_decl.hpp"
#include "Teuchos_ParameterEntry.hpp"
#include "Teuchos_StackedTimer.hpp"
#include "Teuchos_StrUtils.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include <format>

namespace Teuchos {

bool globalReportingEnabled() {
  static bool enableReportingDefault = false;
  return enableReportingDefault;
}

Report::Report(Reporter &reporter, const std::string &name)
    : reporter_(reporter), name_(name) {}

Report::Report(Reporter &reporter, const std::string &name,
               const std::string &context, Teuchos::ParameterList &data)
    : reporter_(reporter), name_(name), context_(context) {
  data_ = Teuchos::rcpFromRef(data);
}

void Report::addReduction(std::string const &name, ReductionOpType op,
                          const std::string reducedName) {
  if (op != None) {
    reporter_.registerForMpiReduction(context_ + "@" + name, op,
                                      context_ + "@" + reducedName);
  }
}

void Report::setPrintFormatting(std::string formatString) {
  reporter_.setPrintFormatting(name_, formatString);
}

Teuchos::Report Reporter::addReport(const std::string name,
                                    const bool addToExistingReport) {
  Teuchos::RCP<Teuchos::ParameterList> pl = Teuchos::null;
  if (reportingEnabled_) {

    auto stacked_timer = Teuchos::TimeMonitor::getStackedTimer();
    auto contextStr = stacked_timer->getRunningTimerName();
    auto context = StrUtils::splitString(contextStr, '@');

    auto pl = Teuchos::rcpFromRef(data_);
    for (auto lvl : context)
      pl = sublist(pl, lvl);

    auto &reportsList = pl->sublist(name);
    if (addToExistingReport) {
      return Teuchos::Report(*this, name, contextStr + "@" + name, reportsList);
    } else {
      auto numReports = reportsList.numParams();
      auto &reportList = reportsList.sublist(std::to_string(numReports));
      return Teuchos::Report(
                             *this, name, contextStr + "@" + name + "@" + std::to_string(numReports),
          reportList);
    }
  } else
    return Teuchos::Report(*this, name);
}

void Reporter::registerForMpiReduction(const std::string name,
                                       ReductionOpType op,
                                       const std::string nameReduced) {
  if (reportingEnabled_) {
    auto names = StrUtils::splitString(name, '@');
    auto pl = Teuchos::rcpFromRef(data_);
    for (int lvl = 0; lvl < names.size() - 1; ++lvl) {
      auto name = names[lvl];
      pl = sublist(pl, name);
    }
    auto e = pl->getEntry(names[names.size() - 1]).getAny();
    auto t = e.typeName();

    auto key = std::make_pair(op, t);
    auto value = std::make_pair(name, nameReduced);
    reductions_[key].insert(value);
  }
}

void Reporter::applyMpiReductions(Teuchos::RCP<const Teuchos::Comm<int>> comm) {
  for (auto red : reductions_) {
    auto op = red.first.first;
    auto typeStr = red.first.second;
    auto l = red.second;

    if (typeStr == "int") {
      applyMpiReductions<int>(comm, op, l);
    } else if (typeStr == "double") {
      applyMpiReductions<double>(comm, op, l);
    } else
      TEUCHOS_ASSERT(false);
  }
}

#ifdef HAVE_TEUCHOSPARAMETERLIST_JSON
nlohmann::json Reporter::toJSON() const { return data_.toJSON(); }
#endif

void Reporter::print(Teuchos::FancyOStream &out, std::string &listName, int lvl,
                     Teuchos::ParameterList &pl) {
  for (ParameterList::ConstIterator i = pl.begin(); i != pl.end(); ++i) {
    RCP<const ParameterEntry> entry = pl.getEntryRCP(i->first);
    if (entry->isList()) {
      auto paramName = pl.name(i);
      auto &spl = Teuchos::getValue<Teuchos::ParameterList>(entry);
      auto j = formatStrings_.find(paramName);
      if (j != formatStrings_.end()) {
        auto formatString = j->second.first;
        auto formatFields = j->second.second;
        Teuchos::Array<std::string> values;
        bool success = true;

        for (int i = 0; i < formatFields.size(); ++i) {
          auto formatField = StrUtils::subString(formatFields[i], 1, formatFields[i].size()-1);
          if (formatField == "name") {
            values.push_back(listName);
          } else if (formatField == "indentation") {
            std::string indent;
            for (int i = 0; i < lvl; ++i)
              indent += "|   ";
            values.push_back(indent);
          } else {
            if (!spl.isParameter(formatField)) {
              success = false;
              break;
            }
            std::stringstream ssE;
            ssE << spl.getEntry(formatField).getAny();
            values.push_back(ssE.str());
          }
        }
        if (success)
          out << StrUtils::varTableSubstitute(formatString, formatFields, values);
      } else {
        print(out, paramName, lvl + 1, spl);
      }
    }
  }
}

void Reporter::setPrintFormatting(std::string name,
                                  std::string formatString) {
  auto substrings = StrUtils::splitString(formatString, '@');
  Teuchos::Array<std::string> formatFields;
  int start, end;
  bool inField = false;
  for (int k = 0; k<formatString.size(); ++k) {
    if (formatString[k] == '@') {
      if (!inField) {
        inField = true;
        start = k;
      } else {
        inField = false;
        end = k;
        formatFields.push_back(StrUtils::subString(formatString, start, end+1));
      }
    }
  }
  formatStrings_[name] = std::make_pair(formatString, formatFields);
}

void Reporter::print(Teuchos::FancyOStream &out) {
  std::string listName;
  print(out, listName, 0, data_);
}

Reporter &getReporter() {
  static Reporter reporter = Reporter();
  return reporter;
}

} // namespace Teuchos
