#ifndef TEUCHOS_REPORTER_HPP
#define TEUCHOS_REPORTER_HPP

#include "Teuchos_Comm.hpp"
#include "Teuchos_CommHelpers.hpp"
#include "Teuchos_EReductionType.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_PerformanceMonitorBase.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_StrUtils.hpp"
#include <limits>
#include <map>
#include <set>
#include <string>
#include <typeinfo>

namespace Teuchos {

enum ReductionOpType : int {
  None,
  Sum,
  Mean,
  StdDev,
  Min,
  Max,
  ArgMin,
  ArgMax,
  Gather,
  // Histogram
};

bool globalReportingEnabled();

class Report;

class Reporter {
public:
  Reporter() {
    reportingEnabled_ = globalReportingEnabled();
    data_ = Teuchos::ParameterList();
  }

  Reporter(const Reporter &) = delete;
  Reporter &operator=(const Reporter &) = delete;

  void setReportingEnabled(bool state) { reportingEnabled_ = state; }

  bool isReportingEnabled() const { return reportingEnabled_; }

  Teuchos::Report addReport(const std::string name,
                            const bool addToExistingReport = true);

  void registerForMpiReduction(const std::string name, ReductionOpType op,
                               const std::string nameReduced);

  template <typename T>
  void applyMpiReductions(Teuchos::RCP<const Teuchos::Comm<int>> comm,
                          ReductionOpType op,
                          std::set<std::pair<std::string, std::string>> &l) {

    const int reduceToRank = 0;

    Array<std::string> fullnames;
    for (auto q : l) {
      auto fullname = q.first;
      fullnames.push_back(fullname);
    }

    Array<std::string> merged_full_names;
    mergeCounterNames(*comm, fullnames, merged_full_names, Union);

    size_t numValues = merged_full_names.size();
    Teuchos::Array<int> reorder(l.size(), -1);
    Teuchos::Array<int> owner(numValues, comm->getSize());
    Teuchos::Array<int> owner2(numValues, comm->getSize());

    int j = 0;
    for (auto q : l) {
      auto fullname = q.first;

      int k = 0;
      for (k = 0; k < numValues; ++k) {
        if (merged_full_names[k] == fullname) {
          reorder[j] = k;
          owner[k] = comm->getRank();
          break;
        }
      }
      TEUCHOS_ASSERT(merged_full_names[k] == fullname);
      ++j;
    }
    Teuchos::reduceAll<int, int>(*comm, REDUCE_MIN, numValues, owner.data(),
                                 owner2.data());

    Teuchos::Array<std::string> newNames;
    Teuchos::Array<std::string> myNewNames;
    j = 0;
    for (auto q : l) {
      if (owner2[reorder[j]] == comm->getRank()) {
        auto newName = q.second;
        myNewNames.push_back(newName);
      }
      ++j;
    }

    if (comm->getRank() == reduceToRank) {
      newNames = Teuchos::Array<std::string>(numValues);

      std::set<int> receiveRanks;
      for (j = 0; j < numValues; ++j) {
        receiveRanks.insert(owner2[j]);
      }

      for (auto ownerRank : receiveRanks) {
        Teuchos::Array<std::string> receivedNewNames;

        if (ownerRank == reduceToRank) {
          receivedNewNames = myNewNames;
        } else {
          Teuchos::receiveStrings(*comm, ownerRank, receivedNewNames);
        }

        int k = 0;
        for (int i = 0; i < numValues; ++i) {
          if (owner2[i] == ownerRank) {
            newNames[i] = receivedNewNames[k];
            ++k;
          }
        }
      }
    } else {
      if (myNewNames.size() > 0) {
        Teuchos::sendStrings(*comm, myNewNames, reduceToRank);
      }
    }

    Teuchos::Array<T> values;
    Teuchos::Array<T> reducedValues;
    Teuchos::Array<T> reducedValues2;
    Teuchos::Array<int> pos;
    Teuchos::Array<int> pos2;

    T initVal;
    if (op == Gather)
      initVal = std::numeric_limits<T>::quiet_NaN();
    else if ((op == Min) || (op == ArgMin))
      initVal = std::numeric_limits<T>::max();
    else if ((op == Max) || (op == ArgMax))
      initVal = std::numeric_limits<T>::lowest();
    else
      initVal = 0;
    values = Teuchos::Array<T>(numValues, initVal);

    j = 0;
    for (auto q : l) {
      auto fullname = q.first;

      int k = 0;
      for (k = 0; k < numValues; ++k) {
        if (merged_full_names[k] == fullname) {
          reorder[j] = k;
          break;
        }
      }
      TEUCHOS_ASSERT(merged_full_names[k] == fullname);

      values[k] = getValue<T>(fullname);
      ++j;
    }

    switch (op) {
    case None:
      break;
    case Sum:
      if (comm->getRank() == reduceToRank)
        reducedValues = Teuchos::Array<T>(numValues);
      Teuchos::reduce<int, T>(values.data(), reducedValues.data(), numValues,
                              REDUCE_SUM, reduceToRank, *comm);
      break;
    case Mean:
      if (comm->getRank() == reduceToRank)
        reducedValues = Teuchos::Array<T>(numValues);
      Teuchos::reduce<int, T>(values.data(), reducedValues.data(), numValues,
                              REDUCE_SUM, reduceToRank, *comm);
      pos = Teuchos::Array<int>(numValues, 0);
      pos2 = Teuchos::Array<int>(numValues, 0);
      for (int i = 0; i < numValues; ++i) {
        if (values[i] != initVal)
          pos[i] = 1;
      }
      Teuchos::reduce<int, int>(pos.data(), pos2.data(), numValues, REDUCE_SUM,
                                reduceToRank, *comm);
      if (comm->getRank() == reduceToRank)
        for (int i = 0; i < numValues; ++i)
          reducedValues[i] /= pos2[i];
      break;
    case StdDev:
      if (comm->getRank() == reduceToRank)
        reducedValues = Teuchos::Array<T>(numValues);
      Teuchos::reduce<int, T>(values.data(), reducedValues.data(), numValues,
                              REDUCE_SUM, reduceToRank, *comm);

      if (comm->getRank() == reduceToRank)
        reducedValues2 = Teuchos::Array<T>(numValues);
      for (int i = 0; i < numValues; ++i)
        values[i] = values[i] * values[i];
      Teuchos::reduce<int, T>(values.data(), reducedValues2.data(), numValues,
                              REDUCE_SUM, reduceToRank, *comm);
      pos = Teuchos::Array<int>(numValues, 0);
      pos2 = Teuchos::Array<int>(numValues, 0);
      for (int i = 0; i < numValues; ++i) {
        if (values[i] != initVal)
          pos[i] = 1;
      }
      Teuchos::reduce<int, int>(pos.data(), pos2.data(), numValues, REDUCE_SUM,
                                reduceToRank, *comm);
      if (comm->getRank() == reduceToRank) {
        for (int i = 0; i < numValues; ++i) {
          reducedValues[i] = Teuchos::ScalarTraits<T>::squareroot(
              (reducedValues2[i] / pos2[i]) -
              (reducedValues[i] / pos2[i]) * (reducedValues[i] / pos2[i]));
        }
      }
      break;
    case Min:
      if (comm->getRank() == reduceToRank)
        reducedValues = Teuchos::Array<T>(numValues);
      Teuchos::reduce<int, T>(values.data(), reducedValues.data(), numValues,
                              REDUCE_MIN, reduceToRank, *comm);
      break;
    case Max:
      if (comm->getRank() == reduceToRank)
        reducedValues = Teuchos::Array<T>(numValues);
      Teuchos::reduce<int, T>(values.data(), reducedValues.data(), numValues,
                              REDUCE_MAX, reduceToRank, *comm);
      break;
    case ArgMin:
      reducedValues = Teuchos::Array<T>(numValues);
      Teuchos::reduceAll<int, T>(*comm, REDUCE_MIN, numValues, values.data(),
                                 reducedValues.data());
      pos = Teuchos::Array<int>(numValues, 0);
      for (int i = 0; i < numValues; ++i)
        if (values[i] == reducedValues[i])
          pos[i] = comm->getRank();
      pos2 = Teuchos::Array<int>(numValues);
      Teuchos::reduce<int, int>(pos.data(), pos2.data(), numValues, REDUCE_MAX,
                                reduceToRank, *comm);
      break;
    case ArgMax:
      reducedValues = Teuchos::Array<T>(numValues);
      Teuchos::reduceAll<int, T>(*comm, REDUCE_MAX, numValues, values.data(),
                                 reducedValues.data());
      pos = Teuchos::Array<int>(numValues, 0);
      for (int i = 0; i < numValues; ++i)
        if (values[i] == reducedValues[i])
          pos[i] = comm->getRank();
      pos2 = Teuchos::Array<int>(numValues);
      Teuchos::reduce<int, int>(pos.data(), pos2.data(), numValues, REDUCE_MAX,
                                reduceToRank, *comm);
      break;
    case Gather:
      if (comm->getRank() == reduceToRank)
        reducedValues = Teuchos::Array<T>(numValues * comm->getSize());
      Teuchos::gather<int, T>(values.data(), numValues, reducedValues.data(),
                              comm->getSize() * numValues, reduceToRank, *comm);
      break;
    }

    if (comm->getRank() == reduceToRank) {
      for (int k = 0; k < numValues; ++k) {
        auto fullname = newNames[k];

        if ((op == ArgMin) || (op == ArgMax)) {
          setValue(fullname, pos2[k]);
        } else if (op == Gather) {
          Teuchos::Array<T> vals(comm->getSize());
          for (int i = 0; i < comm->getSize(); ++i)
            vals[i] = reducedValues[k + i * numValues];
          setValue(fullname, vals);
        } else
          setValue(fullname, reducedValues[k]);
      }
    }
  }

  void applyMpiReductions(Teuchos::RCP<const Teuchos::Comm<int>> comm);

  Teuchos::ParameterList &getData() { return data_; }

#ifdef HAVE_TEUCHOSPARAMETERLIST_JSON
  nlohmann::json toJSON() const;
#endif

  void setPrintFormatting(std::string name, std::string formatString);
  void print(Teuchos::FancyOStream &out, std::string &listName, int lvl,
             Teuchos::ParameterList &pl);
  void print(Teuchos::FancyOStream &out);

private:
  template <typename T> T getValue(std::string const &contextStr) {
    auto names = StrUtils::splitString(contextStr, '@');
    auto pl = Teuchos::rcpFromRef(data_);
    for (int lvl = 0; lvl < names.size() - 1; ++lvl) {
      auto name = names[lvl];
      pl = sublist(pl, name);
    }
    return pl->get<T>(names[names.size() - 1]);
  };

  template <typename T> void setValue(std::string const &contextStr, T value) {
    auto names = StrUtils::splitString(contextStr, '@');
    auto pl = Teuchos::rcpFromRef(data_);
    for (int lvl = 0; lvl < names.size() - 1; ++lvl) {
      auto name = names[lvl];
      pl = sublist(pl, name);
    }
    pl->set<T>(names[names.size() - 1], value);
  };

  template <typename T>
  void setValue(std::string const &contextStr, T value[]) {
    auto names = StrUtils::splitString(contextStr, '@');
    auto pl = Teuchos::rcpFromRef(data_);
    for (int lvl = 0; lvl < names.size() - 1; ++lvl) {
      auto name = names[lvl];
      pl = sublist(pl, name);
    }
    pl->set<T>(names[names.size() - 1], value);
  };

  bool reportingEnabled_;
  Teuchos::ParameterList data_;
  std::map<std::pair<ReductionOpType, std::string>,
           std::set<std::pair<std::string, std::string>>>
      reductions_;
  std::map<std::string, std::pair<std::string, Teuchos::Array<std::string>>>
      formatStrings_;
};

class Report {
public:
  Report(Reporter &reporter, const std::string &name);

  Report(Reporter &reporter, const std::string &name,
         const std::string &context, Teuchos::ParameterList &data);

  template <enum ReductionOpType op, typename T>
  void set(std::string const &name, T value) {
    static_assert(op != Mean);
    static_assert(op != StdDev);
    static_assert(op != ArgMin);
    static_assert(op != ArgMax);
    // static_assert(op != Histogram);

    if (!data_.is_null()) {
      if (op == None)
        data_->set(name, value);
      else if (op == Sum) {
        if (data_->isParameter(name)) {
          data_->set(name, value + data_->get<T>(name));
        } else {
          data_->set(name, value);
        }
      } else if (op == Min) {
        if (data_->isParameter(name)) {
          data_->set(name, std::min(value, data_->get<T>(name)));
        } else {
          data_->set(name, value);
        }
      } else if (op == Max) {
        if (data_->isParameter(name)) {
          data_->set(name, std::max(value, data_->get<T>(name)));
        } else {
          data_->set(name, value);
        }
      } else if (op == Gather) {
        if (data_->isParameter(name)) {
          auto values = data_->get<Teuchos::Array<T>>(name);
          values.push_back(value);
          data_->set(name, values);
        } else {
          Teuchos::Array<T> values;
          values.push_back(value);
          data_->set(name, values);
        }
      }
    }
  }

  template <enum ReductionOpType op, enum ReductionOpType op2, typename T>
  void set(std::string const &name, T value, const std::string reducedName) {
    set<op, T>(name, value);
    if (op2 != None)
      addReduction(name, op2, reducedName);
  }

  void addReduction(std::string const &name, ReductionOpType op,
                    const std::string reducedName);

  void setPrintFormatting(std::string formatString);

  std::string getName() const { return name_; }
  std::string getContext() const { return context_; }

private:
  Reporter &reporter_;
  std::string name_;
  std::string context_;
  Teuchos::RCP<Teuchos::ParameterList> data_;
};

Reporter &getReporter();

} // namespace Teuchos

#endif
