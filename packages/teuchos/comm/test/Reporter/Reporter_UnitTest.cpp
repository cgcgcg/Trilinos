// @HEADER
// *****************************************************************************
//                    Teuchos: Common Tools Package
//
// Copyright 2004 NTESS and the Teuchos contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#include "Teuchos_Assert.hpp"
#include "Teuchos_DefaultComm.hpp"
#include "Teuchos_FancyOStream_decl.hpp"
#include "Teuchos_Reporter.hpp"
#include "Teuchos_StackedTimer.hpp"
#include "Teuchos_TestingHelpers.hpp"
#include "Teuchos_UnitTestHarness.hpp"
#include <sstream>

namespace Teuchos {

  void checkConsistency(const Teuchos::ParameterList &pl, Teuchos::FancyOStream &out, bool success) {
  auto times = pl.get<Teuchos::Array<double >> ("proc times");

  auto minTime = pl.get<double>("time min");
  auto meanTime = pl.get<double>("time mean");
  auto maxTime = pl.get<double>("time max");

  auto procMin = pl.get<int>("proc min");
  auto procMax = pl.get<int>("proc max");
  auto count = pl.get<int>("active processes");

  auto minTime2 = std::numeric_limits<double>::max();
  auto maxTime2 = std::numeric_limits<double>::min();
  double sum2 = 0;
  int count2 = 0;
  for (int i = 0; i<times.size(); ++i) {
    minTime2 = std::min(minTime2, times[i]);
    maxTime2 = std::max(maxTime2, times[i]);
    if (!Teuchos::ScalarTraits<double>::isnaninf(times[i])) {
      sum2 += times[i];
      ++count2;
    }
  }
  double meanTime2 = sum2 / count2;
  int procMin2 = 0;
  int procMax2 = 0;
  for (int i = 0; i<times.size(); ++i) {
    if (minTime2 == times[i])
      procMin2 = i;
    if (maxTime2 == times[i])
      procMax2 = i;
  }

  TEUCHOS_TEST_FLOATING_EQUALITY(minTime, minTime2, 1e-12, out, success);
  TEUCHOS_TEST_FLOATING_EQUALITY(maxTime, maxTime2, 1e-12, out, success);
  TEUCHOS_ASSERT_EQUALITY(procMin, procMin2);
  TEUCHOS_ASSERT_EQUALITY(procMax, procMax2);
  TEUCHOS_ASSERT_EQUALITY(count, count2);
  TEUCHOS_TEST_FLOATING_EQUALITY(meanTime, meanTime2, 1e-12, out, success);

  if (pl.isParameter("time local")) {
    auto localTime = pl.get<double>("time local");
    TEUCHOS_TEST_FLOATING_EQUALITY(times[0], localTime, 1e-12, out, success);
  }
}

TEUCHOS_UNIT_TEST(Reporter, test) {
  const Teuchos::RCP<const Teuchos::Comm<int>> comm =
      Teuchos::DefaultComm<int>::getComm();

  auto &reporter = Teuchos::getReporter();
  reporter.setReportingEnabled(true);

  auto stacked_timer = rcp(new Teuchos::StackedTimer("Mini-EM"));
  Teuchos::TimeMonitor::setStackedTimer(stacked_timer);

  // auto report1 = reporter.addReport("my report");
  // report1.set("value0", 3.5);

  // auto report2 = reporter.addReport("my report");
  // report2.set("value0", 2.5);

  {
    stacked_timer->start("child");
    // auto child_report0 = reporter.addReport("my report");
    // child_report0.set("value0", 1.5);
    stacked_timer->stop("child");
  }

  if (comm->getRank() == 0) {
    stacked_timer->start("rank0");
    // auto child_report0 = reporter.addReport("my report");
    // child_report0.set("value0", 1.5);
    stacked_timer->stop("rank0");
  }
  if (comm->getRank() == 1) {
    stacked_timer->start("rank1");
    // auto child_report0 = reporter.addReport("my report");
    // child_report0.set("value0", 1.5);
    stacked_timer->stop("rank1");
  }

  if (comm->getRank() >= 2) {
    stacked_timer->start("rank23");
    // auto child_report0 = reporter.addReport("my report");
    // child_report0.set("value0", 1.5);
    stacked_timer->stop("rank23");
  }

  stacked_timer->stop("Mini-EM");

  reporter.setReportingEnabled(false);

  reporter.applyMpiReductions(comm);

  auto data = reporter.getData();
  if (comm->getRank() == 0) {
    checkConsistency(data.sublist("Mini-EM").sublist("timing"), out, success);
    checkConsistency(data.sublist("Mini-EM").sublist("child").sublist("timing"), out, success);
    checkConsistency(data.sublist("Mini-EM").sublist("rank0").sublist("timing"), out, success);
    if (comm->getSize() >= 2)
      checkConsistency(data.sublist("Mini-EM").sublist("rank1").sublist("timing"), out, success);
    if (comm->getSize() >= 3)
      checkConsistency(data.sublist("Mini-EM").sublist("rank23").sublist("timing"), out, success);

    // std::cout << std::endl << data << std::endl;
    reporter.setPrintFormatting("timing", "@indentation@@name@: @time mean@ - [@calls mean@] {min=@time min@, max=@time max@, std dev=@time stddev@}\n");o
    reporter.print(out);
  }
}

} // namespace Teuchos
