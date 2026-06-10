// @HEADER
// *****************************************************************************
//                    Teuchos: Common Tools Package
//
// Copyright 2004 NTESS and the Teuchos contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#include <sstream>
#include "Teuchos_Reporter.hpp"
#include "Teuchos_UnitTestHarness.hpp"

namespace Teuchos {


TEUCHOS_UNIT_TEST( Reporter, test )
{
  Reporter reporter("main");
  reporter.setReportingEnabled(true);
  auto report1 = reporter.addReport("my report");
  report1->set("value0", 3.5);

  auto report2 = reporter.addReport("my report");
  report2->set("value0", 2.5);

  auto child0 = reporter.getChildReporter("my child");
  auto child_report0 = child0.addReport("my report");
  child_report0->set("value0", 1.5);
  // std::cout << *child0.getData() << std::endl;

  std::cout << *reporter.getData() << std::endl;

}



} // namespace Teuchos
