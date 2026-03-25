// @HEADER
// *****************************************************************************
//                    Teuchos: Common Tools Package
//
// Copyright 2004 NTESS and the Teuchos contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#include "Teuchos_Assert.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_RCPStdSharedPtrConversions.hpp"
#include "Teuchos_UnitTestHarness.hpp"
#include <memory>

namespace {

class Base {
public:
  Base(int a) { a_ = a; }

  int value() const { return a_; }

private:
  int a_;
};
} // namespace

namespace Teuchos {

template <class T> class FakeRCP : public std::shared_ptr<T> {
public:

  bool is_null() const { return this->get() == 0; }

  T *getRawPtr() const { return this->get(); }

  Ptr<T> ptr() const { return Ptr<T>(getRawPtr()); }

  Ptr<T> operator()() const { return ptr(); }

  FakeRCP<T> &operator=(const std::shared_ptr<T> &r_ptr) {
    auto temp = FakeRCP(r_ptr);
    *this = temp;
    return *this;
  }
};

template <class T> FakeRCP<T> fake_rcp(T *ptr) {
  auto sp = std::shared_ptr<T>(ptr);
  return FakeRCP<T>(sp);
}
} // namespace Teuchos

namespace {

TEUCHOS_UNIT_TEST(SharedPtr, Basic) {
  auto a = std::make_shared<Base>(3);
  // auto a = std::shared_ptr<Base>(new Base(3));
  TEUCHOS_ASSERT_EQUALITY(a->value(), 3);

  TEUCHOS_ASSERT_EQUALITY(a.use_count(), 1);

  Teuchos::RCP<Base> b = Teuchos::rcp(a);
  TEUCHOS_ASSERT_EQUALITY(b->value(), 3);
  TEUCHOS_ASSERT_EQUALITY(a.use_count(), 2);

  Teuchos::FakeRCP<Base> c(a);
  TEUCHOS_ASSERT_EQUALITY(c->value(), 3);
  TEUCHOS_ASSERT_EQUALITY(a.use_count(), 3);

  c = a;
  TEUCHOS_ASSERT_EQUALITY(c->value(), 3);
  TEUCHOS_ASSERT_EQUALITY(a.use_count(), 3);

  auto d = Teuchos::fake_rcp(new Base(4));
  TEUCHOS_ASSERT_EQUALITY(d->value(), 4);
  TEUCHOS_ASSERT_EQUALITY(d.use_count(), 1);

  // c.swap(a);
  // c.swap(c);
}

} // namespace
