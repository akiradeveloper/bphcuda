#include <thrusting/vector.h>

#include <gtest/gtest.h>

namespace {
  using namespace thrusting;
}

TEST(IteratorEqual, Test){
  typedef typename vector<int>::type::iterator It1;
  typedef typename vector<int>::type::iterator It2;
  bool b = (typeid(It1)==typeid(It2));
  EXPECT_TRUE(b);
}
