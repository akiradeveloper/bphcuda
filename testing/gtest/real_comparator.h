#pragma once

#include <bphcuda/real_comparator.h>

#include <thrusting/real.h>

#include <gtest/gtest.h>

namespace {
  using namespace thrusting;
  using namespace bphcuda;
}

TEST(RealComparator, Real){
  real x = 10;
  real y = 11;
  EXPECT_TRUE(make_real_comparator(0.1)(x, y));
  EXPECT_FALSE(make_real_comparator(0)(x, y));
}

TEST(RealComparator, Real3){
  real3 x(10,20,30);
  real3 y(11,22,33);
  EXPECT_TRUE(make_real3_comparator(0.1)(x, y));
}
