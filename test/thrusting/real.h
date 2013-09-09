#pragma once

#include <thrusting/real.h>
#include <thrusting/tuple.h>
#include <thrusting/vectorspace.h>

#include <gtest/gtest.h>

namespace {
  using namespace thrusting;
}

TEST(Real, Create){
  real x = 1.0;
  EXPECT_EQ(real(1.0), x);
}

TEST(Real, TupleCreate){
  real7 t = real7(1,2,3,4,5,6,7);
  std::cout << t << std::endl;
  EXPECT_EQ(thrusting::make_tuple7<real>(1,2,3,4,5,6,7), t);
}

TEST(Real, SimpleArithmatic){
  EXPECT_EQ(real2(2.0, 4.0), 2 * real2(1.0, 2.0));
}
