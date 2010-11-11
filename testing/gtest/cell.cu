#include <thrusting/dtype/real.h>
#include <thrusting/dtype/tuple/real.h>
#include <thrusting/tuple.h>

#include <bphcuda/cell.h>

#include <gtest/gtest.h>

namespace {
  using thrusting::real3;
}

TEST(cell, calc_idx1){
  bphcuda::cell c = bphcuda::make_cell(
    real3(0,0,0),
    real3(1,1,1),
    thrusting::make_tuple3<size_t>(2,2,2));
  
  EXPECT_EQ(bphcuda::calc_idx1(c, real3(1.5,1.5,1.5)), 7);
  EXPECT_EQ(bphcuda::calc_idx1(c, real3(0.5,0.5,0.5)), 0);
}
