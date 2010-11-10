#include <thrusting/dtype/real.h>
#include <thrusting/dtype/tuple/real.h>
#include <thrusting/tuple.h>

#include <bphcuda/cell.h>

#include <gtest/gtest.h>

TEST(cell, calc_idx1){
  Cell c = bphcuda::make_cell(
    thrusting::make_real3(0,0,0),
    thrusting::make_real3(1,1,1),
    thrusting::make_tuple3<size_t>(2,2,2));
  
  EXPECT_EQ(bphcuda::calc_idx1(c, thrusting::make_real3(1.5,1.5,1.5)), 7);
  EXPECT_EQ(bphcuda::calc_idx1(c, thrusting::make_real3(0.5,0.5,0.5)), 0);
}
