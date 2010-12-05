#include <thrusting/real.h>
#include <thrusting/tuple.h>

#include <bphcuda/cell.h>

#include <gtest/gtest.h>

namespace {
  using namespace thrusting;
  using namespace bphcuda;
}

TEST(Cell, CalcIdx1){
  cell c = make_cell(
    real3(0,0,0),
    real3(1,1,1),
    thrusting::make_tuple3<size_t>(2,2,2));
  
  EXPECT_EQ(7, make_cellidx1_calculator(c)(real3(1.5, 1.5, 1.5)));
  EXPECT_EQ(0, make_cellidx1_calculator(c)(real3(0.5, 0.5, 0.5)));
}

TEST(Cell, MinMax){
  cell c = make_cell(
    real3(1,2,3),
    real3(1,1,1),
    make_tuple3<size_t>(2,3,4));
  
  EXPECT_EQ(1, c.x_min());
  EXPECT_EQ(2, c.y_min());
  EXPECT_EQ(3, c.z_min());

  EXPECT_EQ(3, c.x_max());
  EXPECT_EQ(5, c.y_max());
  EXPECT_EQ(7, c.z_max());
}
