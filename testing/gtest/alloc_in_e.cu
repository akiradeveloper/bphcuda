#include <thrust/iterator/constant_iterator.h>

#include <thrusting/dtype/real.h>
#include <thrusting/vector.h>
#include <thrusting/iterator.h>

#include <bphcuda/alloc_in_e.h>

#include <gtest/gtest.h>

namespace {
  using thrusting::real;
}

TEST(alloc_in_e, test1){
  real _xs[] = {1}; THRUSTING_VECTOR<real> xs(_xs, _xs+1);
  real _ys[] = {1}; THRUSTING_VECTOR<real> ys(_ys, _ys+1);
  real _zs[] = {1}; THRUSTING_VECTOR<real> zs(_zs, _zs+1);
  THRUSTING_VECTOR<real> in_e(1);
  bphcuda::alloc_in_e(
    1,
    xs.begin(),
    ys.begin(),
    zs.begin(),
    thrust::make_constant_iterator<real>(2),
    in_e.begin(),
    thrust::make_constant_iterator<real>(2)); 
  EXPECT_EQ(2.0, thrusting::iterator_value_at(0, in_e.begin()));
}
