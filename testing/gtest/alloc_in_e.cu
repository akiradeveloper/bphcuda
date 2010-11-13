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
  size_t n_particle = 1;
  real _xs[] = {1}; THRUSTING_VECTOR<real> xs(_xs, _xs+n_particle);
  real _ys[] = {1}; THRUSTING_VECTOR<real> ys(_ys, _ys+n_particle);
  real _zs[] = {1}; THRUSTING_VECTOR<real> zs(_zs, _zs+n_particle);
  THRUSTING_VECTOR<real> in_e(n_particle);
  bphcuda::alloc_in_e(
    n_particle,
    xs.begin(),
    ys.begin(),
    zs.begin(),
    thrust::constant_iterator<real>(2), // m
    in_e.begin(),
    thrust::constant_iterator<real>(2)); // s

   EXPECT_EQ(2.0, thrusting::iterator_value_at(0, in_e.begin()));
}
